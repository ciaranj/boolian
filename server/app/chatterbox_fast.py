"""A faster speech-token loop for Chatterbox-Turbo.

Same model, precision (fp32) and sampling as ChatterboxTurboTTS.generate() (temperature 0.8,
top-k 1000, top-p 0.95, repetition penalty 1.2, applied in that order). Measured on an RTX 4060 Ti,
the original spends 80-90% of each sentence generating speech tokens at ~50/s (25 tokens = 1 s of
speech), and most of each 20 ms step is Python/Hugging Face overhead launching ~hundreds of small
GPU kernels one by one, not GPU work. So:
- GraphDecoder runs the GPT-2 with a fixed-size KV cache and records one decoding step as a CUDA
  graph, replayed per token as a single launch. Same weights and maths; it reuses each layer's own
  modules and only replaces Hugging Face's growing cache. There is one graph per attention span
  (SPANS), and each token uses the smallest that covers it: with only the full-size cache, every
  token read ~0.3 GB of masked-out cache on top of the 1.24 GB of weights;
- sampling works on the top-k slice, instead of sorting the whole vocabulary every step;
- the stop token is checked every few steps, so the CPU doesn't wait on the GPU after every token.

The audio decoder (S3Gen) has the same problem: its two flow-matching steps were ~94 ms of a
~130 ms decode, with the GPU busy for under a third of it (~5,500 kernel launches per sentence).
FlowGraphs records the flow network as CUDA graphs for padded lengths. Padding is exact for this
network: causal convolutions, per-frame LayerNorm, no resampling along time (one channel level),
and attention that masks padded frames out completely.

(bf16 was tried: no faster, and audibly worse, so everything stays in fp32.)
"""

from __future__ import annotations

import contextlib
import logging

import torch
import torch.nn.functional as F

log = logging.getLogger("spectre.chatterbox")

STOP_CHECK_EVERY = 8  # steps between CPU checks for the stop token (up to 7 wasted steps per sentence)
OOV_FROM = 6561  # speech tokens at or above this (start/stop) aren't audio, as in the original
# Attention spans with a recorded graph. The prompt (voice + text) is ~440 positions, so sentences
# start in the 512 graph; the largest leaves room for ~1000 speech tokens, the original's limit.
SPANS = (512, 640, 768, 896, 1024, 1280, 1536)

_graph_broken = False  # set if capture ever fails; we then fall back to the plain loop
FLOW_STEP = 64  # flow graphs are recorded for lengths (mel frames) in steps of this
FLOW_MAX_SPEECH_S = 15  # sentences up to this long (plus the reference clip) get a recorded graph


def prepare(model) -> None:
    """Call once after loading: put the watermarker on the GPU.

    Chatterbox builds its (Resemble Perth) watermarker on the CPU, where it runs a neural network
    over every sentence. Moving it to the GPU keeps the watermark, it just stops costing CPU time."""
    global _graph_broken
    if str(model.device).startswith("cuda"):
        watermarker_to(model, model.device)
        try:
            with torch.inference_mode():
                _decoder(model.t3).capture_all()
        except Exception:
            log.exception("couldn't capture CUDA graphs; using the plain loop")
            _graph_broken = True
        # FlowGraphs (the audio decoder as CUDA graphs) isn't installed here yet: in bench-chatterbox
        # --decoder it changed one short sentence's audio (largest sample difference 0.02 against a 0.31
        # peak) while matching longer ones exactly. Until that's understood, the decoder runs as before.


def watermarker_to(model, device) -> None:
    import perth

    model.watermarker = perth.PerthImplicitWatermarker(device=device)


class GraphDecoder:
    """GPT-2 decoding for T3 with a static KV cache; each token is one CUDA graph replay."""

    def __init__(self, t3, spans: tuple[int, ...] = SPANS):
        tf = t3.tfmr
        cfg = tf.config
        # The attention below is plain scaled dot-product; refuse configs where GPT-2 does anything else.
        assert cfg.scale_attn_weights and not cfg.scale_attn_by_inverse_layer_idx and not cfg.add_cross_attention
        self.t3, self.tf = t3, tf
        self.width = cfg.n_embd
        self.heads = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.scale = self.head_dim**-0.5
        self.spans = spans
        self.max_len = max_len = spans[-1]
        dev, dt = t3.device, tf.dtype
        shape = (len(tf.h), 1, self.heads, max_len, self.head_dim)
        self.k = torch.zeros(shape, device=dev, dtype=dt)
        self.v = torch.zeros(shape, device=dev, dtype=dt)
        self.slots = torch.arange(max_len, device=dev)
        self.tok = torch.zeros((1, 1), dtype=torch.long, device=dev)
        self.pos = torch.zeros((1,), dtype=torch.long, device=dev)  # where the next token goes
        self.n = 0  # the same, on the CPU, to pick a span without waiting on the GPU
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}

    def _layers(self, x, positions, span: int | None):
        """x (1, T, C) at cache slots `positions` (T,); span=None for the prompt. Returns last-position logits."""
        mask = None if span is None else (self.slots[:span] <= positions[0]).view(1, 1, 1, -1)
        for i, blk in enumerate(self.tf.h):
            q, k, v = blk.attn.c_attn(blk.ln_1(x)).split(self.width, dim=2)
            q, k, v = (t.view(1, -1, self.heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
            self.k[i].index_copy_(2, positions, k)
            self.v[i].index_copy_(2, positions, v)
            if span is None:  # the prompt only attends to itself, causally (no stale slots involved)
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
            else:
                ks, vs = self.k[i][:, :, :span], self.v[i][:, :, :span]
                a = F.scaled_dot_product_attention(q, ks, vs, attn_mask=mask, scale=self.scale)
            x = x + blk.attn.c_proj(a.transpose(1, 2).reshape(1, -1, self.width))
            x = x + blk.mlp(blk.ln_2(x))
        return self.t3.speech_head(self.tf.ln_f(x[:, -1]))

    def prefill(self, embeds):
        """Run the prompt (1, T, C); returns logits (1, V) for the first speech token."""
        n = embeds.shape[1]
        positions = self.slots[:n]
        logits = self._layers(embeds + self.tf.wpe(positions), positions, span=None)
        self.pos.fill_(n)
        self.n = n
        return logits

    def _step(self, span: int):
        x = self.t3.speech_emb(self.tok) + self.tf.wpe(self.pos)
        return self._layers(x, self.pos, span)

    def span_for(self, n: int) -> int:
        """The smallest recorded span that includes slot n."""
        return next(s for s in self.spans if s > n)

    def step(self, tok):
        """Feed one token (0-dim, on the GPU); returns logits (1, V) for the next. Valid until the next call."""
        self.tok.copy_(tok.view(1, 1))
        graph, logits = self.graph_for(self.span_for(self.n))
        graph.replay()
        self.pos += 1
        self.n += 1
        return logits

    def graph_for(self, span: int) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
        if span not in self.graphs:
            self._capture(span)
        return self.graphs[span]

    def capture_all(self) -> None:
        """Record every span up front, between sentences: capture writes cache slot `pos`, so start from
        slot 0, which the next prompt overwrites anyway."""
        self.pos.zero_()
        self.n = 0
        for span in self.spans:
            self.graph_for(span)
        log.info("Chatterbox decoding steps captured as CUDA graphs for spans %s", ", ".join(map(str, self.spans)))

    def _capture(self, span: int) -> None:
        # Warm up on a side stream (allocations, cuBLAS handles), then record. Warm-up writes the same
        # cache slot the replay will, with the same values, so it doesn't disturb a sentence in progress.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._step(span)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):  # Whisper may load in another thread
            logits = self._step(span)
        self.graphs[span] = (graph, logits)


@contextlib.contextmanager
def _masks_without_sync():
    """S3Gen's add_optional_chunk_mask ends with a `.item()` check that would break graph capture.
    With no chunking (this model's configuration) it returns its input mask, and the check can't fire
    for masks with any real frame, so while recording, use a version that just returns the mask."""
    import chatterbox.models.s3gen.decoder as dec

    orig = dec.add_optional_chunk_mask

    def no_sync(xs, masks, use_dynamic_chunk, use_dynamic_left_chunk, decoding_chunk_size, static_chunk_size,
                *rest, **kw):
        if use_dynamic_chunk or static_chunk_size > 0:
            raise RuntimeError("chunked masks can't be recorded")
        return masks

    dec.add_optional_chunk_mask = no_sync
    try:
        yield
    finally:
        dec.add_optional_chunk_mask = orig


class FlowGraphs:
    """S3Gen's flow-matching network (the 'estimator') as CUDA graphs over padded lengths."""

    def __init__(self, estimator, lengths):
        self.est = estimator
        self.eager = estimator.forward  # the original bound method
        self.enabled = True
        self.pool = torch.cuda.graph_pool_handle()  # shared: graphs never run at once, outputs are copied out
        self.graphs: dict[int, tuple] = {}
        for length in lengths:
            self._capture(length)

    @classmethod
    def install(cls, model) -> FlowGraphs:
        est = model.s3gen.flow.decoder.estimator
        if isinstance(getattr(est, "_boolian_flow", None), FlowGraphs):
            return est._boolian_flow
        prompt = model.conds.gen["prompt_feat"].shape[1]  # mel frames of the reference clip
        first = -(-(prompt + 20) // FLOW_STEP) * FLOW_STEP
        last = -(-(prompt + FLOW_MAX_SPEECH_S * 50) // FLOW_STEP) * FLOW_STEP  # 50 mel frames per second
        flow = cls(est, range(first, last + 1, FLOW_STEP))
        est._boolian_flow = flow
        est.forward = flow.forward
        log.info("audio decoder captured as CUDA graphs for %d-%d mel frames", first, last)
        return flow

    def _capture(self, length: int) -> None:
        dev, dt = self.est.final_proj.weight.device, self.est.dtype
        buf = {
            "x": torch.zeros(1, 80, length, device=dev, dtype=dt),
            "mask": torch.ones(1, 1, length, device=dev, dtype=dt),
            "mu": torch.zeros(1, 80, length, device=dev, dtype=dt),
            "t": torch.zeros(1, device=dev, dtype=dt),
            "spks": torch.zeros(1, 80, device=dev, dtype=dt),
            "cond": torch.zeros(1, 80, length, device=dev, dtype=dt),
            "r": torch.zeros(1, device=dev, dtype=dt),
        }
        with _masks_without_sync():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self.eager(**buf)
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
                out = self.eager(**buf)
        self.graphs[length] = (graph, buf, out)

    def forward(self, x, mask, mu, t, spks=None, cond=None, r=None):
        n = x.shape[-1]
        length = -(-n // FLOW_STEP) * FLOW_STEP
        if (not self.enabled or x.shape[0] != 1 or spks is None or cond is None or r is None
                or length not in self.graphs):
            return self.eager(x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, r=r)
        graph, buf, out = self.graphs[length]
        for name, value in (("x", x), ("mask", mask), ("mu", mu), ("cond", cond)):
            buf[name][..., :n].copy_(value)
            buf[name][..., n:].zero_()  # padded frames: masked out, and zero so nothing odd can leak
        buf["t"].copy_(t.reshape(1))
        buf["r"].copy_(r.reshape(1))
        buf["spks"].copy_(spks)
        graph.replay()
        return out[..., :n].clone()


def _decoder(t3) -> GraphDecoder:
    dec = getattr(t3, "_boolian_graph_decoder", None)
    if dec is None:
        dec = t3._boolian_graph_decoder = GraphDecoder(t3)
    return dec


def filtered(logits, seen, temperature: float, top_k: int, top_p: float, penalty: float):
    """The original's logits processors, on just the top-k slice: returns (scores, vocab indices)."""
    vals, idx = torch.topk(logits / temperature, top_k)  # sorted, largest first
    probs = vals.softmax(-1)
    keep = (probs.cumsum(-1) - probs) < top_p  # HF TopP: drop the tail holding <= 1 - top_p of the mass
    vals = vals.masked_fill(~keep, float("-inf"))
    rep = seen[idx]  # HF RepetitionPenalty, applied last as in the original list
    vals = torch.where(rep, torch.where(vals < 0, vals * penalty, vals / penalty), vals)
    return vals, idx


def _sample(logits, seen, temperature: float, top_k: int, top_p: float, penalty: float):
    """One token from a (V,) logits vector, without leaving the GPU."""
    vals, idx = filtered(logits, seen, temperature, top_k, top_p, penalty)
    return idx[torch.multinomial(vals.softmax(-1), 1)].squeeze(0)


@torch.inference_mode()
def speech_tokens(t3, t3_cond, text_tokens, graph: bool = True, temperature=0.8, top_k=1000, top_p=0.95,
                  repetition_penalty=1.2, max_gen_len=1000):
    """Speech tokens for one sentence. graph=False uses Hugging Face's forward pass (the plain loop)."""
    hp = t3.hp
    device = text_tokens.device
    vocab = t3.speech_head.out_features
    top_k = min(top_k, vocab)

    start = torch.full_like(text_tokens[:, :1], hp.start_speech_token)
    embeds, _ = t3.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=start, cfg_weight=0.0)
    if graph:
        dec = _decoder(t3)
        logits = dec.prefill(embeds)
        step = dec.step
        max_gen_len = min(max_gen_len, dec.max_len - embeds.shape[1] - 1)
    else:
        out = t3.tfmr(inputs_embeds=embeds, use_cache=True)
        cache = {"past": out.past_key_values}
        logits = t3.speech_head(out[0][:, -1])

        def step(tok):
            o = t3.tfmr(inputs_embeds=t3.speech_emb(tok.view(1, 1)), past_key_values=cache["past"], use_cache=True)
            cache["past"] = o.past_key_values
            return t3.speech_head(o[0][:, -1])

    # The original penalises the start token for the first pick only, then just the generated tokens.
    seen = torch.zeros(vocab, dtype=torch.bool, device=device)
    seen[hp.start_speech_token] = True
    tokens = torch.empty(max_gen_len + 1, dtype=torch.long, device=device)
    n = 0
    for i in range(max_gen_len + 1):
        tok = _sample(logits.float()[0], seen, temperature, top_k, top_p, repetition_penalty)
        if i == 0:
            seen[hp.start_speech_token] = False
        tokens[i] = tok
        seen[tok] = True
        n = i + 1
        if n % STOP_CHECK_EVERY == 0 and bool((tokens[:n] == hp.stop_speech_token).any()):
            break
        logits = step(tok)

    toks = tokens[:n]
    stops = (toks == hp.stop_speech_token).nonzero()
    if len(stops):
        toks = toks[: int(stops[0, 0])]
    return toks


@torch.inference_mode()
def generate(model, text: str, graph: bool | None = None):
    """Drop-in for ChatterboxTurboTTS.generate(text) with the default settings; returns a float numpy array."""
    global _graph_broken
    from chatterbox.models.s3gen.const import S3GEN_SIL
    from chatterbox.tts_turbo import punc_norm

    text_tokens = model.tokenizer(punc_norm(text), return_tensors="pt", padding=True, truncation=True).input_ids
    text_tokens = text_tokens.to(model.device)
    if graph is None:
        graph = str(model.device).startswith("cuda") and not _graph_broken
    try:
        toks = speech_tokens(model.t3, model.conds.t3, text_tokens, graph=graph)
    except Exception:
        if not graph:
            raise
        log.exception("CUDA graph decoding failed; using the plain loop from now on")
        _graph_broken = True
        toks = speech_tokens(model.t3, model.conds.t3, text_tokens, graph=False)
    toks = toks[toks < OOV_FROM]
    toks = torch.cat([toks, torch.full((3,), S3GEN_SIL, dtype=toks.dtype, device=toks.device)])
    wav, _ = model.s3gen.inference(speech_tokens=toks, ref_dict=model.conds.gen, n_cfm_timesteps=2)
    wav = wav.squeeze(0).detach().cpu().numpy()
    return model.watermarker.apply_watermark(wav, sample_rate=model.sr)
