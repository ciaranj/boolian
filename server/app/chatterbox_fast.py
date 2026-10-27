"""A faster speech-token loop for Chatterbox-Turbo.

Same model, precision (fp32) and sampling as ChatterboxTurboTTS.generate() (temperature 0.8,
top-k 1000, top-p 0.95, repetition penalty 1.2, applied in that order). Measured on an RTX 4060 Ti,
the original spends 80-90% of each sentence generating speech tokens at ~50/s (25 tokens = 1 s of
speech), and most of each 20 ms step is Python/Hugging Face overhead launching ~hundreds of small
GPU kernels one by one, not GPU work. So:
- GraphDecoder runs the GPT-2 with a fixed-size KV cache and records one decoding step as a CUDA
  graph, replayed per token as a single launch. Same weights and maths; it reuses each layer's own
  modules and only replaces Hugging Face's growing cache;
- sampling works on the top-k slice, instead of sorting the whole vocabulary every step;
- the stop token is checked every few steps, so the CPU doesn't wait on the GPU after every token.

(bf16 was tried: no faster, and audibly worse, so everything stays in fp32.)
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

log = logging.getLogger("spectre.chatterbox")

STOP_CHECK_EVERY = 8  # steps between CPU checks for the stop token (up to 7 wasted steps per sentence)
OOV_FROM = 6561  # speech tokens at or above this (start/stop) aren't audio, as in the original
MAX_LEN = 1536  # KV cache positions: ~440 of prompt (voice + text) leaves room for 1000 speech tokens

_graph_broken = False  # set if capture ever fails; we then fall back to the plain loop


def prepare(model) -> None:
    """Call once after loading: put the watermarker on the GPU.

    Chatterbox builds its (Resemble Perth) watermarker on the CPU, where it runs a neural network
    over every sentence. Moving it to the GPU keeps the watermark, it just stops costing CPU time."""
    if str(model.device).startswith("cuda"):
        watermarker_to(model, model.device)


def watermarker_to(model, device) -> None:
    import perth

    model.watermarker = perth.PerthImplicitWatermarker(device=device)


class GraphDecoder:
    """GPT-2 decoding for T3 with a static KV cache; each token is one CUDA graph replay."""

    def __init__(self, t3, max_len: int = MAX_LEN):
        tf = t3.tfmr
        cfg = tf.config
        # The attention below is plain scaled dot-product; refuse configs where GPT-2 does anything else.
        assert cfg.scale_attn_weights and not cfg.scale_attn_by_inverse_layer_idx and not cfg.add_cross_attention
        self.t3, self.tf = t3, tf
        self.width = cfg.n_embd
        self.heads = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.scale = self.head_dim**-0.5
        self.max_len = max_len
        dev, dt = t3.device, tf.dtype
        shape = (len(tf.h), 1, self.heads, max_len, self.head_dim)
        self.k = torch.zeros(shape, device=dev, dtype=dt)
        self.v = torch.zeros(shape, device=dev, dtype=dt)
        self.slots = torch.arange(max_len, device=dev)
        self.tok = torch.zeros((1, 1), dtype=torch.long, device=dev)
        self.pos = torch.zeros((1,), dtype=torch.long, device=dev)  # where the next token goes
        self.graph: torch.cuda.CUDAGraph | None = None
        self.logits: torch.Tensor | None = None

    def _layers(self, x, positions, prefill: bool):
        """x (1, T, C) at cache slots `positions` (T,). Returns speech logits for the last position."""
        mask = None if prefill else (self.slots <= positions[0]).view(1, 1, 1, -1)
        for i, blk in enumerate(self.tf.h):
            q, k, v = blk.attn.c_attn(blk.ln_1(x)).split(self.width, dim=2)
            q, k, v = (t.view(1, -1, self.heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
            self.k[i].index_copy_(2, positions, k)
            self.v[i].index_copy_(2, positions, v)
            if prefill:  # the prompt only attends to itself, causally (no stale slots involved)
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
            else:
                a = F.scaled_dot_product_attention(q, self.k[i], self.v[i], attn_mask=mask, scale=self.scale)
            x = x + blk.attn.c_proj(a.transpose(1, 2).reshape(1, -1, self.width))
            x = x + blk.mlp(blk.ln_2(x))
        return self.t3.speech_head(self.tf.ln_f(x[:, -1]))

    def prefill(self, embeds):
        """Run the prompt (1, T, C); returns logits (1, V) for the first speech token."""
        n = embeds.shape[1]
        positions = self.slots[:n]
        logits = self._layers(embeds + self.tf.wpe(positions), positions, prefill=True)
        self.pos.fill_(n)
        return logits

    def _step(self):
        x = self.t3.speech_emb(self.tok) + self.tf.wpe(self.pos)
        return self._layers(x, self.pos, prefill=False)

    def step(self, tok):
        """Feed one token (0-dim, on the GPU); returns logits (1, V) for the next. Valid until the next call."""
        self.tok.copy_(tok.view(1, 1))
        if self.graph is None:
            self._capture()
        self.graph.replay()
        self.pos += 1
        return self.logits

    def _capture(self):
        # Warm up on a side stream (allocations, cuBLAS handles), then record. Warm-up writes the same
        # cache slot the replay will, with the same values, so it doesn't disturb the sentence.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = self._step()
        log.info("Chatterbox decoding step captured as a CUDA graph")


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
