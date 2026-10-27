"""Chatterbox-Turbo speed check: Chatterbox's own token loop vs ours (app/chatterbox_fast.py),
each on its own and while the LLM is busy generating (the two containers share the GPU).

Run via `scripts/spectre-container.sh bench-chatterbox` (add --quick for just the correctness checks
and the per-token cost breakdown, or --decoder for where the audio decoder's time goes).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import sys
import time
import wave
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
os.environ.setdefault("TQDM_DISABLE", "1")

import torch  # noqa: E402

from app import chatterbox_fast  # noqa: E402
from app.config import load_config  # noqa: E402
from app.llm import OpenAICompatLLM  # noqa: E402

LINES = [
    "A witch!",
    "Hello there! I'm Boolian, the friendliest ghost in the whole town.",
    "I have been drifting around these halls for a hundred and seventy three years. [chuckle] Time flies!",
]


def check_sampler() -> bool:
    """Our filtering must keep exactly the tokens HF's processors keep, with the same scores."""
    from transformers import (LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper,
                              TopKLogitsWarper, TopPLogitsWarper)

    procs = LogitsProcessorList([TemperatureLogitsWarper(0.8), TopKLogitsWarper(1000), TopPLogitsWarper(0.95),
                                 RepetitionPenaltyLogitsProcessor(1.2)])
    g = torch.Generator().manual_seed(0)
    worst = 0.0
    for trial in range(50):
        logits = torch.randn(6563, generator=g) * (1 + trial / 10)  # from flat to very peaked
        history = torch.randint(0, 6561, (1, 40), generator=g)
        seen = torch.zeros(6563, dtype=torch.bool)
        seen[history[0]] = True
        ref = procs(history, logits.clone().unsqueeze(0))[0]
        vals, idx = chatterbox_fast.filtered(logits, seen, 0.8, 1000, 0.95, 1.2)
        ours = torch.full_like(ref, float("-inf"))
        ours[idx] = vals
        if not torch.equal(torch.isfinite(ref), torch.isfinite(ours)):
            print(f"  trial {trial}: kept {int(torch.isfinite(ours).sum())} tokens, HF kept {int(torch.isfinite(ref).sum())}")
            return False
        finite = torch.isfinite(ref)
        worst = max(worst, float((ref[finite] - ours[finite]).abs().max()))
    print(f"  identical token sets in 50 trials, largest score difference {worst:.2e}")
    return worst < 1e-4


def save(path: str, samples, sr: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def gpu_state() -> str:
    """Performance state, clocks and power right now: a card idling in a low P-state is slow to ramp up."""
    q = "pstate,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,utilization.gpu,power.draw,power.limit"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        p, sm, sm_max, mem, mem_max, util, pw, pw_max = (x.strip() for x in out.split(","))
        return f"{p}, SM {sm} of {sm_max}, memory {mem} of {mem_max}, util {util}, power {pw} of {pw_max}"
    except Exception as e:
        return f"nvidia-smi failed: {e!r}"


@contextlib.asynccontextmanager
async def llm_busy(llm: OpenAICompatLLM | None):
    """Keep the LLM generating (sharing the GPU) for the duration; does nothing if llm is None."""
    stop = asyncio.Event()

    async def keep_busy():
        prompt = [{"role": "user", "content": "Tell me a very long story about a lighthouse keeper."}]
        while not stop.is_set():
            async for _ in llm.stream(prompt, max_tokens=400):
                if stop.is_set():
                    break

    task = asyncio.create_task(keep_busy()) if llm else None
    if task:
        await asyncio.sleep(0.5)  # let the LLM get going
    try:
        yield
    finally:
        stop.set()
        if task:
            task.cancel()


@torch.inference_mode()
def per_token_cost(model, label: str) -> None:
    """Time each part of one decoding step in isolation, 100 times over, with the GPU synchronised.

    Graph replay is pure GPU work for one token (no Python, no sampling); dividing the weights it
    has to read by that time gives the memory bandwidth actually achieved, to compare with the card's."""
    t3 = model.t3
    from chatterbox.tts_turbo import punc_norm

    text = model.tokenizer(punc_norm(LINES[1]), return_tensors="pt").input_ids.to(model.device)
    start = torch.full_like(text[:, :1], t3.hp.start_speech_token)
    embeds, _ = t3.prepare_input_embeds(t3_cond=model.conds.t3, text_tokens=text, speech_tokens=start, cfg_weight=0.0)
    tok = torch.tensor(42, device=model.device)
    dec = chatterbox_fast._decoder(t3)
    dec.prefill(embeds)
    dec.step(tok)
    span = dec.span_for(dec.n)
    graph, logits = dec.graph_for(span)
    full_graph, _ = dec.graph_for(dec.max_len)

    def ms_per_call(fn, n=100):
        fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n * 1000

    graph_ms = ms_per_call(graph.replay)
    state = gpu_state()  # sampled straight after sustained load
    full_ms = ms_per_call(full_graph.replay)
    logits, seen = logits[0], torch.zeros(logits.shape[-1], dtype=torch.bool, device=model.device)
    sample_ms = ms_per_call(lambda: chatterbox_fast._sample(logits, seen, 0.8, 1000, 0.95, 1.2))
    out = t3.tfmr(inputs_embeds=embeds, use_cache=True)
    past, emb = out.past_key_values, t3.speech_emb(tok.view(1, 1))
    hf_ms = ms_per_call(lambda: t3.speech_head(t3.tfmr(inputs_embeds=emb, past_key_values=past, use_cache=True)[0][:, -1]), 50)

    weights = [*t3.tfmr.h.parameters(), *t3.tfmr.ln_f.parameters(), *t3.speech_head.parameters()]
    gb = sum(p.numel() * p.element_size() for p in weights) / 1e9
    print(f"  {label}: GPU work per token (graph replay) {graph_ms:5.1f} ms attending to {span} positions "
          f"({full_ms:5.1f} ms to all {dec.max_len}), weights {gb:.2f} GB = {gb / graph_ms * 1000:4.0f} GB/s | "
          f"sampling {sample_ms:4.1f} ms | Hugging Face step {hf_ms:5.1f} ms")
    print(f"  {' ' * len(label)}  GPU during that: {state}")


async def timed(label: str, fn, model, busy: OpenAICompatLLM | None, tag: str) -> float:
    """Synthesise every line; with busy, keep the LLM generating the whole time."""
    total_audio = total_wall = 0.0
    async with llm_busy(busy):
        for i, line in enumerate(LINES, 1):
            t = time.monotonic()
            samples = await asyncio.to_thread(fn, model, line)
            wall = time.monotonic() - t
            secs = len(samples) / model.sr
            total_audio += secs
            total_wall += wall
            print(f"  {label}: {secs:4.1f}s of audio in {wall * 1000:5.0f} ms  ({line[:40]!r})")
            save(f"/tmp/boolian-bench-{tag}-{i}.wav", samples, model.sr)
    rate = total_audio / total_wall
    print(f"  => {rate:.1f}x real time")
    return rate


@torch.inference_mode()
def check_decoder(model) -> bool:
    """The CUDA-graph GPT-2 must predict what Hugging Face's does, for the same prompt and tokens."""
    from chatterbox.tts_turbo import punc_norm

    t3 = model.t3
    text = model.tokenizer(punc_norm(LINES[1]), return_tensors="pt").input_ids.to(model.device)
    start = torch.full_like(text[:, :1], t3.hp.start_speech_token)
    embeds, _ = t3.prepare_input_embeds(t3_cond=model.conds.t3, text_tokens=text, speech_tokens=start, cfg_weight=0.0)
    # Enough tokens to cross from one recorded attention span into the next (prompt ~400 + 200).
    forced = torch.randint(0, chatterbox_fast.OOV_FROM, (200,), device=model.device,
                           generator=torch.Generator(model.device).manual_seed(1))

    out = t3.tfmr(inputs_embeds=embeds, use_cache=True)
    past, ref = out.past_key_values, [t3.speech_head(out[0][:, -1])]
    for tok in forced:
        out = t3.tfmr(inputs_embeds=t3.speech_emb(tok.view(1, 1)), past_key_values=past, use_cache=True)
        past = out.past_key_values
        ref.append(t3.speech_head(out[0][:, -1]))

    dec = chatterbox_fast._decoder(t3)
    got = [dec.prefill(embeds).clone()]
    spans = [dec.span_for(dec.n)]
    for tok in forced:
        got.append(dec.step(tok).clone())
        spans.append(dec.span_for(dec.n))

    ref_t, got_t = torch.cat(ref), torch.cat(got)
    diff = float((ref_t - got_t).abs().max())
    scale = float(ref_t.abs().max())
    same_top = int((ref_t.argmax(-1) == got_t.argmax(-1)).sum())
    print(f"  {len(ref)} steps from a {embeds.shape[1]}-position prompt, through attention spans "
          f"{sorted(set(spans))}: largest logit difference {diff:.2e} (logits up to {scale:.1f}), "
          f"same top token {same_top}/{len(ref)}")
    return diff < 1e-2 * max(scale, 1.0) and same_top == len(ref)


def profile(model, label: str, graph: bool) -> None:
    """Where one sentence's time goes, stage by stage."""
    from chatterbox.models.s3gen.const import S3GEN_SIL
    from chatterbox.tts_turbo import punc_norm

    def now():
        torch.cuda.synchronize()
        return time.monotonic()

    for line in LINES:
        t0 = now()
        text_tokens = model.tokenizer(punc_norm(line), return_tensors="pt", padding=True, truncation=True).input_ids
        text_tokens = text_tokens.to(model.device)
        t1 = now()
        toks = chatterbox_fast.speech_tokens(model.t3, model.conds.t3, text_tokens, graph=graph)
        t2 = now()
        toks = torch.cat([toks[toks < chatterbox_fast.OOV_FROM], torch.full((3,), S3GEN_SIL, device=toks.device)])
        with torch.inference_mode():
            wav, _ = model.s3gen.inference(speech_tokens=toks, ref_dict=model.conds.gen, n_cfm_timesteps=2)
        t3 = now()
        wav = wav.squeeze(0).detach().cpu().numpy()
        model.watermarker.apply_watermark(wav, sample_rate=model.sr)
        t4 = now()
        n = len(toks)
        print(f"  {label}: {len(wav) / model.sr:4.1f}s audio | tokenise {(t1 - t0) * 1000:4.0f} ms | "
              f"tokens {(t2 - t1) * 1000:5.0f} ms ({n} @ {n / (t2 - t1):.0f}/s) | "
              f"decoder {(t3 - t2) * 1000:4.0f} ms | watermark {(t4 - t3) * 1000:4.0f} ms")


@torch.inference_mode()
def decoder_profile(model) -> None:
    """The audio decoder (S3Gen: speech tokens -> waveform), as before and with recorded flow graphs."""
    from chatterbox.models.s3gen.const import S3GEN_SIL
    from chatterbox.tts_turbo import punc_norm

    s3, ref = model.s3gen, model.conds.gen
    flow = chatterbox_fast.FlowGraphs.install(model)
    sil = torch.full((3,), S3GEN_SIL, device=model.device)
    tokens = []
    for line in LINES:
        text = model.tokenizer(punc_norm(line), return_tensors="pt").input_ids.to(model.device)
        t = chatterbox_fast.speech_tokens(model.t3, model.conds.t3, text)
        tokens.append(torch.cat([t[t < chatterbox_fast.OOV_FROM], sil]))

    def decode(toks, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        return s3.inference(speech_tokens=toks, ref_dict=ref, n_cfm_timesteps=2)[0]

    # 1. Same tokens and random seed, with and without the graphs: the audio must match.
    print("  whole decoder, same tokens and seed: as before twice (is it even deterministic?), then with graphs")
    ok = True
    for toks in tokens:
        flow.enabled = False
        a = decode(toks, seed=7)
        a2 = decode(toks, seed=7)
        flow.enabled = True
        b = decode(toks, seed=7)
        base = float((a - a2).abs().max())
        d = (a - b).abs().flatten()
        diff, where = float(d.max()), int(d.argmax())
        differ = float((d > 1e-4).float().mean())
        peak = float(a.abs().max())
        ok &= a.shape == b.shape and diff < 1e-3 * max(peak, 1e-3)
        print(f"    {a.shape[-1] / model.sr:4.1f}s ({len(toks)} tokens): as before vs as before {base:.2e} | with graphs "
              f"{diff:.2e} (peak {peak:.2f}), largest at {where / d.numel():.0%} of the way through, "
              f"{differ:.1%} of samples differ by >1e-4")

    print("  flow network alone, identical random inputs: as before twice, then the graph")
    prompt = ref["prompt_feat"].shape[1]
    for toks in tokens:
        n = prompt + 2 * len(toks)  # mel frames: reference clip + 2 per speech token
        g = torch.Generator(model.device).manual_seed(0)
        dev, dt = model.device, s3.flow.decoder.estimator.dtype

        def rnd(*shape):
            return torch.randn(*shape, device=dev, dtype=dt, generator=g)

        inp = dict(x=rnd(1, 80, n), mask=torch.ones(1, 1, n, device=dev, dtype=dt), mu=rnd(1, 80, n),
                   t=torch.tensor([0.3], device=dev, dtype=dt), spks=rnd(1, 80), cond=rnd(1, 80, n),
                   r=torch.tensor([0.8], device=dev, dtype=dt))
        e1, e2 = flow.eager(**inp), flow.eager(**inp)
        flow.enabled = True
        gr = flow.forward(**inp)
        dd = (e1 - gr).abs().amax(dim=1).flatten()  # per frame
        length = -(-n // chatterbox_fast.FLOW_STEP) * chatterbox_fast.FLOW_STEP
        bad = (dd > 1e-4).nonzero().flatten()
        first_bad = f"first differing frame {int(bad[0])} of {n}" if len(bad) else "no frame differs by >1e-4"
        print(f"    {n} frames (graph for {length}{'' if length in flow.graphs else ', not recorded: ran as before'}): "
              f"as before vs as before {float((e1 - e2).abs().max()):.2e} | graph {float(dd.max()):.2e}, {first_bad}")

    # TF32: cuDNN may use it for convolutions by default, depending on the algorithm it picks for a
    # given length, so the eager (unpadded) and graph (padded) runs can round differently.
    print(f"  TF32 allowed: matmul {torch.backends.cuda.matmul.allow_tf32}, cuDNN {torch.backends.cudnn.allow_tf32}"
          f" | cuDNN benchmark mode {torch.backends.cudnn.benchmark}")
    est = s3.flow.decoder.estimator
    eager = type(est).forward.__get__(est)
    for toks in tokens:
        n = prompt + 2 * len(toks)
        length = -(-n // chatterbox_fast.FLOW_STEP) * chatterbox_fast.FLOW_STEP
        g = torch.Generator(model.device).manual_seed(0)
        dev, dt = model.device, est.dtype

        def rnd(*shape):
            return torch.randn(*shape, device=dev, dtype=dt, generator=g)

        inp = dict(x=rnd(1, 80, n), mask=torch.ones(1, 1, n, device=dev, dtype=dt), mu=rnd(1, 80, n),
                   t=torch.tensor([0.3], device=dev, dtype=dt), spks=rnd(1, 80), cond=rnd(1, 80, n),
                   r=torch.tensor([0.8], device=dev, dtype=dt))
        with_tf32 = eager(**inp)
        prev = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        try:
            exact = eager(**inp)
            strict = chatterbox_fast.FlowGraphs(est, [length])  # recorded with TF32 off
            graph_exact = strict.forward(**inp)
            del strict
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
        print(f"    {n} frames, TF32 off: as before vs graph {float((exact - graph_exact).abs().max()):.2e} | "
              f"for scale, as before with vs without TF32 {float((with_tf32 - exact).abs().max()):.2e}")

    # Listen for yourself: the short sentence, as before and with the graph (TF32 as normal).
    flow.enabled = False
    a = decode(tokens[0], seed=7)
    flow.enabled = True
    b = decode(tokens[0], seed=7)
    save("/tmp/boolian-decoder-as-before.wav", a.flatten().cpu().numpy(), model.sr)
    save("/tmp/boolian-decoder-graph.wav", b.flatten().cpu().numpy(), model.sr)
    print("  saved /tmp/boolian-decoder-as-before.wav and /tmp/boolian-decoder-graph.wav to compare by ear")

    if not ok:
        print("  FAILED: the flow graphs change the audio; not timing them")
        return

    # 2. Each stage, timed with the GPU synchronised either side, without and with the graphs.
    times: dict[str, float] = defaultdict(float)
    patched = []

    def time_method(obj, name, key):
        had_own = name in vars(obj)  # e.g. the estimator's forward, already pointed at FlowGraphs
        orig = getattr(obj, name)

        def wrapper(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            r = orig(*a, **k)
            torch.cuda.synchronize()
            times[key] += time.perf_counter() - t
            return r

        setattr(obj, name, wrapper)
        patched.append((obj, name, orig if had_own else None))

    time_method(s3.flow.encoder, "forward", "encoder")
    time_method(s3.flow.decoder.estimator, "forward", "flow steps")
    time_method(s3.mel2wav, "inference", "vocoder")
    for enabled in (False, True):
        flow.enabled = enabled
        label = "flow graphs" if enabled else "as before  "
        for toks in tokens:
            decode(toks)  # warm-up for this length
            times.clear()
            torch.cuda.synchronize()
            t = time.perf_counter()
            decode(toks)
            torch.cuda.synchronize()
            total = time.perf_counter() - t
            other = total - sum(times.values())
            print(f"  {label}: {len(toks) / 25:4.1f}s of speech: total {total * 1000:4.0f} ms = encoder "
                  f"{times['encoder'] * 1000:3.0f} + flow steps (2) {times['flow steps'] * 1000:3.0f} + vocoder "
                  f"{times['vocoder'] * 1000:3.0f} + other {other * 1000:3.0f}")
    for obj, name, orig in patched:
        if orig is None:
            delattr(obj, name)
        else:
            setattr(obj, name, orig)

    # 3. How much of the wall time the GPU is actually busy, for the shortest sentence.
    from torch.profiler import ProfilerActivity, profile

    for enabled in (False, True):
        flow.enabled = enabled
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            torch.cuda.synchronize()
            t = time.perf_counter()
            decode(tokens[0])
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
        kernels = [e for e in prof.events() if e.device_type.name == "CUDA"]
        busy_us = sum(e.time_range.elapsed_us() for e in kernels)  # one stream, so kernels don't overlap
        print(f"  {'flow graphs' if enabled else 'as before  '} under the profiler: {wall * 1000:.0f} ms wall, "
              f"GPU kernels busy {busy_us / 1000:.0f} ms across {len(kernels)} launches")
    flow.enabled = True


def original(model, text):
    return model.generate(text).squeeze(0).numpy()


async def main() -> int:
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    cfg = load_config()
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    print(f"[machine] {torch.cuda.get_device_name()}, {cores} CPU cores available, torch {torch.__version__}, "
          f"{torch.get_num_threads()} torch threads")
    print(f"  GPU at rest: {gpu_state()}")
    print("\n[sampler] ours vs Hugging Face's logits processors")
    if not check_sampler():
        print("FAILED: the fast sampler doesn't match; not benchmarking it")
        return 1

    print("\n[loading]")
    model = ChatterboxTurboTTS.from_pretrained("cuda")
    if cfg.tts_voice_ref:
        model.prepare_conditionals(cfg.tts_voice_ref)
    llm = OpenAICompatLLM(cfg)
    has_llm = await llm.healthy()
    if not has_llm:
        print(f"  LLM not reachable at {cfg.llm_url}: skipping the busy-GPU runs")

    quick = "--quick" in sys.argv[1:]
    if "--decoder" in sys.argv[1:]:
        chatterbox_fast.prepare(model)
        for warm in LINES[:2]:
            chatterbox_fast.generate(model, warm)
        print(f"\n[audio decoder] GPU at rest: {gpu_state()}")
        decoder_profile(model)
        await llm.aclose()
        return 0
    results = {}
    for warm in LINES[:2]:
        original(model, warm)
    if not quick:
        print("\n[original loop]")
        results["original"] = await timed("original", original, model, None, "original")
        if has_llm:
            print("\n[original loop, LLM busy]")
            results["original+llm"] = await timed("original", original, model, llm, "original-busy")

    chatterbox_fast.prepare(model)
    print("\n[CUDA graph decoder vs Hugging Face, same inputs]")
    if not check_decoder(model):
        print("FAILED: the graph decoder doesn't match Hugging Face's GPT-2; not benchmarking it")
        return 1
    print("\n[per-token cost, each part timed alone]")
    await asyncio.to_thread(per_token_cost, model, "GPU otherwise idle")
    if has_llm:
        async with llm_busy(llm):
            await asyncio.to_thread(per_token_cost, model, "LLM generating    ")
    if quick:
        await llm.aclose()
        return 0
    for warm in LINES[:2]:
        chatterbox_fast.generate(model, warm)
        chatterbox_fast.generate(model, warm, graph=False)
    print("\n[where the time goes]")
    profile(model, "plain loop ", graph=False)
    profile(model, "CUDA graph ", graph=True)
    print("\n[fast loop, CUDA graph]")
    results["fast"] = await timed("fast", chatterbox_fast.generate, model, None, "fast")
    if has_llm:
        print("\n[fast loop, LLM busy]")
        results["fast+llm"] = await timed("fast", chatterbox_fast.generate, model, llm, "fast-busy")
    await llm.aclose()

    print("\nSummary (x real time, higher is better):")
    for k, v in results.items():
        print(f"  {k:14} {v:5.1f}x")
    print("\nListen for any quality difference on the Mac:\n"
          "  scp 'root@boolian-spectre.home.arpa:/tmp/boolian-bench-*.wav' . && open boolian-bench-*.wav")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
