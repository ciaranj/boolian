"""A faster speech-token loop for Chatterbox-Turbo.

Same model and the same sampling as ChatterboxTurboTTS.generate() (temperature 0.8, top-k 1000,
top-p 0.95, repetition penalty 1.2, applied in that order), but:
- the transformer runs in bf16, halving the weights read for every token;
- sampling works on the top-k slice, instead of sorting the whole vocabulary every step;
- the stop token is checked every few steps, not after every token, so the CPU doesn't wait on
  the GPU each step and the GPU always has the next work queued.

Chatterbox's own loop manages ~70 tokens/s (25 tokens = 1 s of speech) on an RTX 4060 Ti.
"""

from __future__ import annotations

import torch

STOP_CHECK_EVERY = 8  # steps between CPU checks for the stop token (up to 7 wasted steps per sentence)
OOV_FROM = 6561  # speech tokens at or above this (start/stop) aren't audio, as in the original


def prepare(model) -> None:
    """Call once after loading: bf16 transformer, and the watermarker on the GPU.

    Chatterbox builds its (Resemble Perth) watermarker on the CPU, where it runs a neural network
    over every sentence. Moving it to the GPU keeps the watermark, it just stops costing CPU time."""
    if not str(model.device).startswith("cuda"):
        return
    tfmr = model.t3.tfmr
    if torch.cuda.is_bf16_supported() and tfmr.dtype != torch.bfloat16:
        tfmr.to(torch.bfloat16)
    watermarker_to(model, model.device)


def watermarker_to(model, device) -> None:
    import perth

    model.watermarker = perth.PerthImplicitWatermarker(device=device)


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
def speech_tokens(t3, t3_cond, text_tokens, temperature=0.8, top_k=1000, top_p=0.95, repetition_penalty=1.2,
                  max_gen_len=1000):
    hp = t3.hp
    dtype = t3.tfmr.dtype
    device = text_tokens.device
    head_dtype = t3.speech_head.weight.dtype
    vocab = t3.speech_head.out_features
    top_k = min(top_k, vocab)

    start = torch.full_like(text_tokens[:, :1], hp.start_speech_token)
    embeds, _ = t3.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=start, cfg_weight=0.0)
    out = t3.tfmr(inputs_embeds=embeds.to(dtype), use_cache=True)
    past, hidden = out.past_key_values, out[0][:, -1:]

    # The original penalises the start token for the first pick only, then just the generated tokens.
    seen = torch.zeros(vocab, dtype=torch.bool, device=device)
    seen[hp.start_speech_token] = True
    tokens = torch.empty(max_gen_len + 1, dtype=torch.long, device=device)
    n = 0
    for step in range(max_gen_len + 1):
        logits = t3.speech_head(hidden[:, -1].to(head_dtype)).float()[0]
        tok = _sample(logits, seen, temperature, top_k, top_p, repetition_penalty)
        if step == 0:
            seen[hp.start_speech_token] = False
        tokens[step] = tok
        seen[tok] = True
        n = step + 1
        if n % STOP_CHECK_EVERY == 0 and bool((tokens[:n] == hp.stop_speech_token).any()):
            break
        out = t3.tfmr(inputs_embeds=t3.speech_emb(tok.view(1, 1)).to(dtype), past_key_values=past, use_cache=True)
        past, hidden = out.past_key_values, out[0]

    toks = tokens[:n]
    stops = (toks == hp.stop_speech_token).nonzero()
    if len(stops):
        toks = toks[: int(stops[0, 0])]
    return toks


@torch.inference_mode()
def generate(model, text: str):
    """Drop-in for ChatterboxTurboTTS.generate(text) with the default settings; returns a float numpy array."""
    from chatterbox.models.s3gen.const import S3GEN_SIL
    from chatterbox.tts_turbo import punc_norm

    text_tokens = model.tokenizer(punc_norm(text), return_tensors="pt", padding=True, truncation=True).input_ids
    toks = speech_tokens(model.t3, model.conds.t3, text_tokens.to(model.device))
    toks = toks[toks < OOV_FROM]
    toks = torch.cat([toks, torch.full((3,), S3GEN_SIL, dtype=toks.dtype, device=toks.device)])
    wav, _ = model.s3gen.inference(speech_tokens=toks, ref_dict=model.conds.gen, n_cfm_timesteps=2)
    wav = wav.squeeze(0).detach().cpu().numpy()
    return model.watermarker.apply_watermark(wav, sample_rate=model.sr)
