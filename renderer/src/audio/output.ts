/**
 * Schedules streamed PCM from the server, adds the ghostly voice effects, and reports
 * when audio is *actually* audible (including Bluetooth / output latency).
 */

import { mulberry32 } from "../ghost/anim";
import type { RudeNoise } from "../protocol";

export interface OutputEvents {
  onStart(gen: number): void;
  onEnd(gen: number, reason: "ended" | "cancelled"): void;
  /** A rude noise is reaching the visitor's ears now. */
  onNoise?(kind: RudeNoise, seconds: number): void;
}

const START_LEAD_S = 0.12; // jitter cushion before the first chunk of a reply

/** Ghostly voice effects, each 0 (off) .. 1. */
export interface VoiceFx {
  waver: number; // slow, uneven pitch wobble, like a voice drifting on the air
  flange: number; // a sweeping, whooshing comb filter
  echo: number; // repeats that darken as they fade, like calling down a well
  shadow: number; // a quiet second voice an octave below his
}

// Pitch shifter for the shadow voice: two read heads drift through a short delay line at the pitch ratio,
// each faded in and out with a sine window, half a window apart, so their seams cancel out.
const SHADOW_WORKLET = `
class Shadow extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(sampleRate);
    this.w = Math.round(sampleRate * 0.07);
    this.ratio = 0.5;
    this.pos = 0;
    this.phase = 0;
  }
  read(d) {
    let p = this.pos - d;
    while (p < 0) p += this.buf.length;
    const i = Math.floor(p), f = p - i;
    return this.buf[i % this.buf.length] * (1 - f) + this.buf[(i + 1) % this.buf.length] * f;
  }
  process(inputs, outputs) {
    const inp = inputs[0][0], out = outputs[0][0];
    if (!out) return true;
    const step = (1 - this.ratio) / this.w;
    for (let n = 0; n < out.length; n++) {
      this.buf[this.pos] = inp ? inp[n] : 0;
      const a = this.phase, b = (this.phase + 0.5) % 1;
      out[n] = this.read(a * this.w) * Math.sin(Math.PI * a) + this.read(b * this.w) * Math.sin(Math.PI * b);
      this.phase = (this.phase + step) % 1;
      this.pos = (this.pos + 1) % this.buf.length;
    }
    return true;
  }
}
registerProcessor("ghost-shadow", Shadow);
`;
const UNDERRUN_LEAD_S = 0.03;

export class AudioOut {
  ctx: AudioContext | null = null;
  /** Extra latency the browser can't see (Bluetooth speakers), in ms. Operator-tunable. */
  extraLatencyMs = 0;

  private master!: GainNode;
  private voiceIn!: GainNode;
  private wet!: GainNode;
  private fxNodes: { waver: GainNode[]; flange: GainNode; echo: GainNode; shadow: GainNode | null } | null = null;
  private dry: GainNode | null = null;
  private fx: VoiceFx = { waver: 0, flange: 0, echo: 0, shadow: 0 };
  private analyser!: AnalyserNode;
  private sampleBuf = new Float32Array(1024);
  private levels: { t: number; v: number }[] = [];

  private gen = -1;
  private rejectUpTo = -1;
  private nextTime = 0;
  private started = false;
  private ended = false;
  private sources = new Set<AudioBufferSourceNode>();
  private volume = 0.9;
  private muted = false;

  constructor(private events: OutputEvents) {}

  get unlocked() {
    return this.ctx?.state === "running";
  }

  /** Must be called from a user gesture (browser autoplay policy). */
  async unlock(): Promise<AudioContext> {
    if (!this.ctx) {
      this.ctx = new AudioContext({ latencyHint: "interactive" });
      let shadow = true;
      try {
        const url = URL.createObjectURL(new Blob([SHADOW_WORKLET], { type: "application/javascript" }));
        await this.ctx.audioWorklet.addModule(url);
        URL.revokeObjectURL(url);
      } catch (err) {
        console.warn("no shadow voice:", err);
        shadow = false;
      }
      this.buildGraph(this.ctx, shadow);
    }
    await this.ctx.resume();
    return this.ctx;
  }

  /** Seconds between "scheduled at ctx.currentTime" and "reaches the visitor's ears". */
  get latencyS(): number {
    const ctx = this.ctx;
    if (!ctx) return 0;
    return (ctx.outputLatency || ctx.baseLatency || 0) + this.extraLatencyMs / 1000;
  }

  private buildGraph(ctx: AudioContext, withShadow: boolean) {
    this.master = ctx.createGain();
    this.master.gain.value = this.volume;
    this.master.connect(ctx.destination);

    this.voiceIn = ctx.createGain();
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.voiceIn.connect(this.analyser); // dry tap drives the mouth, so reverb tails and echoes don't

    // Waver: the whole voice runs through a delay whose length is wobbled, which bends the pitch: a slow drift
    // sharp and flat plus a faster quaver, like a voice on the wind. At 0 it's a fixed 30 ms delay (inaudible).
    const waver = ctx.createDelay(0.1);
    waver.delayTime.value = 0.03;
    const waverDepths = [0.31, 4.7].map((hz, i) => {
      const lfo = ctx.createOscillator();
      lfo.frequency.value = hz;
      const depth = ctx.createGain();
      depth.gain.value = 0;
      lfo.connect(depth).connect(waver.delayTime);
      lfo.start(ctx.currentTime + i * 0.7);
      return depth;
    });
    const voice = ctx.createGain(); // the voice after the waver: everything below starts here
    this.voiceIn.connect(waver).connect(voice);

    const dry = ctx.createGain();
    dry.gain.value = 0.85;
    voice.connect(dry).connect(this.master);
    this.dry = dry;

    // Haunted-hallway reverb from a synthetic impulse response.
    const conv = ctx.createConvolver();
    conv.buffer = impulse(ctx, 2.6, 2.8);
    this.wet = ctx.createGain();
    this.wet.gain.value = 0.35;
    voice.connect(conv).connect(this.wet).connect(this.master);

    // Slow chorus wobble: a short modulated delay mixed in quietly.
    const delay = ctx.createDelay(0.05);
    delay.delayTime.value = 0.014;
    const lfo = ctx.createOscillator();
    lfo.frequency.value = 0.6;
    const depth = ctx.createGain();
    depth.gain.value = 0.004;
    lfo.connect(depth).connect(delay.delayTime);
    lfo.start();
    const chorus = ctx.createGain();
    chorus.gain.value = 0.3;
    voice.connect(delay).connect(chorus).connect(this.master);

    // Flanger: a very short delay swept up and down (about one sweep per sentence), fed back into itself.
    // It takes over from the dry voice as it's turned up, so the comb-filter whoosh isn't masked.
    const flangeDelay = ctx.createDelay(0.02);
    flangeDelay.delayTime.value = 0.0055;
    const sweep = ctx.createOscillator();
    sweep.frequency.value = 0.45;
    const sweepDepth = ctx.createGain();
    sweepDepth.gain.value = 0.0045;
    sweep.connect(sweepDepth).connect(flangeDelay.delayTime);
    sweep.start();
    const flangeFb = ctx.createGain();
    flangeFb.gain.value = 0.7;
    flangeDelay.connect(flangeFb).connect(flangeDelay);
    const flange = ctx.createGain();
    flange.gain.value = 0;
    voice.connect(flangeDelay).connect(flange).connect(this.master);

    // Echo: a third of a second between repeats, each one duller (low-passed) and quieter.
    const echoDelay = ctx.createDelay(1);
    echoDelay.delayTime.value = 0.33;
    const echoTone = ctx.createBiquadFilter();
    echoTone.type = "lowpass";
    echoTone.frequency.value = 2200;
    const echoFb = ctx.createGain();
    echoFb.gain.value = 0.45;
    echoDelay.connect(echoTone).connect(echoFb).connect(echoDelay);
    const echo = ctx.createGain();
    echo.gain.value = 0;
    voice.connect(echoDelay);
    echoTone.connect(echo);
    echo.connect(this.master);
    echo.connect(conv); // echoes ring into the hallway too

    // Shadow: an octave down, low-passed (it's a presence, not a second speaker), a beat behind, into the reverb.
    let shadow: GainNode | null = null;
    if (withShadow) {
      const shifter = new AudioWorkletNode(ctx, "ghost-shadow");
      const shadowTone = ctx.createBiquadFilter();
      shadowTone.type = "lowpass";
      shadowTone.frequency.value = 1400;
      const lag = ctx.createDelay(0.2);
      lag.delayTime.value = 0.04;
      shadow = ctx.createGain();
      shadow.gain.value = 0;
      voice.connect(shifter).connect(shadowTone).connect(lag).connect(shadow);
      shadow.connect(this.master);
      shadow.connect(conv);
    }
    this.fxNodes = { waver: waverDepths, flange, echo, shadow };
    this.setFx(this.fx);
  }

  setFx(fx: VoiceFx) {
    this.fx = { ...fx };
    const n = this.fxNodes;
    if (!n || !this.ctx) return;
    const t = this.ctx.currentTime;
    // Waver at full: about ±4% pitch from the slow drift (two-thirds of a semitone) and ±2.5% quaver.
    n.waver[0].gain.setTargetAtTime(0.02 * fx.waver, t, 0.05);
    n.waver[1].gain.setTargetAtTime(0.00085 * fx.waver, t, 0.05);
    n.flange.gain.setTargetAtTime(0.75 * fx.flange, t, 0.05);
    this.dry?.gain.setTargetAtTime(0.85 * (1 - 0.6 * fx.flange), t, 0.05);
    n.echo.gain.setTargetAtTime(0.5 * fx.echo, t, 0.05);
    n.shadow?.gain.setTargetAtTime(0.7 * fx.shadow, t, 0.05);
  }

  setVolume(v: number) {
    this.volume = v;
    this.applyGain();
  }

  setReverb(v: number) {
    if (this.wet) this.wet.gain.value = v;
  }

  setMuted(m: boolean) {
    this.muted = m;
    this.applyGain();
  }

  get isMuted() {
    return this.muted;
  }

  private applyGain() {
    if (this.master && this.ctx) this.master.gain.setTargetAtTime(this.muted ? 0 : this.volume, this.ctx.currentTime, 0.03);
  }

  begin(gen: number) {
    if (gen <= this.rejectUpTo || gen === this.gen) return;
    if (this.started) this.finish("cancelled");
    this.stopSources();
    this.gen = gen;
    this.started = false;
    this.ended = false;
    this.nextTime = 0;
  }

  enqueue(gen: number, sampleRate: number, pcm: Int16Array, noise: RudeNoise | null = null) {
    const ctx = this.ctx;
    if (!ctx || gen <= this.rejectUpTo || pcm.length === 0) return;
    if (gen !== this.gen) {
      if (gen < this.gen) return; // stale
      this.begin(gen); // speech_start went missing; recover
    }
    const buf = ctx.createBuffer(1, pcm.length, sampleRate);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.voiceIn);

    const now = ctx.currentTime;
    if (this.nextTime < now + UNDERRUN_LEAD_S) this.nextTime = now + (this.started ? UNDERRUN_LEAD_S : START_LEAD_S);
    src.start(this.nextTime);
    if (!this.started) {
      this.started = true;
      const delayMs = (this.nextTime - now + this.latencyS) * 1000;
      const g = gen;
      setTimeout(() => {
        if (this.gen === g && this.started) this.events.onStart(g);
      }, delayMs);
    }
    if (noise) {
      const g = gen;
      setTimeout(() => {
        if (this.gen === g && this.started) this.events.onNoise?.(noise, buf.duration);
      }, (this.nextTime - now + this.latencyS) * 1000);
    }
    this.nextTime += buf.duration;
    this.sources.add(src);
    src.onended = () => {
      this.sources.delete(src);
      this.checkDone();
    };
  }

  end(gen: number) {
    if (gen !== this.gen) return;
    this.ended = true;
    this.checkDone();
  }

  /** A new server connection numbers its replies from 1 again, so forget the old connection's numbers
   * (otherwise every new reply looks stale and is silently dropped). */
  newConnection() {
    this.cancel();
    this.gen = -1;
    this.rejectUpTo = -1;
  }

  /** Stop playback of `gen` (or whatever is playing) and reject any of its late chunks. */
  cancel(gen?: number) {
    const target = gen ?? this.gen;
    this.rejectUpTo = Math.max(this.rejectUpTo, target);
    if (target === this.gen) {
      this.stopSources();
      if (this.started) this.finish("cancelled");
      this.started = false;
      this.ended = false;
    }
  }

  get playing() {
    return this.started;
  }

  /** Call every frame: if speech_end got lost, end the reply once its audio has been silent for a few seconds. */
  reap() {
    if (this.started && this.ctx && this.sources.size === 0 && this.ctx.currentTime > this.nextTime + 3) this.finish("ended");
  }

  private stopSources() {
    for (const s of this.sources) {
      s.onended = null;
      try {
        s.stop();
      } catch {
        /* not started yet */
      }
    }
    this.sources.clear();
  }

  private checkDone() {
    if (!this.ended || this.sources.size > 0 || !this.started) return;
    const g = this.gen;
    setTimeout(() => {
      if (this.gen === g && this.started && this.sources.size === 0) this.finish("ended");
    }, this.latencyS * 1000);
  }

  private finish(reason: "ended" | "cancelled") {
    const g = this.gen;
    this.started = false;
    this.events.onEnd(g, reason);
  }

  /** Mouth openness 0..1, delayed to match what the visitor hears right now. */
  level(): number {
    const ctx = this.ctx;
    if (!ctx || !this.analyser) return 0;
    this.analyser.getFloatTimeDomainData(this.sampleBuf);
    let sum = 0;
    for (const v of this.sampleBuf) sum += v * v;
    const rms = Math.sqrt(sum / this.sampleBuf.length);
    const db = 20 * Math.log10(rms + 1e-9);
    const v = Math.min(1, Math.max(0, (db + 48) / 34)) ** 0.85;
    const now = ctx.currentTime;
    this.levels.push({ t: now, v });
    const want = now - this.latencyS;
    while (this.levels.length > 1 && this.levels[1].t <= want) this.levels.shift();
    return this.levels[0].t <= want ? this.levels[0].v : 0;
  }

  // ---- sound effects (bypass the voice analyser so the mouth stays shut) ----

  whoosh(strength = 0.5) {
    const ctx = this.ctx;
    if (!ctx) return;
    const dur = 0.9;
    const src = ctx.createBufferSource();
    src.buffer = noise(ctx, dur);
    const bp = ctx.createBiquadFilter();
    bp.type = "bandpass";
    bp.Q.value = 2.5;
    const t = ctx.currentTime;
    bp.frequency.setValueAtTime(350, t);
    bp.frequency.exponentialRampToValueAtTime(1600, t + dur * 0.45);
    bp.frequency.exponentialRampToValueAtTime(500, t + dur);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.35 * strength, t + dur * 0.4);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    src.connect(bp).connect(g).connect(this.master);
    g.connect(this.wet);
    src.start(t);
  }

  chime() {
    const ctx = this.ctx;
    if (!ctx) return;
    const t = ctx.currentTime;
    [880, 1318.5, 1760, 2637].forEach((f, i) => {
      const o = ctx.createOscillator();
      o.type = "sine";
      o.frequency.value = f;
      const g = ctx.createGain();
      const st = t + i * 0.07;
      g.gain.setValueAtTime(0.0001, st);
      g.gain.exponentialRampToValueAtTime(0.08, st + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, st + 1.2);
      o.connect(g).connect(this.master);
      g.connect(this.wet);
      o.start(st);
      o.stop(st + 1.3);
    });
  }
}

function noise(ctx: BaseAudioContext, seconds: number): AudioBuffer {
  const buf = ctx.createBuffer(1, Math.floor(ctx.sampleRate * seconds), ctx.sampleRate);
  const d = buf.getChannelData(0);
  for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  return buf;
}

/** Seeded, so the hallway sounds the same on every load (random noise made each reload a slightly different room). */
function impulse(ctx: BaseAudioContext, seconds: number, decay: number): AudioBuffer {
  const len = Math.floor(ctx.sampleRate * seconds);
  const buf = ctx.createBuffer(2, len, ctx.sampleRate);
  const rand = mulberry32(1031);
  for (let c = 0; c < 2; c++) {
    const d = buf.getChannelData(c);
    for (let i = 0; i < len; i++) d[i] = (rand() * 2 - 1) * (1 - i / len) ** decay;
  }
  return buf;
}
