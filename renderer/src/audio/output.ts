/**
 * Schedules streamed PCM from the server, adds the ghostly voice effects, and reports
 * when audio is *actually* audible (including Bluetooth / output latency).
 */

export interface OutputEvents {
  onStart(gen: number): void;
  onEnd(gen: number, reason: "ended" | "cancelled"): void;
}

const START_LEAD_S = 0.12; // jitter cushion before the first chunk of a reply
const UNDERRUN_LEAD_S = 0.03;

export class AudioOut {
  ctx: AudioContext | null = null;
  /** Extra latency the browser can't see (Bluetooth speakers), in ms. Operator-tunable. */
  extraLatencyMs = 0;

  private master!: GainNode;
  private voiceIn!: GainNode;
  private wet!: GainNode;
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
      this.buildGraph(this.ctx);
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

  private buildGraph(ctx: AudioContext) {
    this.master = ctx.createGain();
    this.master.gain.value = this.volume;
    this.master.connect(ctx.destination);

    this.voiceIn = ctx.createGain();
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.voiceIn.connect(this.analyser); // dry tap drives the mouth, so reverb tails don't

    const dry = ctx.createGain();
    dry.gain.value = 0.85;
    this.voiceIn.connect(dry).connect(this.master);

    // Haunted-hallway reverb from a synthetic impulse response.
    const conv = ctx.createConvolver();
    conv.buffer = impulse(ctx, 2.6, 2.8);
    this.wet = ctx.createGain();
    this.wet.gain.value = 0.35;
    this.voiceIn.connect(conv).connect(this.wet).connect(this.master);

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
    this.voiceIn.connect(delay).connect(chorus).connect(this.master);
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

  enqueue(gen: number, sampleRate: number, pcm: Int16Array) {
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

function impulse(ctx: BaseAudioContext, seconds: number, decay: number): AudioBuffer {
  const len = Math.floor(ctx.sampleRate * seconds);
  const buf = ctx.createBuffer(2, len, ctx.sampleRate);
  for (let c = 0; c < 2; c++) {
    const d = buf.getChannelData(c);
    for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / len) ** decay;
  }
  return buf;
}
