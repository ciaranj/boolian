/**
 * Microphone capture: resamples to 16 kHz mono s16 in an AudioWorklet and emits 20 ms frames
 * (320 samples), the same format the ESP32 will send later.
 */

export const MIC_RATE = 16000;
export const FRAME_SAMPLES = 320;

const WORKLET = `
class GhostMic extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / ${MIC_RATE};
    this.a = 1 - Math.exp(-2 * Math.PI * 6500 / sampleRate); // anti-alias low-pass
    this.lp1 = 0; this.lp2 = 0; this.prev = 0; this.k = 0; this.next = 0;
    this.out = new Int16Array(${FRAME_SAMPLES}); this.n = 0;
  }
  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input) return true;
    for (let i = 0; i < input.length; i++) {
      this.lp1 += this.a * (input[i] - this.lp1);
      this.lp2 += this.a * (this.lp1 - this.lp2);
      const cur = this.lp2;
      while (this.next <= this.k) {
        const v = this.prev + (cur - this.prev) * (this.next - (this.k - 1));
        this.out[this.n++] = Math.max(-32768, Math.min(32767, Math.round(v * 32767)));
        if (this.n === ${FRAME_SAMPLES}) {
          this.port.postMessage(this.out.buffer, [this.out.buffer]);
          this.out = new Int16Array(${FRAME_SAMPLES});
          this.n = 0;
        }
        this.next += this.ratio;
      }
      this.prev = cur;
      this.k++;
    }
    return true;
  }
}
registerProcessor("ghost-mic", GhostMic);
`;

export class MicInput {
  /** Most recent frame loudness in dBFS. */
  db = -100;
  onFrame: (pcm: Int16Array, db: number) => void = () => {};
  private stream: MediaStream | null = null;
  private registered = new WeakSet<AudioContext>();
  private node: AudioWorkletNode | null = null;

  get running() {
    return this.node !== null;
  }

  /** Which input the browser actually picked, e.g. "MacBook Pro Microphone (Built-in)". */
  get deviceLabel() {
    return this.stream?.getAudioTracks()[0]?.label || "unknown device";
  }

  async start(ctx: AudioContext, deviceId?: string) {
    this.stop();
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    if (!this.registered.has(ctx)) {
      const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
      await ctx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
      this.registered.add(ctx);
    }
    const src = ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(ctx, "ghost-mic");
    this.node.port.onmessage = (e: MessageEvent<ArrayBuffer>) => {
      const pcm = new Int16Array(e.data);
      let sum = 0;
      for (const s of pcm) sum += s * s;
      this.db = 20 * Math.log10(Math.sqrt(sum / pcm.length) / 32768 + 1e-9);
      this.onFrame(pcm, this.db);
    };
    const sink = ctx.createGain();
    sink.gain.value = 0;
    src.connect(this.node).connect(sink).connect(ctx.destination);
  }

  stop() {
    this.node?.disconnect();
    this.node = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
  }
}

export type VadEvent = "start" | "end" | "abort";

/**
 * Energy VAD with an adaptive noise floor: the fallback when Silero (silero.ts) can't load, or is switched
 * off in the operator panel. It can't tell speech from any other loud noise.
 */
export class EnergyVad {
  onEvent: (ev: VadEvent) => void = () => {};
  noiseDb = -60;
  speaking = false;
  private speechFrames = 0;
  private silenceFrames = 0;
  private utterFrames = 0;

  constructor(
    public opts = { startFrames: 4, endSilenceMs: 750, minUtterMs: 450, maxUtterMs: 12000, marginDb: 14, floorDb: -50 },
  ) {}

  get threshold() {
    return Math.max(this.noiseDb + this.opts.marginDb, this.opts.floorDb);
  }

  reset() {
    this.speaking = false;
    this.speechFrames = this.silenceFrames = this.utterFrames = 0;
  }

  feed(db: number) {
    const ev = this.process(db);
    if (ev) this.onEvent(ev);
  }

  process(db: number): VadEvent | null {
    const isSpeech = db > this.threshold;
    if (!this.speaking) {
      this.noiseDb += (db - this.noiseDb) * (db < this.noiseDb ? 0.1 : 0.01);
      this.speechFrames = isSpeech ? this.speechFrames + 1 : 0;
      if (this.speechFrames >= this.opts.startFrames) {
        this.speaking = true;
        this.utterFrames = this.speechFrames;
        this.silenceFrames = 0;
        return "start";
      }
      return null;
    }
    this.utterFrames++;
    this.silenceFrames = isSpeech ? 0 : this.silenceFrames + 1;
    if (this.silenceFrames * 20 >= this.opts.endSilenceMs || this.utterFrames * 20 >= this.opts.maxUtterMs) {
      const voicedMs = (this.utterFrames - this.silenceFrames) * 20;
      this.reset();
      return voicedMs >= this.opts.minUtterMs ? "end" : "abort";
    }
    return null;
  }
}
