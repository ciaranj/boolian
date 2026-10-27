/**
 * Silero VAD (v5) via ricky0123/vad-web, fed from our own 16 kHz mic frames rather than its MicVAD, so the
 * mic picker, pre-roll and streaming stay as they are. Everything is bundled locally: no CDN on show night.
 */
import { FrameProcessor, Message } from "@ricky0123/vad-web";
import modelUrl from "@ricky0123/vad-web/dist/silero_vad_v5.onnx?url";
import { Silero } from "@ricky0123/vad-web/dist/models";
import * as ort from "onnxruntime-web/wasm";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import type { VadEvent } from "./mic";

const FRAME = 512; // Silero v5 takes 32 ms at 16 kHz
const MS_PER_FRAME = FRAME / 16;

export interface SileroOptions {
  threshold: number; // speech probability that counts as speech (lower = more sensitive)
  pauseMs: number; // quiet after speech before the visitor is taken to have finished
  minSpeechMs: number; // speech needed before we start listening, so a cough or bang doesn't count
  maxUtterMs: number;
}

type FrameEvent = Parameters<Parameters<FrameProcessor["process"]>[1]>[0];

export class SileroVad {
  onEvent: (ev: VadEvent) => void = () => {};
  /** Latest speech probability, for the operator meter. */
  prob = 0;
  private buf = new Float32Array(FRAME);
  private n = 0;
  private queue: Promise<void> = Promise.resolve();
  private gen = 0;
  private utterFrames = -1; // frames since we said "start", or -1 when not in an utterance

  private constructor(private fp: FrameProcessor, private opts: SileroOptions) {
    fp.resume();
  }

  static async load(opts: SileroOptions): Promise<SileroVad> {
    ort.env.wasm.wasmPaths = { wasm: wasmUrl };
    ort.env.wasm.numThreads = 1; // threads need cross-origin isolation, and one is plenty for a 2 MB model
    const model = await Silero.new(ort, () => fetch(modelUrl).then((r) => r.arrayBuffer()));
    const fp = new FrameProcessor(model.process, model.reset_state, SileroVad.fpOptions(opts), MS_PER_FRAME);
    return new SileroVad(fp, opts);
  }

  private static fpOptions(o: SileroOptions) {
    return {
      positiveSpeechThreshold: o.threshold,
      negativeSpeechThreshold: Math.max(0.05, o.threshold - 0.15), // Silero's recommended gap
      redemptionMs: o.pauseMs,
      minSpeechMs: o.minSpeechMs,
      preSpeechPadMs: 0, // we keep our own pre-roll
      submitUserSpeechOnPause: false,
    };
  }

  setOptions(patch: Partial<SileroOptions>) {
    this.opts = { ...this.opts, ...patch };
    this.fp.setOptions(SileroVad.fpOptions(this.opts));
  }

  get threshold() {
    return this.opts.threshold;
  }

  feed(pcm: Int16Array) {
    for (let i = 0; i < pcm.length; i++) {
      this.buf[this.n++] = pcm[i] / 32768;
      if (this.n === FRAME) {
        const frame = this.buf;
        this.buf = new Float32Array(FRAME);
        this.n = 0;
        const gen = this.gen;
        // Frames are scored strictly in order; a reset() drops anything still queued.
        this.queue = this.queue
          .then(async () => {
            if (gen !== this.gen) return;
            await this.fp.process(frame, this.handle);
            if (this.utterFrames >= 0 && ++this.utterFrames * MS_PER_FRAME >= this.opts.maxUtterMs) {
              this.fp.endSegment(this.handle);
            }
          })
          .catch((err) => console.error("Silero VAD failed on a frame", err));
      }
    }
  }

  reset() {
    this.gen++;
    this.n = 0;
    this.prob = 0;
    this.utterFrames = -1;
    this.fp.reset();
  }

  private handle = (ev: FrameEvent) => {
    switch (ev.msg) {
      case Message.FrameProcessed:
        this.prob = ev.probs.isSpeech;
        break;
      case Message.SpeechRealStart:
        this.utterFrames = 0;
        this.onEvent("start");
        break;
      case Message.SpeechEnd:
      case Message.VADMisfire:
        // A misfire never got as far as "start", so there's nothing to take back.
        if (this.utterFrames >= 0) this.onEvent("end");
        this.utterFrames = -1;
        break;
    }
  };
}
