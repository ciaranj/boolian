/** WebSocket link to the Spectre server. Binary layouts are documented in docs/protocol-v1.md. */

const VERSION = 1;
const MIC_MAGIC = [0x47, 0x48, 0x53, 0x54]; // "GHST"
const OUT_MAGIC = [0x47, 0x48, 0x41, 0x55]; // "GHAU"
const MIC_HEADER = 16;
const OUT_HEADER = 20;

export type RudeNoise = "raspberry" | "fart";

export interface AudioChunk {
  gen: number;
  seq: number;
  sampleRate: number;
  last: boolean;
  noise: RudeNoise | null; // this chunk is a rude noise, not speech
  pcm: Int16Array;
}

// Shape of every JSON event the server sends; fields depend on `type`.
export interface ServerEvent {
  type: string;
  generation_id?: number;
  [key: string]: unknown;
}

export function packMicFrame(seq: number, captureMs: number, pcm: Int16Array): ArrayBuffer {
  const buf = new ArrayBuffer(MIC_HEADER + pcm.byteLength);
  const dv = new DataView(buf);
  MIC_MAGIC.forEach((b, i) => dv.setUint8(i, b));
  dv.setUint8(4, VERSION);
  dv.setUint8(5, 0);
  dv.setUint16(6, pcm.byteLength);
  dv.setUint32(8, seq >>> 0);
  dv.setUint32(12, captureMs >>> 0);
  for (let i = 0; i < pcm.length; i++) dv.setInt16(MIC_HEADER + i * 2, pcm[i], true);
  return buf;
}

/** A frame from the ESP32's microphone, relayed by the server: its 16 kHz PCM, or null if it isn't one. */
export function parseMicFrame(buf: ArrayBuffer): Int16Array | null {
  if (buf.byteLength < MIC_HEADER) return null;
  const dv = new DataView(buf);
  for (let i = 0; i < 4; i++) if (dv.getUint8(i) !== MIC_MAGIC[i]) return null;
  if (dv.getUint8(4) !== VERSION) return null;
  const n = (buf.byteLength - MIC_HEADER) >> 1;
  const pcm = new Int16Array(n);
  for (let i = 0; i < n; i++) pcm[i] = dv.getInt16(MIC_HEADER + i * 2, true);
  return pcm;
}

export function parseAudioChunk(buf: ArrayBuffer): AudioChunk | null {
  if (buf.byteLength < OUT_HEADER) return null;
  const dv = new DataView(buf);
  for (let i = 0; i < 4; i++) if (dv.getUint8(i) !== OUT_MAGIC[i]) return null;
  if (dv.getUint8(4) !== VERSION) return null;
  const flags = dv.getUint8(5);
  const n = (buf.byteLength - OUT_HEADER) >> 1;
  const pcm = new Int16Array(n);
  for (let i = 0; i < n; i++) pcm[i] = dv.getInt16(OUT_HEADER + i * 2, true);
  return {
    gen: dv.getUint32(8),
    seq: dv.getUint32(12),
    sampleRate: dv.getUint32(16),
    last: (flags & 1) === 1,
    noise: flags & 2 ? "raspberry" : flags & 4 ? "fart" : null,
    pcm,
  };
}

export class Link {
  onEvent: (e: ServerEvent) => void = () => {};
  onAudio: (c: AudioChunk) => void = () => {};
  onMic: (pcm: Int16Array) => void = () => {};
  onStatus: (connected: boolean) => void = () => {};
  private ws: WebSocket | null = null;
  private backoff = 1000;
  private timer: number | undefined;
  private micSeq = 0;
  private closedByUs = false;

  constructor(public url: string) {}

  get connected() {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  connect() {
    this.closedByUs = false;
    clearTimeout(this.timer);
    let ws: WebSocket;
    try {
      ws = new WebSocket(this.url);
    } catch {
      this.scheduleReconnect();
      return;
    }
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    ws.onopen = () => {
      this.backoff = 1000;
      this.micSeq = 0;
      this.send({ type: "hello", role: "presenter", client: "renderer" });
      this.onStatus(true);
    };
    ws.onmessage = (m) => {
      if (typeof m.data === "string") {
        try {
          this.onEvent(JSON.parse(m.data));
        } catch {
          /* ignore malformed */
        }
      } else {
        const chunk = parseAudioChunk(m.data as ArrayBuffer);
        if (chunk) this.onAudio(chunk);
        else {
          const pcm = parseMicFrame(m.data as ArrayBuffer);
          if (pcm) this.onMic(pcm);
        }
      }
    };
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.ws = null;
      this.onStatus(false);
      if (!this.closedByUs) this.scheduleReconnect();
    };
    ws.onerror = () => ws.close();
  }

  reconnect(url: string) {
    this.url = url;
    this.closedByUs = true;
    this.ws?.close();
    this.ws = null;
    this.onStatus(false);
    this.connect();
  }

  private scheduleReconnect() {
    this.timer = window.setTimeout(() => this.connect(), this.backoff);
    this.backoff = Math.min(this.backoff * 1.6, 5000);
  }

  send(msg: Record<string, unknown>) {
    if (this.connected) this.ws!.send(JSON.stringify({ v: VERSION, ...msg }));
  }

  sendMic(pcm: Int16Array) {
    if (!this.connected) return;
    this.ws!.send(packMicFrame(this.micSeq++, performance.now(), pcm));
  }
}
