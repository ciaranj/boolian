/**
 * Commands and status shared between the stage window (projector) and an optional
 * remote operator window on the laptop screen, over BroadcastChannel.
 */
import type { Action, ConvState, Emotion } from "../ghost/ghost";
import type { Settings } from "../settings";

export type Cmd =
  | { c: "say"; text: string; emotion: Emotion; action: Action }
  | { c: "ask"; text: string }
  | { c: "line"; index: number }
  | { c: "greet" }
  | { c: "reset" }
  | { c: "cancel" }
  | { c: "mute" }
  | { c: "ptt"; down: boolean }
  | { c: "auto"; on: boolean }
  | { c: "express"; emotion: Emotion }
  | { c: "action"; action: Action }
  | { c: "settings"; patch: Partial<Settings> }
  | { c: "nudge"; key: "scale" | "offsetX" | "offsetY"; delta: number }
  | { c: "server"; url: string }
  | { c: "test" }
  | { c: "sync" };

export type LogKind = "visitor" | "ghost" | "dim" | "err";

export type Status =
  | { s: "log"; kind: LogKind; text: string }
  | { s: "state"; state: ConvState }
  | { s: "conn"; ok: boolean }
  | { s: "latency"; text: string }
  | { s: "meter"; db: number; thresh: number | null; voice: number | null; voiceThresh: number }
  | { s: "info"; mic?: string; out?: string; backends?: string }
  | { s: "settings"; settings: Settings }
  | { s: "mics"; mics: { id: string; label: string }[] }
  | { s: "mute"; on: boolean }
  | { s: "talk"; on: boolean }
  | { s: "awake"; on: boolean };

type Msg = { kind: "cmd"; cmd: Cmd } | { kind: "status"; status: Status };

export class Bus {
  private ch: BroadcastChannel | null = null;

  constructor() {
    try {
      this.ch = new BroadcastChannel("spectre-operator");
    } catch {
      this.ch = null;
    }
  }

  onCmd(fn: (c: Cmd) => void) {
    this.ch?.addEventListener("message", (e: MessageEvent<Msg>) => e.data.kind === "cmd" && fn(e.data.cmd));
  }

  onStatus(fn: (s: Status) => void) {
    this.ch?.addEventListener("message", (e: MessageEvent<Msg>) => e.data.kind === "status" && fn(e.data.status));
  }

  cmd(cmd: Cmd) {
    this.ch?.postMessage({ kind: "cmd", cmd } satisfies Msg);
  }

  status(status: Status) {
    this.ch?.postMessage({ kind: "status", status } satisfies Msg);
  }
}
