import { ACTIONS, EMOTIONS, type Action, type Emotion } from "../ghost/ghost";
import { PALETTES } from "../ghost/palettes";
import { PUPPET_LINES } from "../puppet";
import type { Settings } from "../settings";
import type { Cmd, Status } from "./bus";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

export const SLIDERS = ["scale", "offsetX", "offsetY", "glow", "volume", "reverb", "fxWaver", "fxFlange", "fxEcho", "fxShadow", "extraLatencyMs", "vadThreshold", "vadPauseMs"] as const;

const pct = (db: number) => `${Math.max(0, Math.min(100, ((db + 80) / 80) * 100))}%`;

/** The operator panel DOM. Renders Status, emits Cmds; knows nothing about audio or the server. */
export class Panel {
  private el = $("panel");
  private autoListen = false;
  private attractOn = true;
  private talking = false;
  private micDevice = "";

  constructor(private run: (c: Cmd) => void) {
    for (const btn of document.querySelectorAll<HTMLButtonElement>("[data-cmd]")) {
      btn.addEventListener("click", () => run({ c: btn.dataset.cmd as "greet" | "reset" | "cancel" | "mute" }));
    }

    const talk = $("talk-btn");
    talk.addEventListener("pointerdown", () => this.ptt(true));
    talk.addEventListener("pointerup", () => this.ptt(false));
    talk.addEventListener("pointerleave", () => this.ptt(false));

    const attract = $<HTMLInputElement>("attract");
    attract.addEventListener("change", () => run({ c: "settings", patch: { attract: attract.checked } }));
    const auto = $<HTMLInputElement>("auto-listen");
    auto.addEventListener("change", () => run({ c: "auto", on: auto.checked }));

    $("ask-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const input = $<HTMLInputElement>("ask");
      const text = input.value.trim();
      if (text) run({ c: "ask", text });
      input.value = "";
    });

    const emo = $<HTMLSelectElement>("say-emotion");
    const act = $<HTMLSelectElement>("say-action");
    for (const e of EMOTIONS) emo.add(new Option(e, e));
    for (const a of ACTIONS) act.add(new Option(a, a));
    $("say-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const input = $<HTMLInputElement>("say");
      const text = input.value.trim();
      if (text) run({ c: "say", text, emotion: emo.value as Emotion, action: act.value as Action });
      input.value = "";
    });

    PUPPET_LINES.forEach((line, index) => {
      const b = document.createElement("button");
      b.innerHTML = `<kbd>${index + 1}</kbd> `;
      b.append(line.text);
      b.addEventListener("click", () => run({ c: "line", index }));
      $("lines").append(b);
    });

    for (const emotion of EMOTIONS) {
      const b = document.createElement("button");
      b.textContent = emotion;
      b.addEventListener("click", () => run({ c: "express", emotion }));
      $("emotions").append(b);
    }
    for (const action of ACTIONS.filter((a) => a !== "none")) {
      const b = document.createElement("button");
      b.textContent = action;
      b.addEventListener("click", () => run({ c: "action", action }));
      $("actions").append(b);
    }

    for (const key of SLIDERS) {
      const el = $<HTMLInputElement>(`s-${key}`);
      el.addEventListener("input", () => run({ c: "settings", patch: { [key]: Number(el.value) } }));
    }

    const palette = $<HTMLSelectElement>("s-palette");
    for (const [name, p] of Object.entries(PALETTES)) palette.add(new Option(p.label, name));
    palette.addEventListener("change", () => run({ c: "settings", patch: { palette: palette.value } }));

    const vadSel = $<HTMLSelectElement>("s-vad");
    vadSel.addEventListener("change", () => run({ c: "settings", patch: { vad: vadSel.value as Settings["vad"] } }));

    const micSel = $<HTMLSelectElement>("s-mic");
    micSel.addEventListener("change", () => run({ c: "settings", patch: { micDevice: micSel.value } }));

    $("alert").addEventListener("click", () => {
      $("alert").hidden = true;
      document.title = document.title.replace(/^⚠ /, "");
    });

    $("server-form").addEventListener("submit", (e) => {
      e.preventDefault();
      run({ c: "server", url: $<HTMLInputElement>("server").value.trim() });
    });
  }

  get visible() {
    return !this.el.hidden;
  }

  toggle(show = this.el.hidden) {
    this.el.hidden = !show;
    document.body.classList.toggle("visitor", !show);
  }

  ptt(down: boolean) {
    if (down === this.talking) return;
    this.talking = down;
    this.run({ c: "ptt", down });
  }

  get auto() {
    return this.autoListen;
  }

  get attract() {
    return this.attractOn;
  }

  apply(st: Status) {
    switch (st.s) {
      case "log": {
        const el = document.createElement("div");
        el.className = st.kind;
        el.textContent = `${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}  ${st.text}`;
        const box = $("log");
        box.prepend(el);
        while (box.childElementCount > 80) box.lastElementChild?.remove();
        break;
      }
      case "state":
        $("state").textContent = st.state;
        break;
      case "conn":
        $("conn").textContent = st.ok ? "online" : "offline";
        $("conn").className = `pill ${st.ok ? "ok" : "bad"}`;
        break;
      case "latency":
        $("latency").textContent = st.text;
        break;
      case "meter":
        if (this.visible) {
          $("mic-meter").style.width = pct(st.db);
          $("mic-thresh").hidden = st.thresh === null;
          if (st.thresh !== null) $("mic-thresh").style.left = pct(st.thresh);
          $("voice-row").hidden = st.voice === null;
          if (st.voice !== null) {
            $("voice-meter").style.width = `${st.voice * 100}%`;
            $("voice-meter").classList.toggle("on", st.voice >= st.voiceThresh);
            $("voice-thresh").style.left = `${st.voiceThresh * 100}%`;
          }
        }
        break;
      case "info":
        if (st.mic !== undefined) $("mic-info").textContent = st.mic;
        if (st.out !== undefined) $("out-latency").textContent = st.out;
        if (st.backends !== undefined) $("backends").textContent = st.backends;
        break;
      case "settings":
        this.applySettings(st.settings);
        break;
      case "mics": {
        const sel = $<HTMLSelectElement>("s-mic");
        sel.replaceChildren(new Option("System default", ""), ...st.mics.map((m) => new Option(m.label, m.id)));
        sel.value = this.micDevice;
        break;
      }
      case "mute":
        $("mute-btn").classList.toggle("on", st.on);
        break;
      case "talk":
        $("talk-btn").classList.toggle("on", st.on);
        if (!st.on) this.talking = false;
        break;
      case "awake":
        $("awake-note").hidden = st.on;
        break;
      case "alert": {
        const el = $("alert");
        el.textContent = `⚠ ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}  ${st.text}  (click to dismiss)`;
        el.hidden = false;
        if (!document.title.startsWith("⚠")) document.title = `⚠ ${document.title}`;
        break;
      }
    }
  }

  private applySettings(s: Settings) {
    for (const key of SLIDERS) {
      const el = $<HTMLInputElement>(`s-${key}`);
      if (document.activeElement !== el) el.value = String(s[key]);
    }
    $<HTMLSelectElement>("s-palette").value = s.palette;
    $<HTMLSelectElement>("s-vad").value = s.vad;
    this.micDevice = s.micDevice;
    $<HTMLSelectElement>("s-mic").value = s.micDevice;
    this.autoListen = s.autoListen;
    $<HTMLInputElement>("auto-listen").checked = s.autoListen;
    this.attractOn = s.attract;
    $<HTMLInputElement>("attract").checked = s.attract;
    const server = $<HTMLInputElement>("server");
    if (document.activeElement !== server) server.value = s.serverUrl;
  }
}

/** Hotkeys, identical in the stage and remote windows. */
export function bindKeys(panel: Panel, run: (c: Cmd) => void, local: { fullscreen(): void; wake(): boolean }) {
  window.addEventListener("keydown", (e) => {
    const t = e.target;
    const typing = (t instanceof HTMLInputElement && t.type !== "checkbox" && t.type !== "range") || t instanceof HTMLSelectElement;
    if (typing) {
      if (e.key === "Escape") (t as HTMLElement).blur();
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if ((e.key === "Enter" || e.key === " ") && local.wake()) {
      e.preventDefault();
      return;
    }
    const k = e.key.toLowerCase();
    if (e.key === " ") {
      e.preventDefault();
      if (!e.repeat) panel.ptt(true);
      return;
    }
    const repeatable = k.startsWith("arrow") || k === "+" || k === "=" || k === "-";
    if (e.repeat && !repeatable) return;
    if (/^[1-9]$/.test(k)) return run({ c: "line", index: Number(k) - 1 });
    const simple: Record<string, Cmd> = {
      g: { c: "greet" },
      r: { c: "reset" },
      x: { c: "cancel" },
      escape: { c: "cancel" },
      m: { c: "mute" },
      k: { c: "test" },
      arrowleft: { c: "nudge", key: "offsetX", delta: -0.01 },
      arrowright: { c: "nudge", key: "offsetX", delta: 0.01 },
      arrowup: { c: "nudge", key: "offsetY", delta: -0.01 },
      arrowdown: { c: "nudge", key: "offsetY", delta: 0.01 },
      "+": { c: "nudge", key: "scale", delta: 0.03 },
      "=": { c: "nudge", key: "scale", delta: 0.03 },
      "-": { c: "nudge", key: "scale", delta: -0.03 },
    };
    if (simple[k]) {
      if (k.startsWith("arrow")) e.preventDefault();
      return run(simple[k]);
    }
    if (k === "v") run({ c: "auto", on: !panel.auto });
    else if (k === "a") run({ c: "settings", patch: { attract: !panel.attract } });
    else if (k === "o") panel.toggle();
    else if (k === "f") local.fullscreen();
  });

  window.addEventListener("keyup", (e) => {
    if (e.key === " ") panel.ptt(false);
  });
  window.addEventListener("blur", () => panel.ptt(false));
}
