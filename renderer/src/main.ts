import { EnergyVad, MicInput, type VadEvent } from "./audio/mic";
import { AudioOut } from "./audio/output";
import { SileroVad } from "./audio/silero";
import { ACTIONS, EMOTIONS, Ghost, type Action, type ConvState, type Emotion } from "./ghost/ghost";
import { Ghost3D } from "./ghost/ghost3d";
import { paletteOf } from "./ghost/palettes";
import { drawTestPattern } from "./ghost/testscene";
import { Bus, type Cmd, type LogKind, type Status } from "./operator/bus";
import { Panel, bindKeys } from "./operator/panel";
import { Link, type ServerEvent } from "./protocol";
import { PUPPET_LINES } from "./puppet";
import { loadSettings, saveSettings, serverUrlWithToken } from "./settings";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const q = new URLSearchParams(location.search);
const bus = new Bus();

const fullscreen = () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen().catch(() => {});
};

if (q.has("remote")) startRemote();
else startStage();

/**
 * Remote operator window: just the panel, driving the stage window on the projector.
 * Audio unlock and fullscreen still need one click/keypress in the stage window itself.
 */
function startRemote() {
  document.title = "Boolian · operator";
  document.body.classList.add("remote");
  $("stage").hidden = true;
  $("wake").hidden = true;
  $("conn").textContent = "no stage window";
  const panel = new Panel((c) => bus.cmd(c));
  panel.toggle(true);
  bus.onStatus((s) => panel.apply(s));
  bindKeys(panel, (c) => bus.cmd(c), { fullscreen, wake: () => false });
  bus.cmd({ c: "sync" });
}

/** The projector window: ghost, audio, microphone and the server connection. */
function startStage() {
  const settings = loadSettings();
  const ghost = new Ghost(Number(q.get("seed")) || Date.now());
  const energyVad = new EnergyVad();
  let silero: SileroVad | null = null;
  let vad: EnergyVad | SileroVad = energyVad;
  let vadOpen = false; // was the VAD being fed last frame (it's reset when feeding resumes)
  const mic = new MicInput();

  let conv: ConvState = "sleeping";
  let awake = false;
  let testPattern = false;
  let capturing = false;
  let captureBy: "ptt" | "vad" | null = null;
  let tailUntil = 0;
  let errorTimer: number | undefined;
  let thinkingSince = 0;
  let listenEndAt = 0;
  let speakingLocally = false;
  let localFlap = 0;
  let localSilentFor = 0;
  let autoBlockedSince = 0;
  let micInfo = "mic off";
  let backends = "";
  const latencies: number[] = [];
  const preRoll: Int16Array[] = [];
  const PRE_ROLL_FRAMES = 30; // 600 ms: Silero only says "start" once it's heard 250 ms of speech

  const panel = new Panel(exec);
  bus.onCmd(exec);

  function report(s: Status) {
    panel.apply(s);
    bus.status(s);
  }
  const log = (kind: LogKind, text: string) => report({ s: "log", kind, text });

  // ---------------------------------------------------------------- audio out

  const out = new AudioOut({
    onStart(gen) {
      clearTimeout(errorTimer);
      setConv("speaking");
      link.send({ type: "playback", generation_id: gen, event: "started" });
      if (listenEndAt) {
        const ms = performance.now() - listenEndAt;
        listenEndAt = 0;
        latencies.push(ms);
        const sorted = [...latencies].sort((a, b) => a - b);
        const p = (x: number) => sorted[Math.min(sorted.length - 1, Math.round(x * (sorted.length - 1)))].toFixed(0);
        report({ s: "latency", text: `end of speech → audible: ${ms.toFixed(0)} ms (p50 ${p(0.5)}, p95 ${p(0.95)}, n=${latencies.length})` });
      }
    },
    onEnd(gen, reason) {
      link.send({ type: "playback", generation_id: gen, event: reason });
      tailUntil = performance.now() + 600;
      ghost.relaxSoon(1.5);
      if (conv === "speaking") setConv("idle");
    },
  });

  // ---------------------------------------------------------------- server link

  const link = new Link(serverUrlWithToken(settings.serverUrl));

  link.onStatus = (ok) => {
    report({ s: "conn", ok });
    $("conn-dot").classList.toggle("ok", ok);
    if (!ok) {
      out.cancel();
      stopCapture(false);
      if (!speakingLocally) setConv("sleeping");
      log("dim", "server disconnected");
    } else {
      out.newConnection();
      log("dim", "server connected");
      if (awake) setConv("idle");
    }
  };

  link.onEvent = (e: ServerEvent) => {
    const gen = e.generation_id ?? -1;
    switch (e.type) {
      case "welcome": {
        const b = e.backends as Record<string, string>;
        backends = `server: stt=${b.stt} llm=${b.llm} tts=${b.tts}`;
        report({ s: "info", backends });
        break;
      }
      case "state":
        if (e.state === "thinking") {
          if (conv !== "thinking") out.whoosh(0.35);
          setConv("thinking");
        } else if (e.state === "idle" && conv !== "speaking") {
          listenEndAt = 0; // utterance was ignored; don't time the next unrelated reply against it
          setConv("idle");
        } else if (e.state === "error") {
          listenEndAt = 0;
          setConv("error");
          log("err", "the ghost couldn't answer (backend error)");
          clearTimeout(errorTimer);
          errorTimer = window.setTimeout(() => conv === "error" && setConv("idle"), 3000);
        }
        break;
      case "transcript":
        if (e.rejected) log("dim", `heard “${e.text || "…"}”, ignored (${e.rejected})`);
        else log("visitor", `${e.typed ? "⌨ " : ""}${e.text}`);
        break;
      case "reply":
        if (e.text) log("ghost", String(e.text));
        break;
      case "expression":
        // Held until playback ends (relaxSoon), so the face matches the whole line.
        ghost.express(asEmotion(e.emotion), asAction(e.action), Number(e.intensity ?? 0.8), 20000);
        break;
      case "speech_start":
        out.begin(gen);
        break;
      case "speech_end":
        out.end(gen);
        break;
      case "cancel":
        out.cancel(gen);
        if (conv === "thinking") setConv("idle");
        break;
      case "session_reset":
        log("dim", `session reset (${e.reason})`);
        break;
    }
  };

  link.onAudio = (c) => out.enqueue(c.gen, c.sampleRate, c.pcm);

  // ---------------------------------------------------------------- conversation state

  function setConv(s: ConvState) {
    if (s === "thinking" && conv !== "thinking") thinkingSince = performance.now();
    conv = s;
    ghost.setState(s);
    report({ s: "state", state: s });
  }

  function micAllowed() {
    return autoBlocker() === null;
  }

  /** Why auto-listen is ignoring the mic right now, or null if it's listening for speech. */
  function autoBlocker(): string | null {
    if (conv === "speaking" || conv === "thinking") return `ghost is ${conv}`;
    if (out.playing) return "reply audio still playing";
    if (speakingLocally) return "browser voice still speaking";
    if (performance.now() <= tailUntil) return "waiting for the echo to die away";
    return null;
  }

  function startCapture(by: "ptt" | "vad") {
    if (capturing) return;
    if (!link.connected || !mic.running) {
      if (by === "ptt") log("err", !link.connected ? "can't listen: not connected to the server" : `can't listen: ${micInfo.startsWith("mic unavailable") ? micInfo : "microphone not started, click wake"}`);
      return;
    }
    if (by === "ptt" && (conv === "speaking" || conv === "thinking" || out.playing)) {
      // Operator override: hush the ghost and listen.
      out.cancel();
      link.send({ type: "cancel" });
    }
    capturing = true;
    captureBy = by;
    link.send({ type: "listen_start" });
    for (const f of preRoll) link.sendMic(f);
    preRoll.length = 0;
    setConv("listening");
    report({ s: "talk", on: true });
  }

  function stopCapture(commit: boolean) {
    if (!capturing) return;
    capturing = false;
    captureBy = null;
    vad.reset();
    vadOpen = false;
    report({ s: "talk", on: false });
    if (commit) {
      link.send({ type: "listen_end" });
      listenEndAt = performance.now();
      setConv("thinking");
    } else {
      link.send({ type: "listen_abort" });
      if (conv === "listening") setConv("idle");
    }
  }

  mic.onFrame = (pcm, db) => {
    ghost.micLevel = capturing ? Math.min(1, Math.max(0, (db + 55) / 40)) : 0;
    if (capturing) {
      link.sendMic(pcm);
    } else {
      preRoll.push(pcm);
      if (preRoll.length > PRE_ROLL_FRAMES) preRoll.shift();
    }
    const feed = settings.autoListen && (capturing ? captureBy === "vad" : micAllowed());
    if (feed && !vadOpen) vad.reset(); // don't let stale audio from before Boolian spoke count
    vadOpen = feed;
    if (!feed) return;
    if (vad instanceof EnergyVad) vad.feed(db);
    else vad.feed(pcm);
  };

  // Silero answers a few ms after the frame, so re-check that listening is still wanted.
  function onVad(ev: VadEvent) {
    if (ev === "start") {
      if (!capturing && settings.autoListen && micAllowed()) startCapture("vad");
    } else if (captureBy === "vad") {
      stopCapture(ev === "end");
    }
  }
  energyVad.onEvent = onVad;

  SileroVad.load({ threshold: settings.vadThreshold, pauseMs: settings.vadPauseMs, minSpeechMs: 250, maxUtterMs: 12000 })
    .then((v) => {
      silero = v;
      silero.onEvent = onVad;
      log("dim", "voice detector: Silero ready");
      applySettings();
    })
    .catch((err) => {
      console.error(err);
      log("err", `Silero voice detector failed to load (${(err as Error).message}): using loudness only`);
    });

  // ---------------------------------------------------------------- commands

  function say(text: string, emotion: Emotion, action: Action) {
    listenEndAt = 0; // puppet lines aren't conversational latency
    if (link.connected) {
      out.cancel();
      link.send({ type: "say", text, emotion, action });
      return;
    }
    // Offline fallback: the browser's own voice with a pretend mouth, so puppet mode always works.
    if (!("speechSynthesis" in window)) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.pitch = 1.25;
    u.rate = 0.95;
    u.onstart = () => {
      speakingLocally = true;
      setConv("speaking");
    };
    u.onend = u.onerror = () => {
      speakingLocally = false;
      ghost.relaxSoon(1.5);
      setConv(link.connected ? "idle" : "sleeping");
    };
    ghost.express(emotion, action, 0.9, 20000);
    speechSynthesis.speak(u);
    log("ghost", `(offline voice) ${text}`);
  }

  function saveAndApply() {
    applySettings();
    saveSettings(settings);
    report({ s: "settings", settings });
  }

  function exec(cmd: Cmd) {
    switch (cmd.c) {
      case "say":
        say(cmd.text, cmd.emotion, cmd.action);
        break;
      case "line": {
        const line = PUPPET_LINES[cmd.index];
        if (line) say(line.text, line.emotion, line.action);
        break;
      }
      case "ask":
        if (!link.connected) return log("err", "not connected to the server");
        out.cancel();
        link.send({ type: "text", text: cmd.text });
        listenEndAt = performance.now();
        break;
      case "greet":
        if (link.connected) {
          out.cancel();
          out.chime();
          listenEndAt = 0;
          link.send({ type: "greet" });
        } else say(PUPPET_LINES[0].text, "happy", "bounce");
        break;
      case "reset":
        out.cancel();
        link.send({ type: "reset" });
        ghost.express("surprised", "none", 0.5, 800);
        break;
      case "cancel":
        out.cancel();
        if ("speechSynthesis" in window) speechSynthesis.cancel();
        stopCapture(false);
        link.send({ type: "cancel" });
        if (conv !== "sleeping") setConv("idle");
        break;
      case "mute":
        out.setMuted(!out.isMuted);
        report({ s: "mute", on: out.isMuted });
        log("dim", out.isMuted ? "muted" : "unmuted");
        break;
      case "ptt":
        if (cmd.down) startCapture("ptt");
        else if (captureBy === "ptt") stopCapture(true);
        break;
      case "auto":
        settings.autoListen = cmd.on;
        vad.reset();
        vadOpen = false;
        if (!cmd.on && captureBy === "vad") stopCapture(false);
        saveAndApply();
        log("dim", `auto-listen ${cmd.on ? "on" : "off"}`);
        break;
      case "express":
        ghost.express(cmd.emotion, "none", 0.9, 4000);
        break;
      case "action":
        ghost.doAction(cmd.action, 0.9, true);
        break;
      case "settings": {
        const micChanged = cmd.patch.micDevice !== undefined && cmd.patch.micDevice !== settings.micDevice;
        Object.assign(settings, cmd.patch);
        saveAndApply();
        if (micChanged) startMic();
        break;
      }
      case "nudge":
        settings[cmd.key] = Math.round((settings[cmd.key] + cmd.delta) * 1000) / 1000;
        saveAndApply();
        break;
      case "server":
        settings.serverUrl = cmd.url;
        saveAndApply();
        link.reconnect(serverUrlWithToken(settings.serverUrl));
        break;
      case "test":
        testPattern = !testPattern;
        break;
      case "sync":
        bus.status({ s: "conn", ok: link.connected });
        bus.status({ s: "state", state: conv });
        bus.status({ s: "settings", settings });
        bus.status({ s: "mute", on: out.isMuted });
        bus.status({ s: "awake", on: awake });
        bus.status({ s: "info", mic: micInfo, backends });
        bus.status({ s: "mics", mics });
        break;
    }
  }

  function asEmotion(v: unknown): Emotion {
    return (EMOTIONS as readonly string[]).includes(v as string) ? (v as Emotion) : "neutral";
  }
  function asAction(v: unknown): Action {
    return (ACTIONS as readonly string[]).includes(v as string) ? (v as Action) : "none";
  }

  function applySettings() {
    ghost.settings = { scale: settings.scale, offsetX: settings.offsetX, offsetY: settings.offsetY, glow: settings.glow };
    ghost.palette = paletteOf(q.get("palette") ?? settings.palette);
    out.setVolume(settings.volume);
    out.setReverb(settings.reverb);
    out.extraLatencyMs = settings.extraLatencyMs;
    energyVad.opts.endSilenceMs = settings.vadPauseMs;
    silero?.setOptions({ threshold: settings.vadThreshold, pauseMs: settings.vadPauseMs });
    const want = settings.vad === "silero" && silero ? silero : energyVad;
    if (want !== vad) {
      if (captureBy === "vad") stopCapture(false);
      vad = want;
      vad.reset();
      vadOpen = false;
    }
  }

  // ---------------------------------------------------------------- waking up (audio unlock)

  let mics: { id: string; label: string }[] = [];

  async function startMic() {
    const ctx = out.ctx;
    if (!ctx) return; // not awake yet: wake() starts it with the saved choice
    if (capturing) stopCapture(false);
    try {
      try {
        await mic.start(ctx, settings.micDevice || undefined);
      } catch (err) {
        if (!settings.micDevice) throw err;
        log("err", "the chosen microphone isn't available: using the system default");
        await mic.start(ctx);
      }
      micInfo = `mic: ${mic.deviceLabel} (${ctx.sampleRate} Hz → 16 kHz)`;
      log("dim", micInfo);
    } catch (err) {
      micInfo = `mic unavailable: ${(err as Error).message}`;
      log("err", "microphone unavailable: use typed input or puppet lines");
    }
    report({ s: "info", mic: micInfo });
    await listMics();
  }

  // Device names are only visible once the page has mic permission, so list them after starting.
  async function listMics() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      mics = devices
        .filter((d) => d.kind === "audioinput" && d.deviceId !== "default" && d.deviceId !== "communications")
        .map((d) => ({ id: d.deviceId, label: d.label || "unnamed microphone" }));
    } catch {
      mics = [];
    }
    report({ s: "mics", mics });
  }
  navigator.mediaDevices?.addEventListener("devicechange", () => awake && listMics());

  async function wake() {
    if (awake) return;
    awake = true;
    $("wake").hidden = true;
    report({ s: "awake", on: true });
    await out.unlock();
    applySettings();
    out.chime();
    await startMic();
    setConv(link.connected ? "idle" : "sleeping");
  }

  $("wake-btn").addEventListener("click", wake);
  $("remote-btn").addEventListener("click", () => window.open("?remote", "spectre-remote", "width=420,height=900"));
  bindKeys(panel, exec, {
    fullscreen,
    wake: () => {
      if (awake) return false;
      wake();
      return true;
    },
  });

  // ---------------------------------------------------------------- render loop

  const canvas = $<HTMLCanvasElement>("stage");
  const ctx = canvas.getContext("2d")!;
  // ?renderer=3d draws him with three.js behind the 2D canvas, which then only shows the test pattern.
  const ghost3d = q.get("renderer") === "3d" ? new Ghost3D(ghost, q.get("q3d") === "low" ? "low" : "high") : null;
  if (ghost3d) canvas.before(ghost3d.canvas);
  let cw = 0;
  let ch = 0;

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cw = window.innerWidth;
    ch = window.innerHeight;
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ghost3d?.resize(cw, ch);
  }
  window.addEventListener("resize", resize);
  resize();

  // Art/debug mode without a server: ?state=speaking&emotion=happy&mouth=0.6
  const debugMouth = q.has("mouth") ? Number(q.get("mouth")) : null;
  if (q.get("state")) {
    setConv(q.get("state") as ConvState);
    if (q.get("emotion")) ghost.express(asEmotion(q.get("emotion")), asAction(q.get("action")), 1, 1e9);
    for (let i = 0; i < 60; i++) ghost.update(0.05); // settle the pose for screenshots
  }

  let last = performance.now();
  let frames = 0;
  let fpsT = last;
  let meterT = last;

  function frame(now: number) {
    const dt = (now - last) / 1000;
    last = now;

    // Chrome sometimes never fires onend for speechSynthesis; don't let that block the mic forever.
    localSilentFor = speakingLocally && !speechSynthesis.speaking ? localSilentFor + dt : 0;
    if (localSilentFor > 1) {
      speakingLocally = false;
      localSilentFor = 0;
      setConv(link.connected ? "idle" : "sleeping");
    }
    if (speakingLocally) {
      localFlap += dt * 11;
      ghost.speechLevel = 0.35 + 0.35 * Math.sin(localFlap) * Math.sin(localFlap * 0.37 + 1);
    } else {
      ghost.speechLevel = debugMouth ?? out.level();
    }
    if (conv === "thinking" && now - thinkingSince > 25000) setConv("idle"); // never get stuck pondering
    out.reap();

    ghost.update(dt);
    if (ghost3d && !testPattern) {
      ctx.clearRect(0, 0, cw, ch);
      ghost3d.render();
    } else {
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, cw, ch);
      if (testPattern) drawTestPattern(ctx, cw, ch, ghost);
      else ghost.draw(ctx, cw, ch);
    }
    ghost3d?.setVisible(!testPattern);

    frames++;
    if (now - meterT > 100) {
      report(
        vad instanceof EnergyVad
          ? { s: "meter", db: mic.db, thresh: energyVad.threshold, voice: null, voiceThresh: 0 }
          : { s: "meter", db: mic.db, thresh: null, voice: vad.prob, voiceThresh: vad.threshold },
      );
      meterT = now;
      const blocker = settings.autoListen && awake && mic.running && !capturing ? autoBlocker() : null;
      if (!blocker || conv === "speaking" || conv === "thinking") autoBlockedSince = 0;
      else if (!autoBlockedSince) autoBlockedSince = now;
      else if (autoBlockedSince > 0 && now - autoBlockedSince > 8000) {
        log("err", `auto-listen paused: ${blocker}`);
        autoBlockedSince = -1; // once per stuck spell
      }
    }
    if (now - fpsT > 1000) {
      report({ s: "info", out: `${frames} fps · output latency ${(out.latencyS * 1000).toFixed(0)} ms` });
      frames = 0;
      fpsT = now;
    }
    requestAnimationFrame(frame);
  }

  applySettings();
  report({ s: "settings", settings });
  panel.toggle(q.has("operator"));
  if (!q.get("state")) link.connect(); // debug poses stay put without a server
  requestAnimationFrame(frame);
}
