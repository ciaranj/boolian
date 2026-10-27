export interface Settings {
  scale: number;
  offsetX: number;
  offsetY: number;
  glow: number;
  volume: number;
  reverb: number;
  fxWaver: number; // ghostly voice effects, 0 = off (see VoiceFx in audio/output.ts)
  fxFlange: number;
  fxEcho: number;
  fxShadow: number;
  extraLatencyMs: number;
  serverUrl: string;
  autoListen: boolean;
  attract: boolean; // he calls out now and then while nobody is talking to him
  palette: string;
  micDevice: string; // "" = the system default input
  vad: "silero" | "energy"; // how auto-listen decides someone is talking
  vadThreshold: number; // Silero speech probability needed (lower = more sensitive)
  vadPauseMs: number; // quiet after speech before they're taken to have finished
}

const KEY = "spectre.settings.v1";

function defaultServer(): string {
  const host = location.hostname || "localhost";
  return `ws://${host}:8765/ws`;
}

export function loadSettings(): Settings {
  const s: Settings = {
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    glow: 1,
    volume: 0.9,
    reverb: 0.35,
    fxWaver: 0,
    fxFlange: 0,
    fxEcho: 0,
    fxShadow: 0,
    extraLatencyMs: 0,
    serverUrl: defaultServer(),
    autoListen: false,
    attract: true,
    palette: "spectral",
    micDevice: "",
    vad: "silero",
    vadThreshold: 0.5,
    vadPauseMs: 700,
  };
  try {
    Object.assign(s, JSON.parse(localStorage.getItem(KEY) ?? "{}"));
  } catch {
    /* storage unavailable: defaults are fine */
  }
  const q = new URLSearchParams(location.search);
  const server = q.get("server");
  if (server) s.serverUrl = server;
  return s;
}

export function saveSettings(s: Settings) {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* ignore */
  }
}

/** Server URL with the optional ?token= from the page URL appended. */
export function serverUrlWithToken(url: string): string {
  const token = new URLSearchParams(location.search).get("token");
  if (!token) return url;
  return url + (url.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(token);
}
