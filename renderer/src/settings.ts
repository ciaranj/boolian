export interface Settings {
  scale: number;
  offsetX: number;
  offsetY: number;
  glow: number;
  volume: number;
  reverb: number;
  extraLatencyMs: number;
  serverUrl: string;
  autoListen: boolean;
  palette: string;
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
    extraLatencyMs: 0,
    serverUrl: defaultServer(),
    autoListen: false,
    palette: "spectral",
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
