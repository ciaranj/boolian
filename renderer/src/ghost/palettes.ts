type RGB = readonly [number, number, number];

export interface Palette {
  label: string;
  /** Body radial gradient, highlight → edge. */
  core: string;
  mid: string;
  edge: string;
  glow: RGB;
  /** Tinted shading towards the tail. */
  shade: RGB;
  shadeAlpha: number;
  /** Darker outline: keeps the silhouette readable when projected onto a pale surface. */
  rim: string;
  rimWidth: number;
  blush: RGB;
  blushAlpha: number;
  wispHues: readonly [number, number];
}

export const PALETTES = {
  spectral: {
    label: "Spectral (aqua)",
    core: "#e2fdf8", mid: "#79e6d5", edge: "#3a82e0",
    glow: [70, 220, 255], shade: [110, 70, 220], shadeAlpha: 0.45,
    rim: "#2a62c9", rimWidth: 3.5,
    blush: [255, 95, 160], blushAlpha: 0.75, wispHues: [180, 280],
  },
  violet: {
    label: "Violet",
    core: "#f3e6ff", mid: "#bf8cff", edge: "#7040dc",
    glow: [185, 110, 255], shade: [90, 40, 190], shadeAlpha: 0.45,
    rim: "#4d2aa8", rimWidth: 3.5,
    blush: [255, 100, 170], blushAlpha: 0.75, wispHues: [270, 300],
  },
  pumpkin: {
    label: "Pumpkin",
    core: "#ffefd6", mid: "#ffb55a", edge: "#f06a14",
    glow: [255, 150, 50], shade: [200, 60, 20], shadeAlpha: 0.35,
    rim: "#a8400c", rimWidth: 3.5,
    blush: [255, 90, 90], blushAlpha: 0.7, wispHues: [30, 50],
  },
  slime: {
    label: "Slime green",
    core: "#ecffd9", mid: "#9cef6a", edge: "#38a83f",
    glow: [120, 255, 90], shade: [20, 120, 60], shadeAlpha: 0.35,
    rim: "#217a35", rimWidth: 3.5,
    blush: [255, 120, 150], blushAlpha: 0.7, wispHues: [100, 140],
  },
  classic: {
    label: "Classic (pale, best on dark surfaces)",
    core: "#ffffff", mid: "#eef3ff", edge: "#b8c9f2",
    glow: [150, 210, 255], shade: [160, 140, 230], shadeAlpha: 0.28,
    rim: "#8fa6dc", rimWidth: 0,
    blush: [255, 140, 185], blushAlpha: 0.5, wispHues: [190, 265],
  },
} satisfies Record<string, Palette>;

export type PaletteName = keyof typeof PALETTES;

export const paletteOf = (name: string): Palette => PALETTES[name as PaletteName] ?? PALETTES.spectral;

export const rgba = ([r, g, b]: RGB, a: number) => `rgba(${r}, ${g}, ${b}, ${Math.min(1, Math.max(0, a))})`;
