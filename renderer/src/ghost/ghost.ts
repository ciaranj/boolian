import { Spring, approach, clamp, easeInOut, easeOutBack, lerp, mulberry32 } from "./anim";
import { PALETTES, rgba, type Palette } from "./palettes";

export type ConvState = "idle" | "listening" | "thinking" | "speaking" | "error" | "sleeping";
export const EMOTIONS = ["neutral", "happy", "curious", "surprised", "mischievous", "confused"] as const;
export const ACTIONS = ["none", "bounce", "tilt", "spin", "hide", "peek"] as const;
export type Emotion = (typeof EMOTIONS)[number];
export type Action = (typeof ACTIONS)[number];
export type Rude = "raspberry" | "fart";

export interface GhostSettings {
  scale: number;
  offsetX: number; // fraction of width
  offsetY: number; // fraction of height
  clipLeft: number; // the viewport mask, as fractions of width (0 = no mask on that side)
  clipRight: number;
  glow: number;
}

interface Face {
  eyeScale: number;
  eyeSquash: number;
  lid: number; // 0 open .. 1 closed
  browL: number; // lift
  browR: number;
  browAngle: number; // + inner ends down (mischief), - worried
  smile: number; // -1 frown .. 1 grin
  skew: number; // lopsided grin
  round: number; // "o" mouth
  blush: number;
  tilt: number;
  happyArc: number; // ^ ^ eyes
}

const NEUTRAL: Face = {
  eyeScale: 1, eyeSquash: 1, lid: 0, browL: 0, browR: 0, browAngle: 0,
  smile: 0.45, skew: 0, round: 0, blush: 0.25, tilt: 0, happyArc: 0,
};

const FACES: Record<Emotion, Partial<Face>> = {
  neutral: {},
  happy: { smile: 1, blush: 0.7, browL: 0.25, browR: 0.25, happyArc: 1, eyeScale: 1.02 },
  curious: { browL: 0.9, browR: 0.05, eyeScale: 1.12, smile: 0.3, tilt: 0.13, skew: 0.2 },
  surprised: { eyeScale: 1.38, browL: 1, browR: 1, round: 1, smile: 0, blush: 0.4 },
  mischievous: { browAngle: 0.45, eyeSquash: 0.72, smile: 0.95, skew: 0.9, tilt: -0.09, blush: 0.45, browL: -0.1, browR: 0.25 },
  confused: { browL: 0.85, browR: -0.35, browAngle: -0.1, smile: -0.15, skew: -0.5, tilt: 0.16, eyeScale: 1.05 },
};

const INK = "#191330";
const DRAMATIC: Action[] = ["spin", "hide", "peek"];
const RUDE_AFTERGLOW = 1.2; // seconds the smug face and fart cloud linger after the noise

interface ActiveAction {
  kind: "spin" | "hide" | "peek";
  start: number;
  dur: number;
  edge: -1 | 1; // peek: which way he leans out (-1 from behind a left edge, 1 a right one)
}

interface Pose {
  dx: number;
  dy: number;
  rot: number;
  spin: number; // turn about the vertical axis (2D just adds it to rot)
  alpha: number;
  sx: number;
  sy: number;
  pin: number; // 0..1: how much the idle sway is held still (so a peek stays tucked behind its edge)
}

/** Where the body sits this frame, in ghost units around the screen centre (y down). */
export interface Layout {
  x: number;
  y: number;
  rot: number;
  spin: number;
  sx: number;
  sy: number;
  alpha: number; // includes the sleepy dimming
  fade: number; // just the hide/peek fade
  glow: number;
  calm: number;
}

export interface Wisp {
  x: number;
  y: number;
  vy: number;
  size: number;
  phase: number;
  hue: 0 | 1;
}

export class Ghost {
  state: ConvState = "sleeping";
  /** Playback loudness 0..1 at the speaker, set every frame by the audio player. */
  speechLevel = 0;
  /** Visitor microphone loudness 0..1 while listening. */
  micLevel = 0;
  settings: GhostSettings = { scale: 1, offsetX: 0, offsetY: 0, clipLeft: 0, clipRight: 0, glow: 1 };
  /** Screen width / height, set by whoever owns the canvas; the peek needs it to find the mask edges. */
  aspect = 16 / 9;
  palette: Palette = PALETTES.spectral;

  private t = 0;
  private rand: () => number;
  private face: Face = { ...NEUTRAL };
  private emotion: Emotion = "neutral";
  private intensity = 0;
  private emotionUntil = 0;
  private gaze = { x: 0, y: 0 };
  private gazeTarget = { x: 0, y: 0 };
  private nextGlance = 2;
  private blink = 0;
  private blinkStart = -1;
  private nextBlink = 1.5;
  private doubleBlink = false;
  private mouth = 0;
  private lean = 0;
  private glowBoost = 0;
  private bounce = new Spring(160, 9);
  private squash = new Spring(240, 11);
  private tilt = new Spring(70, 6);
  private action: ActiveAction | null = null;
  private lastDramatic = -100;
  private nextFlourish = 40;
  private stateSince = 0;
  private wisps: Wisp[] = [];
  private rudeNoise: { kind: Rude; start: number; dur: number } | null = null;
  private lastX = 0;
  private drag = 0;

  constructor(seed = Date.now()) {
    this.rand = mulberry32(seed);
    for (let i = 0; i < 34; i++) this.wisps.push(this.newWisp(true));
  }

  setState(state: ConvState) {
    if (state === this.state) return;
    const prev = this.state;
    this.state = state;
    this.stateSince = this.t;
    if (state === "listening") {
      this.squash.kick(-1.2);
      this.gazeTarget = { x: 0, y: 0.15 };
    }
    if (state === "thinking") this.gazeTarget = { x: 0.6, y: -0.8 };
    if (state === "speaking" || state === "idle") this.gazeTarget = { x: 0, y: 0 };
    if (prev === "sleeping" && state === "idle") {
      this.express("surprised", "bounce", 0.8, 1200);
    }
  }

  express(emotion: Emotion, action: Action = "none", intensity = 0.8, durationMs = 4500) {
    this.emotion = emotion;
    this.intensity = clamp(intensity, 0, 1);
    this.emotionUntil = this.t + durationMs / 1000;
    if (emotion === "surprised") this.glowBoost = 0.8;
    this.doAction(action, this.intensity);
  }

  /** A rude noise is being heard: tongue out for a raspberry; a jolt, a shocked face, then a smug grin for a fart. */
  rude(kind: Rude, seconds: number) {
    this.rudeNoise = { kind, start: this.t, dur: Math.max(0.4, seconds) };
    if (kind === "fart") {
      this.bounce.kick(-520);
      this.squash.kick(2.8);
    } else {
      this.squash.kick(1.5);
    }
  }

  /** Let the current emotion fade shortly (e.g. when speech finishes). */
  relaxSoon(seconds: number) {
    this.emotionUntil = Math.min(this.emotionUntil, this.t + seconds);
  }

  /** `force` skips the rate limit; for operator buttons. */
  doAction(action: Action, intensity = 0.8, force = false) {
    if (action === "none") return;
    if (DRAMATIC.includes(action)) {
      // Rate-limit the big moves so a chatty model can't turn Boolian into a washing machine.
      if (this.action || (!force && this.t - this.lastDramatic < 6)) return;
      this.lastDramatic = this.t;
      const dur = { spin: 1.0, hide: 2.4, peek: 2.8 }[action as "spin" | "hide" | "peek"];
      this.action = { kind: action as ActiveAction["kind"], start: this.t, dur, edge: this.peekEdge() };
      return;
    }
    if (action === "bounce") {
      this.bounce.kick(-650 * (0.5 + intensity));
      this.squash.kick(2.5);
    }
    if (action === "tilt") this.tilt.kick((this.rand() < 0.5 ? -1 : 1) * 3.5 * (0.5 + intensity));
  }

  update(dt: number) {
    dt = Math.min(dt, 0.1);
    this.t += dt;
    const t = this.t;

    if (this.emotion !== "neutral" && t > this.emotionUntil) this.emotion = "neutral";
    this.updateFace(dt);
    this.updateEyes(dt);

    // The noise isn't speech, so the mouth doesn't flap to it (a raspberry shows the tongue instead).
    const target = this.state === "speaking" && !this.rudeNow ? clamp(this.speechLevel, 0, 1) : 0;
    this.mouth = approach(this.mouth, target, target > this.mouth ? 28 : 13, dt);

    this.lean = approach(this.lean, this.state === "listening" ? 1 : 0, 5, dt);
    this.glowBoost = approach(this.glowBoost, 0, 1.5, dt);

    this.bounce.step(dt);
    this.squash.step(dt);
    this.tilt.step(dt);
    if (this.action && t - this.action.start > this.action.dur) this.action = null;
    if (this.rudeNoise && t - this.rudeNoise.start > this.rudeNoise.dur + RUDE_AFTERGLOW) this.rudeNoise = null;

    if (this.state === "speaking" && this.speechLevel > 0.55 && this.rand() < dt * 1.2) this.squash.kick(1.2);

    // Occasional idle flourish keeps an empty porch interesting and draws visitors over.
    if (this.state === "idle" && t > this.nextFlourish && t - this.stateSince > 8) {
      const pick = this.rand();
      if (pick < 0.3) this.express("curious", "tilt", 0.6, 2500);
      else if (pick < 0.55) this.express("happy", "bounce", 0.6, 2000);
      else if (pick < 0.8) this.express("mischievous", "peek", 0.7, 3500);
      else this.express("happy", "spin", 0.7, 2000);
      this.nextFlourish = t + 25 + this.rand() * 30;
    }

    for (const w of this.wisps) {
      w.y -= w.vy * dt;
      w.x += Math.sin(t * 0.7 + w.phase) * 6 * dt;
      if (w.y < -320) Object.assign(w, this.newWisp(false));
    }
  }

  private newWisp(anywhere: boolean): Wisp {
    return {
      x: (this.rand() - 0.5) * 520,
      y: anywhere ? -320 + this.rand() * 620 : 300,
      vy: 8 + this.rand() * 18,
      size: 0.8 + this.rand() * 2.2,
      phase: this.rand() * Math.PI * 2,
      hue: this.rand() < 0.6 ? 0 : 1, // index into palette.wispHues
    };
  }

  private updateFace(dt: number) {
    const emo = FACES[this.emotion];
    const k = this.emotion === "neutral" ? 0 : this.intensity;
    const target: Face = { ...NEUTRAL };
    for (const key of Object.keys(emo) as (keyof Face)[]) target[key] = lerp(NEUTRAL[key], emo[key]!, k);

    switch (this.state) {
      case "listening":
        target.eyeScale *= 1.08 + this.micLevel * 0.12;
        target.browL += 0.25;
        target.browR += 0.25;
        target.smile = Math.max(target.smile, 0.35);
        break;
      case "thinking":
        target.browL += 0.45;
        target.skew += 0.35;
        target.smile = Math.min(target.smile, 0.2);
        target.tilt += 0.06;
        break;
      case "error":
        target.lid = 0.35;
        target.browAngle = -0.35;
        target.smile = 0.05;
        break;
      case "sleeping":
        target.lid = 0.82;
        target.smile = 0.3;
        target.tilt += 0.07;
        target.happyArc = 0;
        break;
    }
    // Squinty ^^ eyes look odd while talking a lot; soften them during speech.
    if (this.state === "speaking") target.happyArc *= 0.6;

    const r = this.rudeNoise;
    if (r) {
      const since = this.t - r.start;
      if (r.kind === "raspberry" && since < r.dur) {
        Object.assign(target, { happyArc: 1, blush: 1, browAngle: 0.35, browL: 0.3, browR: 0.3, smile: 0.2, round: 0, skew: 0 });
      } else if (r.kind === "fart" && since < 0.3) {
        Object.assign(target, { eyeScale: 1.35, browL: 1, browR: 1, round: 1, smile: 0, happyArc: 0, lid: 0 });
      } else if (r.kind === "fart") {
        Object.assign(target, { browAngle: 0.45, eyeSquash: 0.7, smile: 0.95, skew: 0.9, blush: 0.9, happyArc: 0, round: 0, lid: 0.15 });
      }
    }

    for (const key of Object.keys(target) as (keyof Face)[]) {
      this.face[key] = approach(this.face[key], target[key], 7, dt);
    }
  }

  private updateEyes(dt: number) {
    const t = this.t;
    if (t > this.nextGlance && (this.state === "idle" || this.state === "speaking")) {
      const wide = this.state === "idle" ? 0.8 : 0.35;
      this.gazeTarget = this.rand() < 0.4 ? { x: 0, y: 0 } : { x: (this.rand() - 0.5) * 2 * wide, y: (this.rand() - 0.5) * wide };
      this.nextGlance = t + 1.8 + this.rand() * 4;
    }
    if (this.state === "thinking") {
      this.gazeTarget = { x: 0.6 + Math.sin(t * 1.3) * 0.2, y: -0.8 };
    }
    this.gaze.x = approach(this.gaze.x, this.gazeTarget.x, 9, dt);
    this.gaze.y = approach(this.gaze.y, this.gazeTarget.y, 9, dt);

    if (this.blinkStart < 0 && t > this.nextBlink && this.state !== "sleeping") {
      this.blinkStart = t;
    }
    if (this.blinkStart >= 0) {
      const p = (t - this.blinkStart) / 0.15;
      this.blink = p < 1 ? Math.sin(p * Math.PI) : 0;
      if (p >= 1) {
        this.blinkStart = -1;
        if (!this.doubleBlink && this.rand() < 0.2) {
          this.doubleBlink = true;
          this.nextBlink = t + 0.12;
        } else {
          this.doubleBlink = false;
          this.nextBlink = t + 1.8 + this.rand() * 4.5;
        }
      }
    }
  }

  private pose(): Pose {
    const pose: Pose = { dx: 0, dy: 0, rot: 0, spin: 0, alpha: 1, sx: 1, sy: 1, pin: 0 };
    const r = this.rudeNoise;
    if (r) {
      const since = this.t - r.start;
      if (r.kind === "raspberry" && since < r.dur) pose.dx += Math.sin(since * 70) * 2.2; // buzzing
      if (r.kind === "fart") pose.rot += Math.sin(since * 28) * 0.06 * Math.exp(-since * 3); // a startled wobble
    }
    const a = this.action;
    if (!a) return pose;
    const p = clamp((this.t - a.start) / a.dur, 0, 1);
    if (a.kind === "spin") {
      pose.spin = easeInOut(p) * Math.PI * 2;
      pose.sx = 1 - 0.18 * Math.sin(p * Math.PI);
      pose.dy = -35 * Math.sin(p * Math.PI);
    } else if (a.kind === "hide") {
      if (p < 0.2) {
        const q = p / 0.2;
        pose.alpha = 1 - q;
        pose.dy = 50 * q;
        pose.sy = 1 - 0.25 * q;
      } else if (p < 0.65) {
        pose.alpha = 0;
        pose.dy = 50;
      } else {
        const q = (p - 0.65) / 0.35;
        pose.alpha = q;
        pose.dy = 50 * (1 - easeOutBack(q));
      }
    } else if (a.kind === "peek") {
      const side = this.peekX(a.edge);
      const lean = -0.4 * a.edge;
      if (p < 0.15) {
        pose.alpha = 1 - p / 0.15;
      } else if (p < 0.3) {
        const q = (p - 0.15) / 0.15;
        pose.alpha = q;
        pose.dx = side;
        pose.rot = lean;
        pose.pin = 1;
      } else if (p < 0.62) {
        pose.dx = side + Math.sin((p - 0.3) * 30) * 6;
        pose.rot = lean + Math.sin((p - 0.3) * 20) * 0.04;
        pose.pin = 1;
      } else {
        const q = easeInOut((p - 0.62) / 0.38);
        pose.dx = side * (1 - q);
        pose.rot = lean * (1 - q);
        pose.pin = 1 - q;
      }
    }
    return pose;
  }

  /** The side to peek from: the masked edge nearest where he floats, else his left (-1) as before. */
  private peekEdge(): -1 | 1 {
    const { clipLeft, clipRight, offsetX } = this.settings;
    if (clipLeft <= 0 && clipRight <= 0) return -1;
    if (clipLeft <= 0) return 1;
    if (clipRight <= 0) return -1;
    const cx = 0.5 + offsetX;
    return cx - clipLeft <= 1 - clipRight - cx ? -1 : 1;
  }

  /** Sideways shift (ghost units) that tucks him behind that side's mask edge with an eye showing. */
  private peekX(edge: -1 | 1) {
    const s = this.settings;
    const clip = edge < 0 ? s.clipLeft : s.clipRight;
    if (clip <= 0) return 260 * edge; // no mask that side: just pop up beside where he was
    const edgeX = edge < 0 ? clip : 1 - clip; // fraction of width
    const unitsPerWidth = (this.aspect * 700) / s.scale; // matches draw(): one unit is h / 700 * scale px
    return (edgeX - 0.5 - s.offsetX) * unitsPerWidth + 15 * edge; // centre just behind the edge
  }

  /** Advances the sideways drag, so call it once per frame (draw() does). */
  layout(): Layout {
    const t = this.t;
    const pose = this.pose();
    const calm = this.state === "listening" ? 0.4 : this.state === "sleeping" ? 0.6 : 1;

    const bob = Math.sin(t * 1.25) * 11 * calm + Math.sin(t * 0.47) * 4;
    const sway = (Math.sin(t * 0.5) * 18 + Math.sin(t * 0.21) * 12) * calm;
    const x = sway * (1 - pose.pin) + pose.dx;
    this.drag = approach(this.drag, clamp((x - this.lastX) * 2.5, -25, 25), 4, 1 / 60);
    this.lastX = x;

    const rot = pose.rot + this.face.tilt + this.tilt.x * 0.05 + Math.sin(t * 0.5 + 1) * 0.03 * calm - this.drag * 0.002;
    const stretch = this.squash.x * 0.04 + this.lean * 0.04;
    const sleepy = this.state === "sleeping" || this.state === "error";
    return {
      x,
      y: bob + this.bounce.x + pose.dy - 30,
      rot,
      spin: pose.spin,
      sx: pose.sx * (1 - stretch * 0.5) * (1 + this.lean * 0.04),
      sy: pose.sy * (1 + stretch) * (1 + this.lean * 0.04),
      alpha: pose.alpha * (sleepy ? 0.78 : 1),
      fade: pose.alpha,
      glow:
        this.settings.glow * (0.75 + 0.25 * Math.sin(t * 1.4)) + this.speechLevel * 0.5 + this.glowBoost + (this.state === "thinking" ? 0.2 * Math.sin(t * 5) : 0),
      calm,
    };
  }

  get time() {
    return this.t;
  }
  get currentEmotion() {
    return this.emotion;
  }
  get currentIntensity() {
    return this.emotion === "neutral" ? 0 : this.intensity;
  }
  get gazeX() {
    return this.gaze.x;
  }
  get leanAmount() {
    return this.lean;
  }
  get dragAmount() {
    return this.drag;
  }
  /** The big move in progress (spin, hide, peek), if any. */
  get currentAction(): "spin" | "hide" | "peek" | null {
    return this.action?.kind ?? null;
  }
  /** The rude noise being heard right now, if any (its after-effects linger a little longer). */
  get currentRude(): { kind: Rude; since: number; dur: number } | null {
    const r = this.rudeNoise;
    return r ? { kind: r.kind, since: this.t - r.start, dur: r.dur } : null;
  }
  private get rudeNow(): boolean {
    const r = this.rudeNoise;
    return !!r && this.t - r.start < r.dur;
  }
  get wispList(): readonly Wisp[] {
    return this.wisps;
  }

  draw(ctx: CanvasRenderingContext2D, w: number, h: number, outline = false) {
    const s = this.settings;
    const unit = (h / 700) * s.scale;
    this.aspect = w / h;
    const l = this.layout();
    const cx = w / 2 + s.offsetX * w + l.x * unit;
    const cy = h / 2 + s.offsetY * h + l.y * unit;

    ctx.save();
    ctx.translate(w / 2 + s.offsetX * w, h / 2 + s.offsetY * h);
    ctx.scale(unit, unit);
    this.drawWisps(ctx);
    ctx.restore();

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(l.rot + l.spin);
    ctx.scale(unit * l.sx, unit * l.sy);
    ctx.globalAlpha = l.alpha;

    if (outline) {
      this.bodyPath(ctx);
      ctx.strokeStyle = "#4af";
      ctx.lineWidth = 2 / unit;
      ctx.stroke();
    } else {
      this.drawBody(ctx, l.glow, unit);
      this.drawFartCloud(ctx);
      this.drawFace(ctx);
      if (this.state === "thinking") this.drawThinkingDots(ctx);
      if (this.state === "sleeping") this.drawZzz(ctx);
    }
    ctx.restore();
  }

  private drawWisps(ctx: CanvasRenderingContext2D) {
    for (const w of this.wisps) {
      const a = 0.12 + 0.18 * (0.5 + 0.5 * Math.sin(this.t * 1.7 + w.phase));
      const fade = clamp((300 - w.y) / 80, 0, 1) * clamp((w.y + 320) / 120, 0, 1);
      ctx.fillStyle = `hsla(${this.palette.wispHues[w.hue]}, 90%, 75%, ${a * fade})`;
      ctx.beginPath();
      ctx.arc(w.x, w.y, w.size, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  private bodyPath(ctx: CanvasRenderingContext2D) {
    const t = this.t;
    const R = 100;
    const H = 150;
    const tx = -this.drag;
    const n = 4;
    const right = R * 0.98 + tx;
    const seg = (R * 0.98 * 2) / n;
    ctx.beginPath();
    ctx.moveTo(-R, 0);
    ctx.arc(0, 0, R, Math.PI, 0);
    ctx.bezierCurveTo(R * 1.02, 55, R * 1.08 + tx * 0.5, 110, right, H);
    for (let i = 0; i < n; i++) {
      const x0 = right - i * seg;
      const x1 = x0 - seg;
      const wave = Math.sin(t * 3.1 + i * 1.7) * 7;
      const endY = i < n - 1 ? H - 4 + Math.sin(t * 2.4 + i * 2.1) * 4 : H;
      ctx.bezierCurveTo(x0 - seg * 0.1, H + 36 + wave, x1 + seg * 0.1, H + 36 + wave, x1, endY);
    }
    ctx.bezierCurveTo(-R * 1.08 + tx * 0.5, 110, -R * 1.02, 55, -R, 0);
    ctx.closePath();
  }

  /** Arm rotation in canvas radians (y down), about its shoulder at (side * 84, 40). */
  armAngle(side: -1 | 1) {
    const t = this.t;
    const talk = this.state === "speaking" ? this.speechLevel * 0.35 * Math.sin(t * 9 + side) : 0;
    const cheer = this.emotion === "happy" || this.emotion === "surprised" ? 0.5 * this.intensity : 0;
    return side * (0.55 + cheer + Math.sin(t * 1.8 + side) * 0.12 + talk) - this.lean * side * 0.25;
  }

  private drawArm(ctx: CanvasRenderingContext2D, side: -1 | 1, fill: CanvasGradient) {
    const angle = this.armAngle(side);
    ctx.save();
    ctx.translate(side * 84, 40);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.ellipse(side * 18, 20, 15, 30, -side * 0.15, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    if (this.palette.rimWidth > 0) ctx.stroke();
    ctx.restore();
  }

  private drawBody(ctx: CanvasRenderingContext2D, glow: number, unit: number) {
    const pal = this.palette;
    const grad = ctx.createRadialGradient(-28, -40, 10, 0, 20, 230);
    grad.addColorStop(0, pal.core);
    grad.addColorStop(0.5, pal.mid);
    grad.addColorStop(1, pal.edge);

    // Soft outer halo layer: a slightly larger translucent body behind the real one.
    ctx.save();
    ctx.translate(0, -4);
    ctx.scale(1.04, 1.035);
    this.bodyPath(ctx);
    ctx.fillStyle = rgba(pal.glow, 0.1 * glow);
    ctx.shadowColor = rgba(pal.glow, 0.6 * glow);
    ctx.shadowBlur = 70 * glow * unit;
    ctx.fill();
    ctx.restore();

    ctx.save();
    ctx.shadowColor = rgba(pal.glow, 0.75 * glow);
    ctx.shadowBlur = 35 * glow * unit;
    ctx.strokeStyle = pal.rim;
    ctx.lineWidth = pal.rimWidth;
    ctx.lineJoin = "round";
    this.drawArm(ctx, -1, grad);
    this.drawArm(ctx, 1, grad);
    this.bodyPath(ctx);
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.restore();

    // Tinted inner shading towards the tail for depth.
    ctx.save();
    this.bodyPath(ctx);
    ctx.clip();
    const shade = ctx.createLinearGradient(0, 30, 0, 190);
    shade.addColorStop(0, rgba(pal.shade, 0));
    shade.addColorStop(1, rgba(pal.shade, pal.shadeAlpha));
    ctx.fillStyle = shade;
    ctx.fillRect(-140, 0, 280, 220);
    ctx.restore();

    if (pal.rimWidth > 0) {
      ctx.save();
      this.bodyPath(ctx);
      ctx.strokeStyle = pal.rim;
      ctx.lineWidth = pal.rimWidth;
      ctx.lineJoin = "round";
      ctx.stroke();
      ctx.restore();
    }
  }

  /** In ghost units around the body centre; the 3D renderer paints it onto a texture. */
  drawFace(ctx: CanvasRenderingContext2D) {
    const f = this.face;
    const g = this.gaze;

    // Cheeks
    if (f.blush > 0.02) {
      for (const side of [-1, 1]) {
        const bx = side * 60 + g.x * 3;
        const blush = ctx.createRadialGradient(bx, 22, 0, bx, 22, 20);
        blush.addColorStop(0, rgba(this.palette.blush, this.palette.blushAlpha * f.blush));
        blush.addColorStop(1, rgba(this.palette.blush, 0));
        ctx.fillStyle = blush;
        ctx.fillRect(bx - 22, 0, 44, 44);
      }
    }

    ctx.fillStyle = INK;
    ctx.strokeStyle = INK;
    ctx.lineCap = "round";

    // Eyes
    const closed = Math.max(this.blink, f.lid);
    for (const side of [-1, 1] as const) {
      const ex = side * 34 + g.x * 7;
      const ey = -8 + g.y * 6;
      const rx = 13 * f.eyeScale;
      const fullRy = 19 * f.eyeScale * f.eyeSquash;
      const ry = fullRy * (1 - closed);
      if (f.happyArc > 0.5 && this.blink < 0.5) {
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.arc(ex, ey + 6, rx, Math.PI * 1.15, Math.PI * 1.85);
        ctx.stroke();
      } else if (ry < Math.max(2.5, fullRy * 0.32)) {
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.moveTo(ex - rx, ey + fullRy * 0.3);
        ctx.quadraticCurveTo(ex, ey + fullRy * 0.3 + 5, ex + rx, ey + fullRy * 0.3);
        ctx.stroke();
      } else {
        const cy = ey + (fullRy - ry) * 0.5;
        ctx.beginPath();
        ctx.ellipse(ex, cy, rx, ry, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "rgba(255,255,255,0.95)";
        ctx.beginPath();
        ctx.arc(ex - rx * 0.32 + g.x * 2, cy - ry * 0.42, rx * 0.34, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(ex + rx * 0.3, cy + ry * 0.38, rx * 0.15, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = INK;
      }

      // Brows
      const lift = side < 0 ? f.browL : f.browR;
      ctx.save();
      ctx.translate(side * 34 + g.x * 4, -42 - lift * 10 - (f.eyeScale - 1) * 14);
      ctx.rotate(-side * f.browAngle);
      ctx.globalAlpha *= 0.85;
      ctx.lineWidth = 5.5;
      ctx.beginPath();
      ctx.moveTo(-12, 2);
      ctx.quadraticCurveTo(0, -4, 12, 2);
      ctx.stroke();
      ctx.restore();
    }

    // Mouth
    const o = this.mouth;
    const mx = g.x * 3;
    const my = 34;
    const w = lerp(30 + f.smile * 10, 15, f.round) + o * 5;
    const h = 3 + o * 30 + f.round * 10;
    const sk = f.skew * 8;
    const cl = my - f.smile * 5 + sk * 0.3;
    const cr = my - f.smile * 5 - sk;
    ctx.save();
    ctx.translate(mx, 0);
    if (this.rudeNow && this.rudeNoise!.kind === "raspberry") {
      // Pursed lips with the tongue poking out, fluttering.
      const flutter = Math.sin(this.t * 60) * 1.5;
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(-14, my);
      ctx.quadraticCurveTo(0, my - 5, 14, my);
      ctx.stroke();
      ctx.fillStyle = "#ff7fa8";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(-10, my + 1);
      ctx.lineTo(-10, my + 12 + flutter);
      ctx.quadraticCurveTo(-10, my + 24 + flutter, 0, my + 24 + flutter);
      ctx.quadraticCurveTo(10, my + 24 + flutter, 10, my + 12 + flutter);
      ctx.lineTo(10, my + 1);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.strokeStyle = "#d9507c";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, my + 5);
      ctx.lineTo(0, my + 16 + flutter);
      ctx.stroke();
    } else if (o < 0.07 && f.round < 0.3) {
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(-w / 2, cl);
      ctx.quadraticCurveTo(f.skew * 6, my + f.smile * 14, w / 2, cr);
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.moveTo(-w / 2, cl);
      ctx.quadraticCurveTo(0, my + f.smile * 6 - f.round * 8, w / 2, cr);
      ctx.quadraticCurveTo(f.skew * 4, my + h * 1.5 + f.smile * 6, -w / 2, cl);
      ctx.closePath();
      ctx.fill();
      if (o > 0.3) {
        ctx.save();
        ctx.clip();
        ctx.fillStyle = "#ff8fb0";
        ctx.beginPath();
        ctx.ellipse(f.skew * 3, my + h * 0.95, w * 0.3, h * 0.35, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    }
    ctx.restore();
  }

  /** A little green cloud puffing out from under his tail (the 3D view adds particles too). */
  private drawFartCloud(ctx: CanvasRenderingContext2D) {
    const r = this.rudeNoise;
    if (!r || r.kind !== "fart") return;
    const p = (this.t - r.start) / (r.dur + RUDE_AFTERGLOW);
    if (p >= 1) return;
    for (let i = 0; i < 7; i++) {
      const side = i % 2 ? 1 : -1;
      const x = side * (20 + i * 14) * (0.4 + p * 1.4);
      const y = 175 + (i % 3) * 8 + p * 25 - p * p * 40;
      const rad = (10 + i * 2) * (0.5 + p * 1.6);
      ctx.fillStyle = `rgba(${150 + i * 8}, 215, ${80 + i * 6}, ${0.45 * (1 - p) * Math.min(1, p * 8)})`;
      ctx.beginPath();
      ctx.arc(x, y, rad, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  /** In ghost units around the body centre; the 3D renderer paints it onto a texture. */
  drawThinkingDots(ctx: CanvasRenderingContext2D) {
    for (let i = 0; i < 3; i++) {
      const a = this.t * 2.2 + (i * Math.PI * 2) / 3;
      const x = 70 + Math.cos(a) * 30;
      const y = -125 + Math.sin(a) * 10;
      ctx.fillStyle = rgba(this.palette.glow, 0.55 + 0.4 * Math.sin(a));
      ctx.beginPath();
      ctx.arc(x, y, 5 + Math.sin(a) * 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  /** In ghost units around the body centre; the 3D renderer paints it onto a texture. */
  drawZzz(ctx: CanvasRenderingContext2D) {
    ctx.font = "bold 22px ui-rounded, system-ui, sans-serif";
    for (let i = 0; i < 3; i++) {
      const p = (this.t * 0.35 + i / 3) % 1;
      ctx.fillStyle = rgba(this.palette.glow, 0.7 * Math.sin(p * Math.PI));
      ctx.fillText("z", 80 + p * 40 + i * 4, -90 - p * 70);
    }
  }
}
