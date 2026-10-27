import * as THREE from "three";

const VERT = /* glsl */ `
  attribute float aSize;
  attribute float aAlpha;
  attribute vec3 aColor;
  uniform float uScale;
  varying float vAlpha;
  varying vec3 vColor;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = aSize * uScale / -mv.z;
    gl_Position = projectionMatrix * mv;
    vAlpha = aAlpha;
    vColor = aColor;
  }`;
// A soft round dot with a brighter core, so it glows without a texture.
const FRAG = /* glsl */ `
  varying float vAlpha;
  varying vec3 vColor;
  void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    if (d > 1.0) discard;
    float a = (1.0 - d) * (1.0 - d) + 0.6 * exp(-d * d * 18.0);
    gl_FragColor = vec4(vColor, a * vAlpha);
  }`;

export interface Spawn {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  life: number; // seconds
  size: number; // world units
  color: THREE.Color;
  drag?: number; // fraction of speed lost per second
  lift?: number; // upward acceleration, units/s²
  alpha?: number;
}

/** A pool of glowing points: spawn() reuses the oldest slot, update() moves and fades them. */
export class Particles {
  readonly points: THREE.Points;
  private pos: Float32Array;
  private vel: Float32Array;
  private col: Float32Array;
  private size: Float32Array;
  private alpha: Float32Array;
  private age: Float32Array;
  private life: Float32Array;
  private baseSize: Float32Array;
  private baseAlpha: Float32Array;
  private drag: Float32Array;
  private lift: Float32Array;
  private next = 0;
  private uScale = { value: 1 };

  constructor(readonly capacity: number) {
    const n = capacity;
    this.pos = new Float32Array(n * 3);
    this.vel = new Float32Array(n * 3);
    this.col = new Float32Array(n * 3);
    this.size = new Float32Array(n);
    this.alpha = new Float32Array(n);
    this.age = new Float32Array(n);
    this.life = new Float32Array(n).fill(0);
    this.baseSize = new Float32Array(n);
    this.baseAlpha = new Float32Array(n);
    this.drag = new Float32Array(n);
    this.lift = new Float32Array(n);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(this.pos, 3));
    geo.setAttribute("aColor", new THREE.BufferAttribute(this.col, 3));
    geo.setAttribute("aSize", new THREE.BufferAttribute(this.size, 1));
    geo.setAttribute("aAlpha", new THREE.BufferAttribute(this.alpha, 1));
    this.points = new THREE.Points(
      geo,
      new THREE.ShaderMaterial({
        uniforms: { uScale: this.uScale },
        vertexShader: VERT,
        fragmentShader: FRAG,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    this.points.frustumCulled = false;
  }

  /** Pixels per world unit at distance 1, so sizes are in world units. */
  setViewport(heightPx: number, fovDeg: number) {
    this.uScale.value = heightPx / (2 * Math.tan(THREE.MathUtils.degToRad(fovDeg / 2)));
  }

  spawn(s: Spawn) {
    const i = this.next;
    this.next = (this.next + 1) % this.capacity;
    this.pos.set([s.x, s.y, s.z], i * 3);
    this.vel.set([s.vx, s.vy, s.vz], i * 3);
    this.col.set([s.color.r, s.color.g, s.color.b], i * 3);
    this.age[i] = 0;
    this.life[i] = s.life;
    this.baseSize[i] = s.size;
    this.baseAlpha[i] = s.alpha ?? 1;
    this.drag[i] = s.drag ?? 0.5;
    this.lift[i] = s.lift ?? 0;
  }

  update(dt: number, fade = 1) {
    for (let i = 0; i < this.capacity; i++) {
      if (this.life[i] <= 0) {
        this.alpha[i] = 0;
        continue;
      }
      this.age[i] += dt;
      const p = this.age[i] / this.life[i];
      if (p >= 1) {
        this.life[i] = 0;
        this.alpha[i] = 0;
        continue;
      }
      const k = Math.exp(-this.drag[i] * dt);
      const j = i * 3;
      this.vel[j] *= k;
      this.vel[j + 1] = this.vel[j + 1] * k + this.lift[i] * dt;
      this.vel[j + 2] *= k;
      this.pos[j] += this.vel[j] * dt;
      this.pos[j + 1] += this.vel[j + 1] * dt;
      this.pos[j + 2] += this.vel[j + 2] * dt;
      // Quick fade in, long fade out; shrink a little as it goes.
      const env = Math.min(1, p / 0.12) * (1 - p) ** 1.5;
      this.alpha[i] = this.baseAlpha[i] * env * fade;
      this.size[i] = this.baseSize[i] * (1 - 0.4 * p);
    }
    const a = this.points.geometry.attributes;
    a.position.needsUpdate = a.aColor.needsUpdate = a.aSize.needsUpdate = a.aAlpha.needsUpdate = true;
  }

  /** For particles placed by hand each frame (orbs) rather than simulated. */
  place(i: number, x: number, y: number, z: number, size: number, alpha: number, color: THREE.Color) {
    this.pos.set([x, y, z], i * 3);
    this.col.set([color.r, color.g, color.b], i * 3);
    this.size[i] = size;
    this.alpha[i] = alpha;
  }

  commit() {
    const a = this.points.geometry.attributes;
    a.position.needsUpdate = a.aColor.needsUpdate = a.aSize.needsUpdate = a.aAlpha.needsUpdate = true;
  }
}
