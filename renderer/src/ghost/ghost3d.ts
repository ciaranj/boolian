import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { clamp } from "./anim";
import type { Emotion, Ghost } from "./ghost";
import type { Palette } from "./palettes";
import { Particles } from "./particles";

/*
 * Boolian in three.js. All behaviour (face, gaze, actions, springs) still comes from Ghost; this only
 * draws it. World units are ghost units / 100 with y up, and the screen is 7 units tall, matching the
 * 2D renderer's h / 700, so scale and offset settings put him in the same place.
 */

const COLS = 72;
const DOME = 14; // rows from the crown to the widest point
const SKIRT = 40; // rows from there to the hem
const ROWS = 1 + DOME + SKIRT;
const DEPTH = 0.8; // flattened front to back, so the face reads well
const FACE_BOX = { x0: -100, y0: -90, size: 200 }; // ghost units covered by the face texture
const OVERLAY_BOX = { x0: 0, y0: -210, size: 180 }; // thinking dots and zzz, above his right shoulder
const VIEW_H = 7;
const FACE_LAYER = 1; // drawn after the bloom, so a bright glow never washes out his expression
const FOV = 30;

const MOOD: Record<Emotion, string | null> = {
  neutral: null, // palette glow
  happy: "#ffae42",
  curious: "#b47cff",
  surprised: "#ffffff",
  mischievous: "#4dff5e",
  confused: "#5c8dff",
};

const WHITE = new THREE.Color(1, 1, 1);
const rgb = ([r, g, b]: readonly [number, number, number]) => new THREE.Color(`rgb(${r}, ${g}, ${b})`);

const OUTLINE_VERT = /* glsl */ `
  uniform float width;
  void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position + normal * width, 1.0);
  }`;
const OUTLINE_FRAG = /* glsl */ `
  uniform vec3 color;
  uniform float opacity;
  void main() { gl_FragColor = vec4(color, opacity); }`;

const HALO_VERT = /* glsl */ `
  uniform float width;
  varying vec3 vNormal;
  varying vec3 vView;
  varying float vY;
  void main() {
    vY = position.y;
    vec4 mv = modelViewMatrix * vec4(position + normal * width, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vView = normalize(-mv.xyz);
    gl_Position = projectionMatrix * mv;
  }`;
// Drawn on the back faces of a fattened shell: brightest next to the body, fading to nothing at its edge.
const HALO_FRAG = /* glsl */ `
  uniform vec3 color;
  uniform float strength;
  varying vec3 vNormal;
  varying vec3 vView;
  varying float vY;
  void main() {
    float d = -dot(normalize(vNormal), normalize(vView));
    // d² falls off smoothly with distance from the middle (d itself drops sharply at the shell's edge).
    float tail = smoothstep(-1.85, -0.9, vY); // wisps away towards the hem instead of stopping in a line
    gl_FragColor = vec4(color, pow(clamp(d * d / 0.4, 0.0, 1.0), 2.0) * strength * tail);
  }`;

function softDot(size: number, stretch = 1): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = size;
  c.height = Math.round(size / stretch);
  const g = c.getContext("2d")!;
  g.scale(1, 1 / stretch);
  const grad = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.4, "rgba(255,255,255,0.35)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function paintedTexture(px: number) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = px;
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return { canvas, ctx: canvas.getContext("2d")!, tex };
}

export class Ghost3D {
  readonly canvas: HTMLCanvasElement;
  private renderer: THREE.WebGLRenderer;
  private composer: EffectComposer | null = null;
  private bloom: UnrealBloomPass | null = null;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 100);
  private faceCamera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 100);
  private root = new THREE.Group(); // projection settings
  private mover = new THREE.Group(); // bob, sway, tilt, squash
  private spinner = new THREE.Group(); // turning about the vertical axis

  private bodyGeo = new THREE.BufferGeometry();
  private rest: Float32Array; // body vertex positions at rest, in ghost units (y down)
  private bodyMat: THREE.MeshStandardMaterial;
  private faceMat: THREE.MeshBasicMaterial;
  private outlineMat: THREE.ShaderMaterial;
  private haloMat: THREE.ShaderMaterial;
  private armMat: THREE.MeshStandardMaterial;
  private armOutlineMat: THREE.MeshBasicMaterial;
  private arms: { side: -1 | 1; pivot: THREE.Group }[] = [];
  private bow = new THREE.Group();
  private bowMat = new THREE.MeshStandardMaterial({ roughness: 0.25, transparent: true, opacity: 0.92 });
  private bowOutlineMat = new THREE.MeshBasicMaterial({ side: THREE.BackSide, transparent: true });
  private bowLoops: { side: -1 | 1; mesh: THREE.Mesh }[] = [];
  private bowTails: { side: -1 | 1; mesh: THREE.Mesh }[] = [];
  private outlines: THREE.Object3D[] = [];
  private face: ReturnType<typeof paintedTexture>;
  private overlay = paintedTexture(256);
  private overlayMesh: THREE.Mesh;

  private hemi = new THREE.HemisphereLight();
  private key = new THREE.DirectionalLight(0xfff0dd, 1.5);
  private rimL = new THREE.PointLight(0xffffff, 0, 0, 2);
  private rimR = new THREE.PointLight(0xffffff, 0, 0, 2);
  private mood = new THREE.PointLight(0xffffff, 0, 0, 2);
  private moodColor = new THREE.Color();

  private wisps: THREE.Points;
  private wispDepth: number[] = [];
  private mists: { mesh: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>; x: number; phase: number }[] = [];

  // Ectoplasm shed from the hem and bursts for big moments; spirit orbs circling him.
  private sparks = new Particles(900);
  private orbs = new Particles(5);
  private orbPhase = 0;
  private orbSpeed = 0.45;
  private trailDue = 0;
  private lastT = 0;
  private lastAction: string | null = null;
  private lastEmotion = "neutral";
  private lastRude: { since: number } | null = null;
  private mouthVertex = 0; // the body vertex nearest his mouth, where raspberry spit comes from
  private pulse = 0; // a ripple running down the sheet after a burst
  private rim = { value: 0 };
  private rimColor = { value: new THREE.Color() };
  private tmp = new THREE.Vector3();
  private tmpDir = new THREE.Vector3();
  private tmpColor = new THREE.Color();

  private palette: Palette | null = null;
  private glowColor = new THREE.Color();
  private w = 1;
  private h = 1;

  constructor(private ghost: Ghost, quality: "high" | "low" = "high") {
    this.canvas = document.createElement("canvas");
    this.canvas.className = "stage3d";
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(quality === "low" ? 1 : Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.toneMapping = THREE.NeutralToneMapping;
    this.renderer.setClearColor(0x000000, 1);
    this.camera.position.set(0, 0, VIEW_H / 2 / Math.tan(THREE.MathUtils.degToRad(FOV / 2)));
    this.faceCamera.position.copy(this.camera.position);
    this.faceCamera.layers.set(FACE_LAYER);
    if (quality === "low") this.camera.layers.enable(FACE_LAYER); // no bloom, so one pass draws everything

    this.scene.add(this.root);
    this.root.add(this.mover);
    this.mover.add(this.spinner);
    this.root.add(this.hemi, this.mood, this.rimL, this.rimR);
    this.scene.add(this.key);
    this.key.position.set(2, 3, 6); // opposite the painted highlight, so he gets some modelling on both sides
    this.rimL.position.set(-2.6, 1.4, -2.2);
    this.rimR.position.set(2.6, 0.4, -2.2);
    this.mood.position.set(0, -1.9, 2.8);

    // ---- body: one deforming mesh, with the face, outline and halo sharing its vertices
    this.rest = this.buildBody();
    let best = Infinity;
    for (let v = 0; v < ROWS * COLS; v++) {
      const d = Math.hypot(this.rest[v * 3], this.rest[v * 3 + 1] - 45) - this.rest[v * 3 + 2] * 0.01;
      if (this.rest[v * 3 + 2] > 0 && d < best) [best, this.mouthVertex] = [d, v];
    }
    this.bodyMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, side: THREE.DoubleSide });
    // Glowing edges, like light through a thin sheet; the bloom picks them up.
    this.bodyMat.onBeforeCompile = (shader) => {
      shader.uniforms.uRim = this.rim;
      shader.uniforms.uRimColor = this.rimColor;
      shader.fragmentShader = shader.fragmentShader
        .replace("#include <common>", "#include <common>\nuniform float uRim;\nuniform vec3 uRimColor;")
        .replace(
          "#include <emissivemap_fragment>",
          `#include <emissivemap_fragment>
          float rimF = 1.0 - abs(dot(normal, normalize(vViewPosition)));
          totalEmissiveRadiance += uRimColor * pow(rimF, 3.0) * uRim;`,
        );
    };
    const body = new THREE.Mesh(this.bodyGeo, this.bodyMat);

    this.face = paintedTexture(quality === "low" ? 256 : 512);
    // Unlit, so the ink stays crisp and dark however he's lit.
    this.faceMat = new THREE.MeshBasicMaterial({
      map: this.face.tex, transparent: true, depthWrite: false, toneMapped: false,
      polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4,
    });
    const face = new THREE.Mesh(this.buildFaceGeo(), this.faceMat);
    face.layers.set(FACE_LAYER);

    this.outlineMat = new THREE.ShaderMaterial({
      uniforms: { width: { value: 0.035 }, color: { value: new THREE.Color() }, opacity: { value: 1 } },
      vertexShader: OUTLINE_VERT, fragmentShader: OUTLINE_FRAG, side: THREE.BackSide, transparent: true,
    });
    const outline = new THREE.Mesh(this.bodyGeo, this.outlineMat);
    this.haloMat = new THREE.ShaderMaterial({
      uniforms: { width: { value: 0.3 }, color: { value: new THREE.Color() }, strength: { value: 1 } },
      vertexShader: HALO_VERT, fragmentShader: HALO_FRAG, side: THREE.BackSide,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    const halo = new THREE.Mesh(this.bodyGeo, this.haloMat);
    halo.renderOrder = -1;
    this.spinner.add(halo, outline, body, face);
    this.outlines.push(outline);

    // ---- arms: squashed spheres hanging from the shoulders, a little behind the body
    this.armMat = new THREE.MeshStandardMaterial({ roughness: 0.55 });
    this.armOutlineMat = new THREE.MeshBasicMaterial({ side: THREE.BackSide, transparent: true });
    const armGeo = new THREE.SphereGeometry(1, 24, 16);
    for (const side of [-1, 1] as const) {
      const pivot = new THREE.Group();
      pivot.position.set(side * 0.84, -0.4, -0.08);
      const arm = new THREE.Mesh(armGeo, this.armMat);
      arm.position.set(side * 0.18, -0.2, 0);
      arm.rotation.z = side * 0.15;
      arm.scale.set(0.15, 0.3, 0.13);
      const armOutline = new THREE.Mesh(armGeo, this.armOutlineMat);
      armOutline.scale.set(1.25, 1.12, 1.25);
      arm.add(armOutline);
      pivot.add(arm);
      this.spinner.add(pivot);
      this.arms.push({ side, pivot });
      this.outlines.push(armOutline);
    }

    this.buildBow();

    // ---- thinking dots and zzz: painted by Ghost, on a card that bobs with him but doesn't spin
    const os = OVERLAY_BOX.size / 100;
    this.overlayMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(os, os),
      new THREE.MeshBasicMaterial({ map: this.overlay.tex, transparent: true, depthWrite: false, toneMapped: false }),
    );
    this.overlayMesh.position.set((OVERLAY_BOX.x0 + OVERLAY_BOX.size / 2) / 100, -(OVERLAY_BOX.y0 + OVERLAY_BOX.size / 2) / 100, 0.6);
    this.mover.add(this.overlayMesh);

    // ---- atmosphere: rising wisps and a low drifting mist
    const n = ghost.wispList.length;
    const wispGeo = new THREE.BufferGeometry();
    wispGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    wispGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    for (let i = 0; i < n; i++) this.wispDepth.push(-2.5 + ((i * 0.618) % 1) * 4);
    this.wisps = new THREE.Points(
      wispGeo,
      new THREE.PointsMaterial({
        size: 0.11, map: softDot(64), vertexColors: true, transparent: true,
        depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false,
      }),
    );
    this.wisps.frustumCulled = false;
    this.root.add(this.wisps, this.sparks.points, this.orbs.points);

    const mistTex = softDot(256, 3);
    for (let i = 0; i < 5; i++) {
      const mesh = new THREE.Mesh(
        new THREE.PlaneGeometry(4.5, 1.2),
        new THREE.MeshBasicMaterial({ map: mistTex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0.1 }),
      );
      mesh.position.set(0, -2.2 + (i % 2) * 0.2, -1.8 + i * 0.8);
      this.root.add(mesh);
      this.mists.push({ mesh, x: -2 + i, phase: i * 1.9 });
    }

    if (quality === "high") {
      this.composer = new EffectComposer(this.renderer);
      this.composer.addPass(new RenderPass(this.scene, this.camera));
      this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.5, 0.3, 0.9);
      this.composer.addPass(this.bloom);
      const facePass = new RenderPass(this.scene, this.faceCamera);
      facePass.clear = false; // on top of the bloomed frame, depth-tested against the body
      this.composer.addPass(facePass);
      this.composer.addPass(new OutputPass());
    }
  }

  resize(w: number, h: number) {
    this.w = w;
    this.h = h;
    this.renderer.setSize(w, h, false);
    this.composer?.setPixelRatio(this.renderer.getPixelRatio());
    this.composer?.setSize(w, h);
    for (const cam of [this.camera, this.faceCamera]) {
      cam.aspect = w / h;
      cam.updateProjectionMatrix();
    }
    for (const p of [this.sparks, this.orbs]) p.setViewport(h * this.renderer.getPixelRatio(), FOV);
  }

  setVisible(on: boolean) {
    this.canvas.style.display = on ? "" : "none";
  }

  render() {
    const g = this.ghost;
    const t = g.time;
    const s = g.settings;
    const l = g.layout();
    const dt = clamp(t - this.lastT, 0, 0.1);
    this.lastT = t;
    if (g.palette !== this.palette) this.applyPalette(g.palette);

    // Hanging back moves him away from the camera; x and y grow to match, so he shrinks about his spot.
    const camZ = this.camera.position.z;
    const back = camZ * (1 / l.depth - 1);
    this.root.scale.setScalar(s.scale);
    this.root.position.set((s.offsetX * (this.w / this.h) * VIEW_H) / l.depth, (-s.offsetY * VIEW_H) / l.depth, -back);
    this.mover.position.set(l.x / 100, -l.y / 100, 0);
    this.mover.rotation.z = -l.rot;
    this.mover.scale.set(l.sx, l.sy, l.sx);
    // A slow idle turn and a glance towards where he's looking; leaning in when listening.
    const drift = (Math.sin(t * 0.31) * 0.22 + Math.sin(t * 0.13 + 2) * 0.12) * l.calm;
    this.spinner.rotation.y = l.spin + drift + g.gazeX * 0.15;
    this.spinner.rotation.x = g.leanAmount * 0.12;

    this.pulse *= Math.exp(-2.5 * dt);
    this.deformBody(t, g.dragAmount, g.state === "speaking" ? g.speechLevel : 0);
    for (const { side, pivot } of this.arms) pivot.rotation.z = -g.armAngle(side);
    this.bow.visible = s.bow;
    for (const { side, mesh } of this.bowLoops) mesh.scale.setScalar(1 + Math.sin(t * 2.2 + side * 0.8) * 0.06);
    for (const { side, mesh } of this.bowTails) mesh.rotation.z = side * 0.35 + Math.sin(t * 2.6 + side * 1.3) * 0.12;
    this.paintFace();
    this.paintOverlay();

    // Hide/peek fade the whole body; sleeping dims the light instead (a see-through mesh looks wrong).
    const fade = l.fade;
    const dim = (g.state === "sleeping" || g.state === "error" ? 0.55 : 1) * (1 - (1 - l.depth) * 0.8); // and further back, fainter
    const fading = fade < 0.999;
    this.spinner.visible = fade > 0.01;
    for (const m of [this.bodyMat, this.armMat]) {
      m.transparent = fading;
      m.opacity = fade;
      m.emissiveIntensity = (0.1 + 0.16 * l.glow) * dim;
    }
    this.faceMat.opacity = fade;
    this.bowMat.opacity = 0.92 * fade;
    this.bowOutlineMat.opacity = 0.85 * fade;
    this.bowMat.emissiveIntensity = (0.22 + 0.12 * l.glow + 0.1 * Math.sin(t * 0.9)) * dim; // the shimmer
    this.outlineMat.uniforms.opacity.value = fade;
    this.armOutlineMat.opacity = fade;
    this.haloMat.uniforms.strength.value = clamp(0.5 * l.glow, 0, 1.6) * fade * dim;

    this.hemi.intensity = 0.8 * dim;
    this.key.intensity = 1.5 * dim;
    this.rimL.intensity = this.rimR.intensity = (14 + 10 * l.glow) * dim;
    const emo = g.currentEmotion;
    const target = this.glowColor.clone().lerp(new THREE.Color(MOOD[emo] ?? this.glowColor), g.currentIntensity);
    this.moodColor.lerp(target, 1 - Math.exp(-3 / 60));
    this.mood.color.copy(this.moodColor);
    this.mood.intensity = (6 + 14 * g.currentIntensity + 8 * g.speechLevel) * dim;

    this.updateWisps(t, fade);
    this.rim.value = (0.35 + 0.25 * l.glow) * dim * fade;
    this.scene.updateMatrixWorld();
    this.emit(dt, l.fade, dim);
    this.moveOrbs(t, dt, fade * dim);
    for (const m of this.mists) {
      m.mesh.position.x = m.x + Math.sin(t * 0.07 + m.phase) * 1.6;
      m.mesh.material.opacity = (0.035 + 0.02 * Math.sin(t * 0.3 + m.phase)) * (0.6 + 0.4 * s.glow);
    }

    if (this.composer && this.bloom) {
      this.bloom.strength = clamp(0.2 + 0.2 * l.glow, 0, 0.9);
      this.composer.render();
    } else {
      this.renderer.render(this.scene, this.camera);
    }
  }

  // ------------------------------------------------------------------ particles

  /** A body vertex's current position, in root space (where the particles live, so trails stay behind). */
  private vertexInRoot(v: number, out: THREE.Vector3) {
    const pos = this.bodyGeo.attributes.position.array as Float32Array;
    out.set(pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2]);
    return this.root.worldToLocal(this.spinner.localToWorld(out));
  }

  /** A direction given in the spinner's frame, turned into root space, times `speed`. */
  private dirInRoot(x: number, y: number, z: number, speed: number) {
    const d = this.tmpDir.set(x, y, z).transformDirection(this.spinner.matrixWorld);
    return d.multiplyScalar(speed);
  }

  private sparkColor(whiteness: number) {
    const p = this.palette!;
    return this.tmpColor.setHSL(p.wispHues[Math.random() < 0.6 ? 0 : 1] / 360, 0.9, 0.7).lerp(this.glowColor, 0.3).lerp(WHITE, whiteness);
  }

  private emit(dt: number, fade: number, dim: number) {
    const g = this.ghost;
    // Ectoplasm drips off the hem and drifts back up: more when he moves fast or talks.
    const moving = Math.min(1, Math.abs(g.dragAmount) / 12);
    const speech = g.state === "speaking" ? g.speechLevel : 0;
    this.trailDue += dt * (12 + 45 * moving + 30 * speech) * fade * (g.state === "sleeping" ? 0.3 : 1);
    const hem = (ROWS - 1) * COLS;
    while (this.trailDue >= 1) {
      this.trailDue -= 1;
      const j = Math.floor(Math.random() * COLS);
      const p = this.vertexInRoot(hem + j, this.tmp);
      const u = (j / COLS) * Math.PI * 2;
      const v = this.dirInRoot(Math.sin(u), -1.5, Math.cos(u), 0.2 + Math.random() * 0.15);
      this.sparks.spawn({
        x: p.x, y: p.y, z: p.z, vx: v.x + (Math.random() - 0.5) * 0.1, vy: v.y, vz: v.z,
        life: 1.4 + Math.random(), size: 0.07 + Math.random() * 0.08, color: this.sparkColor(0.15),
        drag: 1.5, lift: 0.35, alpha: 0.75 * dim,
      });
    }

    // Bursts when a big move starts, or he's suddenly surprised or delighted.
    const action = g.currentAction;
    if (action && action !== this.lastAction) this.burst(action);
    this.lastAction = action;
    const emo = g.currentEmotion;
    if (emo !== this.lastEmotion && (emo === "surprised" || emo === "happy") && g.currentIntensity > 0.5) this.burst(emo);
    this.lastEmotion = emo;
    this.rudeEffects();

    this.sparks.update(dt);
  }

  /** A green puff from the hem for a fart (and a ripple through the sheet); spit flying for a raspberry. */
  private rudeEffects() {
    const r = this.ghost.currentRude;
    const fresh = r !== null && (this.lastRude === null || r.since < this.lastRude.since);
    this.lastRude = r;
    if (!r) return;
    if (r.kind === "fart" && fresh) {
      this.pulse = 1;
      const hem = (ROWS - 1) * COLS;
      for (let i = 0; i < 70; i++) {
        const j = Math.floor(Math.random() * COLS);
        const p = this.vertexInRoot(hem - COLS * Math.floor(Math.random() * 3) + j, this.tmp);
        const u = (j / COLS) * Math.PI * 2;
        const d = this.dirInRoot(Math.sin(u), -0.6 + Math.random() * 0.5, Math.cos(u) - 0.6, 0.35 + Math.random() * 0.5);
        this.sparks.spawn({ x: p.x, y: p.y, z: p.z, vx: d.x, vy: d.y, vz: d.z, life: 1.4 + Math.random() * 1.2,
          size: 0.22 + Math.random() * 0.3, color: this.tmpColor.setHSL(0.2 + Math.random() * 0.08, 0.75, 0.55),
          drag: 1.6, lift: 0.25, alpha: 0.4 });
      }
    } else if (r.kind === "raspberry" && r.since < r.dur) {
      const p = this.vertexInRoot(this.mouthVertex, this.tmp);
      for (let i = 0; i < 3; i++) {
        const d = this.dirInRoot((Math.random() - 0.5) * 1.2, -0.1 + Math.random() * 0.5, 1, 1 + Math.random() * 1.2);
        this.sparks.spawn({ x: p.x, y: p.y, z: p.z, vx: d.x, vy: d.y, vz: d.z, life: 0.4 + Math.random() * 0.4,
          size: 0.03 + Math.random() * 0.03, color: this.tmpColor.setRGB(0.85, 0.95, 1), drag: 1, lift: -3, alpha: 0.9 });
      }
    }
  }

  private burst(kind: "spin" | "hide" | "peek" | "surprised" | "happy") {
    const n = ROWS * COLS;
    const fromBody = (count: number, fn: (u: number, v: number) => void) => {
      for (let i = 0; i < count; i++) {
        const v = Math.floor(Math.random() * n);
        fn(((v % COLS) / COLS) * Math.PI * 2, v);
      }
    };
    if (kind === "spin") {
      // Sparks flung off sideways as he twirls.
      fromBody(80, (u, v) => {
        const p = this.vertexInRoot(v, this.tmp);
        const d = this.dirInRoot(Math.sin(u) + Math.cos(u) * 1.5, 0.4 + Math.random() * 0.5, Math.cos(u) - Math.sin(u) * 1.5, 1.2 + Math.random());
        this.sparks.spawn({ x: p.x, y: p.y, z: p.z, vx: d.x, vy: d.y, vz: d.z, life: 0.8 + Math.random() * 0.6,
          size: 0.05 + Math.random() * 0.06, color: this.sparkColor(0.5), drag: 2.5, lift: -0.3 });
      });
      this.pulse = 1;
    } else if (kind === "hide" || kind === "peek") {
      // A soft poof of mist where he was.
      fromBody(kind === "hide" ? 90 : 45, (u, v) => {
        const p = this.vertexInRoot(v, this.tmp);
        const d = this.dirInRoot(Math.sin(u), Math.random() * 0.6, Math.cos(u), 0.3 + Math.random() * 0.4);
        this.sparks.spawn({ x: p.x, y: p.y, z: p.z, vx: d.x, vy: d.y, vz: d.z, life: 1 + Math.random() * 0.7,
          size: 0.18 + Math.random() * 0.2, color: this.sparkColor(0.6), drag: 1.8, lift: 0.2, alpha: 0.3 });
      });
    } else {
      // A ring of sparkles round his head (surprised), or a shower rising from his shoulders (happy).
      const head = this.tmp.set(0, 0.3, 0.2);
      this.root.worldToLocal(this.spinner.localToWorld(head));
      const count = kind === "surprised" ? 36 : 24;
      for (let i = 0; i < count; i++) {
        const a = (i / count) * Math.PI * 2;
        const v =
          kind === "surprised"
            ? { x: Math.cos(a) * 1.8, y: Math.sin(a) * 1.8, life: 0.7, lift: 0 }
            : { x: (Math.random() - 0.5) * 1.2, y: 0.6 + Math.random() * 0.8, life: 1.2, lift: -0.4 };
        this.sparks.spawn({ x: head.x, y: head.y, z: head.z, vx: v.x, vy: v.y, vz: (Math.random() - 0.5) * 0.3,
          life: v.life + Math.random() * 0.4, size: 0.05 + Math.random() * 0.05, color: this.sparkColor(0.7), drag: 2.2, lift: v.lift });
      }
      if (kind === "surprised") this.pulse = Math.max(this.pulse, 0.6);
    }
  }

  /** Spirit orbs circling him: lazy when idle, whirling while he thinks, drawn in while he listens. */
  private moveOrbs(t: number, dt: number, fade: number) {
    const g = this.ghost;
    const want = { thinking: 2.6, listening: 0.3, sleeping: 0.12, speaking: 0.7, error: 0.2, idle: 0.45 }[g.state];
    this.orbSpeed += (want - this.orbSpeed) * (1 - Math.exp(-2 * dt));
    this.orbPhase += this.orbSpeed * dt;
    const radius = g.state === "thinking" ? 1.35 : g.state === "listening" ? 1.45 : 1.75;
    const m = this.mover.position;
    const glow = g.state === "speaking" ? 0.6 + g.speechLevel : g.state === "sleeping" ? 0.35 : 0.75;
    for (let i = 0; i < this.orbs.capacity; i++) {
      const a = this.orbPhase + (i / this.orbs.capacity) * Math.PI * 2;
      const r = radius + 0.15 * Math.sin(t * 0.7 + i * 2);
      const c = this.tmpColor.setHSL(this.palette!.wispHues[i % 2] / 360, 0.85, 0.7);
      const twinkle = 0.75 + 0.25 * Math.sin(t * 3.1 + i * 1.7);
      // A tilted orbit: low past the front (by his tail, clear of the face), high round the back.
      this.orbs.place(i, m.x + Math.cos(a) * r, m.y - 0.3 - 0.6 * Math.sin(a) + 0.12 * Math.sin(a * 2 + i), Math.sin(a) * r * 0.7,
        0.15, glow * twinkle * fade, c);
    }
    this.orbs.commit();
  }

  // ------------------------------------------------------------------ body geometry

  /** Vertices in rows from the crown down, COLS around; u = 0 faces the camera. */
  /** A ribbon bow on the top right of his head: the 2D drawBow's loop and tail outlines, extruded, and a knot. */
  private buildBow() {
    const part = (geo: THREE.BufferGeometry, outline: number) => {
      const mesh = new THREE.Mesh(geo, this.bowMat);
      const edge = new THREE.Mesh(geo, this.bowOutlineMat);
      edge.scale.set(outline, outline, 1); // wider, and set back, so it rims the ribbon instead of tinting it
      edge.position.z = -0.025;
      mesh.add(edge);
      this.outlines.push(edge);
      this.bow.add(mesh);
      return mesh;
    };
    // Outlines in 2D ghost units (y down), turned into 3D units (y up) and centred in z.
    const ribbon = (draw: (s: THREE.Shape) => void) => {
      const shape = new THREE.Shape();
      draw(shape);
      const geo = new THREE.ExtrudeGeometry(shape, {
        depth: 4, bevelEnabled: true, bevelThickness: 2, bevelSize: 1.5, bevelSegments: 3, curveSegments: 16,
      });
      geo.scale(0.01, 0.01, 0.01);
      geo.rotateX(Math.PI); // y down to y up; a negative scale would turn the faces inside out
      geo.translate(0, 0, 0.02);
      return geo;
    };
    for (const side of [-1, 1] as const) {
      const tail = part(ribbon((sh) => {
        sh.moveTo(side * 2, 4);
        sh.quadraticCurveTo(side * 6, 16, side * 16, 30);
        sh.lineTo(side * 11, 26);
        sh.lineTo(side * 5, 32);
        sh.quadraticCurveTo(side * 1, 17, side * -4, 5);
      }), 1.08);
      tail.position.z = -0.03;
      this.bowTails.push({ side, mesh: tail });
      const loop = part(ribbon((sh) => {
        sh.moveTo(0, 0);
        sh.bezierCurveTo(side * 12, -28, side * 46, -32, side * 44, -6);
        sh.bezierCurveTo(side * 42, 14, side * 16, 12, 0, 0);
      }), 1.07);
      this.bowLoops.push({ side, mesh: loop });
    }
    part(new THREE.SphereGeometry(0.09, 16, 12), 1.25).scale.set(1, 0.9, 0.7);
    // Perched on the head (a unit sphere), standing a little proud of it and facing the front, tilted with the curve.
    const at = new THREE.Vector3(0.5, 0.82, 0.42).normalize().multiplyScalar(1.04);
    this.bow.position.copy(at);
    this.bow.lookAt(at.clone().add(new THREE.Vector3(0.15, 0.2, 1)));
    this.bow.rotateZ(-0.55);
    this.bow.scale.setScalar(1.3);
    this.spinner.add(this.bow);
  }

  private buildBody(): Float32Array {
    const n = ROWS * COLS;
    this.bodyGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    this.bodyGeo.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    this.bodyGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    const index: number[] = [];
    for (let i = 0; i < ROWS - 1; i++) {
      for (let j = 0; j < COLS; j++) {
        const a = i * COLS + j;
        const a1 = i * COLS + ((j + 1) % COLS);
        const b = a + COLS;
        const b1 = a1 + COLS;
        if (i > 0) index.push(a, b, a1);
        index.push(b, b1, a1);
      }
    }
    this.bodyGeo.setIndex(index);
    this.deformBody(0, 0);
    // Rest pose in ghost units (y down) for the face's texture coordinates and the colour gradient.
    const pos = this.bodyGeo.attributes.position.array as Float32Array;
    const rest = new Float32Array(pos.length);
    for (let k = 0; k < pos.length; k += 3) {
      rest[k] = pos[k] * 100;
      rest[k + 1] = -pos[k + 1] * 100;
      rest[k + 2] = pos[k + 2] * 100;
    }
    return rest;
  }

  /** The hem ripples in eight lobes, the tail flutters and lags behind sideways moves (like bodyPath). */
  private deformBody(t: number, drag: number, speech = 0) {
    const pos = this.bodyGeo.attributes.position.array as Float32Array;
    for (let i = 0; i < ROWS; i++) {
      for (let j = 0; j < COLS; j++) {
        const u = (j / COLS) * Math.PI * 2;
        let r: number;
        let y: number;
        let lag = 0;
        if (i <= DOME) {
          const th = (i / DOME) * (Math.PI / 2);
          r = Math.sin(th);
          y = Math.cos(th);
        } else {
          const s = (i - DOME) / SKIRT;
          const lobe = Math.abs(Math.sin(4 * u + 0.4)) ** 0.5; // rounded lobes, sharp notches
          const wave = Math.sin(t * 3.1 + u * 4.4) * 0.07;
          r = 1 + 0.07 * Math.sin(s * Math.PI * 0.85) + s ** 3 * 0.05 * Math.sin(t * 2.4 + u * 3);
          // Ripples in the sheet, growing towards the hem so the face stays calm: soft vertical folds that
          // sway, bands rolling downwards, a quick shiver while he talks, and a ripple after a burst.
          const deep = s ** 1.5;
          r += 0.022 * deep * Math.cos(5 * u + 0.8 * Math.sin(t * 0.6 + u));
          r += 0.03 * deep * Math.sin(t * 2.3 - s * 7.5 + Math.sin(u * 2 + t * 0.5) * 1.4);
          r += 0.03 * speech * s * Math.sin(t * 11 - s * 12);
          r += 0.07 * this.pulse * s * Math.sin(t * 14 - s * 10);
          y = -1.45 * s - s ** 4 * (0.38 * lobe + wave);
          lag = (-drag / 100) * s * s;
        }
        const k = (i * COLS + j) * 3;
        pos[k] = r * Math.sin(u) + lag;
        pos[k + 1] = y;
        pos[k + 2] = r * Math.cos(u) * DEPTH;
      }
    }
    this.bodyGeo.attributes.position.needsUpdate = true;
    this.bodyGeo.computeVertexNormals();
    const nrm = this.bodyGeo.attributes.normal.array as Float32Array;
    for (let j = 0; j < COLS; j++) nrm.set([0, 1, 0], j * 3); // the crown's vertices all coincide
    this.bodyGeo.computeBoundingSphere();
  }

  /** Front triangles only, with UVs from a straight-on projection, so the 2D face lands where it does in 2D. */
  private buildFaceGeo(): THREE.BufferGeometry {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", this.bodyGeo.attributes.position);
    geo.setAttribute("normal", this.bodyGeo.attributes.normal);
    const n = ROWS * COLS;
    const uv = new Float32Array(n * 2);
    for (let v = 0; v < n; v++) {
      uv[v * 2] = (this.rest[v * 3] - FACE_BOX.x0) / FACE_BOX.size;
      uv[v * 2 + 1] = 1 - (this.rest[v * 3 + 1] - FACE_BOX.y0) / FACE_BOX.size;
    }
    geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    const front = (v: number) => {
      const [x, y, z] = [this.rest[v * 3], this.rest[v * 3 + 1], this.rest[v * 3 + 2]];
      return z > 15 && x > FACE_BOX.x0 && x < FACE_BOX.x0 + FACE_BOX.size && y > FACE_BOX.y0 && y < FACE_BOX.y0 + FACE_BOX.size;
    };
    const body = this.bodyGeo.index!.array;
    const index: number[] = [];
    for (let k = 0; k < body.length; k += 3) {
      if (front(body[k]) && front(body[k + 1]) && front(body[k + 2])) index.push(body[k], body[k + 1], body[k + 2]);
    }
    geo.setIndex(index);
    return geo;
  }

  private applyPalette(p: Palette) {
    this.palette = p;
    this.glowColor.copy(rgb(p.glow));
    this.moodColor.copy(this.glowColor);
    // The 2D radial gradient (core → mid → edge from a highlight up and left), then shade towards the tail.
    const core = new THREE.Color(p.core);
    const mid = new THREE.Color(p.mid);
    const edge = new THREE.Color(p.edge);
    const shade = rgb(p.shade);
    const col = this.bodyGeo.attributes.color.array as Float32Array;
    const c = new THREE.Color();
    for (let v = 0; v < ROWS * COLS; v++) {
      const [x, y, z] = [this.rest[v * 3], this.rest[v * 3 + 1], this.rest[v * 3 + 2]];
      const g = clamp((Math.hypot(x + 28, y + 40) - 10) / 220 + Math.max(0, -z / 100) * 0.25, 0, 1);
      if (g < 0.5) c.copy(core).lerp(mid, g / 0.5);
      else c.copy(mid).lerp(edge, (g - 0.5) / 0.5);
      c.lerp(shade, p.shadeAlpha * clamp((y - 30) / 160, 0, 1));
      col.set([c.r, c.g, c.b], v * 3);
    }
    this.bodyGeo.attributes.color.needsUpdate = true;
    this.bodyMat.emissive.copy(this.glowColor);
    this.armMat.color.copy(mid);
    this.armMat.emissive.copy(this.glowColor);
    const rim = new THREE.Color(p.rim);
    this.outlineMat.uniforms.color.value.copy(rim);
    this.armOutlineMat.color.copy(rim);
    const blush = rgb(p.blush);
    this.bowMat.color.copy(blush).lerp(WHITE, 0.6);
    this.bowMat.emissive.copy(blush).lerp(WHITE, 0.35);
    this.bowOutlineMat.color.copy(blush).multiplyScalar(0.5);
    for (const o of this.outlines) o.visible = p.rimWidth > 0;
    this.haloMat.uniforms.color.value.copy(this.glowColor);
    this.rimColor.value.copy(this.glowColor).lerp(WHITE, 0.3);
    this.hemi.color.copy(this.glowColor).lerp(new THREE.Color(0xffffff), 0.6);
    this.hemi.groundColor.copy(shade);
    this.rimL.color.copy(this.glowColor);
    this.rimR.color.setHSL(p.wispHues[1] / 360, 0.9, 0.65);
    for (const m of this.mists) m.mesh.material.color.copy(this.glowColor);
  }

  // ------------------------------------------------------------------ painted parts

  private paintFace() {
    const { canvas, ctx, tex } = this.face;
    const k = canvas.width / FACE_BOX.size;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(k, 0, 0, k, -FACE_BOX.x0 * k, -FACE_BOX.y0 * k);
    this.ghost.drawFace(ctx);
    tex.needsUpdate = true;
  }

  private paintOverlay() {
    const g = this.ghost;
    const on = g.state === "thinking" || g.state === "sleeping";
    this.overlayMesh.visible = on;
    if (!on) return;
    const { canvas, ctx, tex } = this.overlay;
    const k = canvas.width / OVERLAY_BOX.size;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(k, 0, 0, k, -OVERLAY_BOX.x0 * k, -OVERLAY_BOX.y0 * k);
    if (g.state === "thinking") g.drawThinkingDots(ctx);
    else g.drawZzz(ctx);
    tex.needsUpdate = true;
  }

  private updateWisps(t: number, fade: number) {
    const pos = this.wisps.geometry.attributes.position.array as Float32Array;
    const col = this.wisps.geometry.attributes.color.array as Float32Array;
    const c = new THREE.Color();
    this.ghost.wispList.forEach((w, i) => {
      // Same flicker and edge fade-out as the 2D wisps.
      const a = 0.12 + 0.18 * (0.5 + 0.5 * Math.sin(t * 1.7 + w.phase));
      const edge = clamp((300 - w.y) / 80, 0, 1) * clamp((w.y + 320) / 120, 0, 1);
      pos.set([w.x / 100, -w.y / 100, this.wispDepth[i]], i * 3);
      c.setHSL(this.palette!.wispHues[w.hue] / 360, 0.9, 0.75).multiplyScalar(a * edge * (0.6 + w.size * 0.6) * (0.5 + 0.5 * fade));
      col.set([c.r, c.g, c.b], i * 3);
    });
    this.wisps.geometry.attributes.position.needsUpdate = true;
    this.wisps.geometry.attributes.color.needsUpdate = true;
  }
}
