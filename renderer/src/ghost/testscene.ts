import type { Ghost } from "./ghost";

/** Projector calibration: grid, black-level steps and the ghost's outline at its current placement. */
export function drawTestPattern(ctx: CanvasRenderingContext2D, w: number, h: number, ghost: Ghost) {
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 1;
  for (let i = 1; i < 10; i++) {
    ctx.beginPath();
    ctx.moveTo((w * i) / 10, 0);
    ctx.lineTo((w * i) / 10, h);
    ctx.moveTo(0, (h * i) / 10);
    ctx.lineTo(w, (h * i) / 10);
    ctx.stroke();
  }
  ctx.strokeStyle = "#888";
  ctx.strokeRect(1, 1, w - 2, h - 2);

  // Black-level steps: on a good setup the first patch is invisible and the rest just visible.
  const levels = [0, 4, 8, 16, 32, 64];
  const size = Math.min(w, h) * 0.06;
  levels.forEach((v, i) => {
    ctx.fillStyle = `rgb(${v},${v},${v})`;
    ctx.fillRect(size + i * size * 1.2, h - size * 2, size, size);
  });

  ctx.fillStyle = "#777";
  ctx.font = "14px system-ui, sans-serif";
  ctx.fillText(`Projection test · ${w}×${h} · arrows move, +/- scale, K to exit`, size, size);
  ghost.draw(ctx, w, h, true);
}
