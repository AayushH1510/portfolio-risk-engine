/**
 * smoke-field.ts — the hero's ambient espresso plume.
 *
 * Framework-agnostic. No React, no DOM assumptions beyond a <canvas>.
 * Every visual constant is an option; the defaults live in tokens.json > smoke
 * so the palette can be re-themed without touching this file.
 *
 * Cost: gridWidth * gridHeight * octaves * 2 fbm calls per frame, capped at
 * `fps`. At the defaults that is ~86k hash evaluations per frame at 24fps,
 * which is comfortably under a millisecond on a mid-range laptop. The result
 * is drawn to a tiny offscreen buffer and upscaled with a blur, so the
 * destination canvas resolution barely matters.
 */

export interface SmokeFieldOptions {
  /** [stop 0..1, hex] pairs, ascending. Maps field intensity to colour. */
  ramp: Array<[number, string]>;
  /** Simulation grid. Larger = more detail, quadratically more work. */
  gridWidth?: number;
  gridHeight?: number;
  /** Backing store width of the visible canvas; height follows its aspect ratio. */
  outputWidth?: number;
  /** Frame cap. The plume is slow — 24 is plenty and halves CPU vs 60. */
  fps?: number;
  /** Noise octaves. 3 is the sweet spot; 4 costs 33% more for little gain. */
  octaves?: number;
  /** Simulation clock increment per frame. */
  timeStep?: number;
  /** How fast the field scrolls upward (the "rising" of the smoke). */
  riseSpeed?: number;
  /** Domain-warp strength. 0 = flat noise, 4+ = turbulent. */
  swirl?: number;
  /** Intensity below this is clamped to the darkest ramp stop. MUST sit below
   *  the mean of the normalised field (~0.5) or the plume goes black. */
  floor?: number;
  /** Multiplier applied after `floor` is subtracted. */
  contrast?: number;
  /** Horizontal falloff. Higher = narrower plume. */
  spread?: number;
  /** Upscale blur in px, applied at the offscreen resolution. */
  blur?: number;
  /** Pause when the tab is hidden. Default true. */
  pauseWhenHidden?: boolean;
  /** Skip animation and paint one static frame. Wire this to
   *  prefers-reduced-motion. Default false. */
  reducedMotion?: boolean;
}

type RGB = [number, number, number];

const hexToRgb = (hex: string): RGB => {
  const h = hex.replace('#', '');
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
};

/**
 * 32-bit integer hash → float in [0, 1).
 *
 * NOTE the logical `>>>` shifts. An arithmetic `>>` propagates the sign bit,
 * so `n ^ (n >> 16)` always leaves bit 31 equal to bit 31 of n — which pins
 * the output ceiling at 0.5 and makes the whole field render black. This was
 * a real bug; do not "simplify" these back to `>>`.
 */
const hash = (x: number, y: number): number => {
  let n = (Math.imul(x, 374761393) + Math.imul(y, 668265263)) | 0;
  n = (n ^ (n >>> 13)) | 0;
  n = Math.imul(n, 1274126177) | 0;
  return ((n ^ (n >>> 16)) >>> 0) / 4294967295;
};

const smoothNoise = (x: number, y: number): number => {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const u = xf * xf * (3 - 2 * xf);
  const v = yf * yf * (3 - 2 * yf);
  const a = hash(xi, yi);
  const b = hash(xi + 1, yi);
  const c = hash(xi, yi + 1);
  const d = hash(xi + 1, yi + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
};

export function createSmokeField(
  canvas: HTMLCanvasElement,
  options: SmokeFieldOptions
): { destroy: () => void } {
  const {
    ramp,
    gridWidth: W = 108,
    gridHeight: H = 66,
    outputWidth = 520,
    fps = 24,
    octaves = 3,
    timeStep = 0.0055,
    riseSpeed = 1.6,
    swirl = 2.4,
    floor = 0.28,
    contrast = 2.2,
    spread = 1.45,
    blur = 9,
    pauseWhenHidden = true,
    reducedMotion = false,
  } = options;

  const ctx = canvas.getContext('2d');
  if (!ctx) return { destroy: () => {} };

  const offscreen = document.createElement('canvas');
  offscreen.width = W;
  offscreen.height = H;
  const octx = offscreen.getContext('2d')!;
  const image = octx.createImageData(W, H);
  const data = image.data;

  // fbm sums amplitudes 0.5, 0.25, 0.125… — precompute the ceiling so the
  // field can be normalised to a true 0..1 regardless of octave count.
  let ceiling = 0;
  for (let i = 0, amp = 0.5; i < octaves; i++, amp *= 0.5) ceiling += amp;

  const fbm = (x: number, y: number): number => {
    let sum = 0;
    let amp = 0.5;
    let freq = 1;
    for (let i = 0; i < octaves; i++) {
      sum += amp * smoothNoise(x * freq, y * freq);
      freq *= 2.03; // non-integer avoids octave alignment artefacts
      amp *= 0.5;
    }
    return sum;
  };

  const stops: Array<{ at: number; rgb: RGB }> = ramp.map(([at, hex]) => ({
    at,
    rgb: hexToRgb(hex),
  }));

  const shade = (v: number): RGB => {
    for (let i = 1; i < stops.length; i++) {
      if (v <= stops[i].at || i === stops.length - 1) {
        const a = stops[i - 1];
        const b = stops[i];
        const t = Math.min(1, Math.max(0, (v - a.at) / (b.at - a.at)));
        return [
          a.rgb[0] + (b.rgb[0] - a.rgb[0]) * t,
          a.rgb[1] + (b.rgb[1] - a.rgb[1]) * t,
          a.rgb[2] + (b.rgb[2] - a.rgb[2]) * t,
        ];
      }
    }
    return stops[0].rgb;
  };

  const resize = () => {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width) return;
    canvas.width = outputWidth;
    canvas.height = Math.max(
      180,
      Math.round(outputWidth * (rect.height / rect.width))
    );
  };
  resize();

  let raf = 0;
  let lastFrame = 0;
  let t = 0;
  let visible = true;
  const frameBudget = 1000 / fps;

  const paint = () => {
    let p = 0;
    const rise = t * riseSpeed;
    for (let y = 0; y < H; y++) {
      const ny = y * 0.052;
      const vertical = Math.pow(y / H, 2.1) * 0.5;
      for (let x = 0; x < W; x++) {
        const nx = x * 0.052;
        const q = fbm(nx + t * 0.5, ny - rise);
        const raw = fbm(nx + swirl * q, ny - rise * 0.7 + 1.9 * q) / ceiling;
        const v = (raw - floor) * contrast;
        const cx = x / W - 0.5;
        // plume mask: bright up the centre column, fading to the base
        const fall = 1 - Math.min(1, cx * cx * spread + vertical);
        const [r, g, b] = shade(Math.min(1, Math.max(0, v * (0.44 + 0.56 * fall))));
        data[p++] = r;
        data[p++] = g;
        data[p++] = b;
        data[p++] = 255;
      }
    }
    octx.putImageData(image, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.filter = `blur(${blur}px)`;
    ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
    ctx.filter = 'none';
  };

  const onVisibility = () => {
    visible = !document.hidden;
  };
  const onResize = () => resize();

  if (reducedMotion) {
    paint();
  } else {
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop);
      if (pauseWhenHidden && !visible) return;
      if (now - lastFrame < frameBudget) return;
      lastFrame = now;
      t += timeStep;
      paint();
    };
    raf = requestAnimationFrame(loop);
    document.addEventListener('visibilitychange', onVisibility);
  }

  window.addEventListener('resize', onResize, { passive: true });

  return {
    destroy() {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      document.removeEventListener('visibilitychange', onVisibility);
    },
  };
}
