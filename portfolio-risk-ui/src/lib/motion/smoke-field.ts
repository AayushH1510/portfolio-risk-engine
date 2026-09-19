/**
 * smoke-field.ts — the hero's ambient espresso plume.
 *
 * Framework-agnostic. No React, no DOM assumptions beyond a <canvas>.
 * Every visual constant is an option; the defaults live in tokens.json > smoke
 * so the palette can be re-themed without touching this file.
 *
 * Cost: gridWidth * gridHeight * octaves * 2 fbm calls per frame, capped at
 * `fps`. gridHeight is no longer fixed — see resize()'s own comment — it now
 * scales with the container's aspect ratio, so a tall phone hero costs
 * several times the desktop baseline. Measured with performance.mark/measure
 * around paint() on dev hardware (chromium, not a phone — a genuine floor,
 * not a ceiling, this needs to hold on real devices too), 24fps unthrottled:
 *   desktop  1440px, H≈71 (baseline ~66):  ~1.7-1.9ms/paint
 *   phone     384px, H≈341:                ~8.2-9.8ms/paint  (~5x the rows, ~5x the cost)
 *   phone     320px, H≈517:                ~12.0-12.2ms/paint (~7.5x the rows)
 * Unthrottled at 24fps, 384px would spend ~20% of every 41.6ms frame budget
 * on this one decorative element. `effectiveFps()` (below) throttles frame
 * rate down as H grows past baseline so total cost-per-second at phone
 * widths lands within ~1.4-1.7x of desktop's, not up to 2.7x — see its own
 * comment for why a small floor (not full proportional throttling) was kept.
 */

export interface SmokeFieldOptions {
  /** [stop 0..1, hex] pairs, ascending. Maps field intensity to colour. */
  ramp: Array<[number, string]>;
  /** Simulation grid. Larger = more detail, quadratically more work.
   *  gridWidth is the only one actually fixed at this value — gridHeight is
   *  the *baseline* row count, tuned against the ~0.6 aspect ratio a
   *  landscape-oriented hero renders at; resize() rescales the live row
   *  count from this baseline so a taller container gets more rows at the
   *  same visual cell size instead of the same row count stretched
   *  further. See resize()'s own comment. */
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
    gridHeight: baseH = 66,
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

  // baseAspect is the ratio baseH was tuned against — derived, not
  // hardcoded, so a future change to the tokens.json default stays
  // self-consistent. H, the offscreen buffer, and its ImageData are all
  // re-derived in resize() below, not created once here — see that
  // function's comment.
  const baseAspect = baseH / W;
  let H = baseH;
  let offscreen = document.createElement('canvas');
  offscreen.width = W;
  offscreen.height = H;
  let octx = offscreen.getContext('2d')!;
  let image = octx.createImageData(W, H);
  let data = image.data;

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

  // Earlier version of this fix capped the aspect ratio the canvas would
  // ever target and let the container's own overflow:hidden crop what was
  // left — that traded the banding for a hard, visible seam where the
  // (now shorter) canvas ended and flat background began, since the CSS
  // side (Hero.jsx) was stretching a *capped* canvas to less than 100% of
  // the container rather than covering it. The actual fix stays in the
  // opposite direction: keep covering the full container (canvas.height
  // tracks the container's real, uncapped aspect ratio, same as before any
  // of this), and instead rescale the SIMULATION to match — H (row count)
  // is re-derived from that same aspect ratio every resize, at the ratio
  // baseH was tuned against (baseAspect = baseH/W), so canvas.height/H (the
  // vertical stretch the blur has to smooth over) stays equal to
  // canvas.width/W (the horizontal one) at every aspect ratio, not just
  // ~0.6. A taller container gets more rows at the same visual cell size
  // instead of the same 66 rows stretched further — which is also why the
  // blur radius doesn't need its own scaling: the stretch ratio it's
  // smoothing over is now aspect-independent by construction, not just
  // "close enough" at the one ratio it was tuned against.
  const resize = () => {
    const container = canvas.parentElement;
    const rect = (container ?? canvas).getBoundingClientRect();
    if (!rect.width) return;
    const aspect = rect.height / rect.width;
    canvas.width = outputWidth;
    canvas.height = Math.max(180, Math.round(outputWidth * aspect));

    const newH = Math.max(24, Math.round(aspect * W));
    if (newH !== H) {
      H = newH;
      offscreen.height = H;
      image = octx.createImageData(W, H);
      data = image.data;
    }
  };
  resize();

  // Pure proportional throttling (fps * baseH/H, no floor) would fully
  // bound cost-per-second to roughly the desktop baseline at every aspect
  // ratio — but at the most extreme phone aspects that works out to ~3fps,
  // which reads as a slideshow rather than ambient motion for a
  // continuously-drifting cloud. A small floor trades some of that cost
  // bound back for still-visible motion: floored at 6fps, measured
  // cost-per-second at phone widths lands at ~1.4-1.7x the desktop
  // baseline (was up to ~2.7x with no throttling at all — see the
  // module-level comment for the raw per-paint numbers this was tuned
  // against) — not fully flat, a deliberate choppier-but-not-static trade
  // rather than a fully solved one. Never exceeds the configured `fps`
  // when H is at or below baseline (a wide/landscape aspect, where
  // H<baseH, must not speed up).
  const effectiveFps = () => Math.max(6, Math.min(fps, fps * (baseH / H)));

  let raf = 0;
  let lastFrame = 0;
  let t = 0;
  let visible = true;

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
      const targetFps = effectiveFps();
      if (now - lastFrame < 1000 / targetFps) return;
      lastFrame = now;
      // Advance the simulation clock by real elapsed time (scaled against
      // the configured `fps`), not a fixed amount per rendered frame — so
      // throttling to a lower effectiveFps at a tall aspect makes the
      // animation choppier, not slower. Without this the plume would
      // visibly rise at half speed wherever the throttle cuts frame rate
      // in half, since half as many fixed-size steps would land per
      // second of real time.
      t += timeStep * (fps / targetFps);
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
