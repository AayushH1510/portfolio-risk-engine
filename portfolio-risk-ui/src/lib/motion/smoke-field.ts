/**
 * smoke-field.ts — the hero's ambient espresso plume.
 *
 * Framework-agnostic. No React, no DOM assumptions beyond a <canvas>.
 * Every visual constant is an option; the defaults live in tokens.json > smoke
 * so the palette can be re-themed without touching this file.
 *
 * Three earlier attempts at making this aspect-safe retuned the *shape* of
 * the simulation per viewport (a hard aspect cap; rescaling row count 1:1
 * with aspect; a blur/fps retune to pay for it) — each traded one visible
 * defect for another because the field's own geometry kept changing. This
 * version fixes the geometry — one field, cover-scaled onto its container
 * the way `background-size: cover` would (see resize()'s CSS-scaling
 * comment) — and only ever adjusts *density* (`H`, the row count) on top of
 * that fixed geometry, within a bounded cost. Cover-scaling alone left phone
 * widths under-textured versus desktop: not because the field renders
 * differently, but because "cover" clips most of a tall container's width
 * away, and a fixed-resolution field only has so much detail in the sliver
 * that survives. `H` compensates for exactly that clipping — see its own
 * comment for the derivation — capped so cost never exceeds roughly 2x
 * desktop's baseline.
 *
 * Cost: gridWidth * H * octaves * 2 fbm calls per frame, capped at `fps`.
 * H is now the *only* aspect-dependent parameter — gridWidth, outputWidth,
 * blur, fps are all still fixed regardless of viewport. Desktop measures
 * ~1.7ms/paint at H's floor (66); phone widths measure ~3.2-3.3ms/paint at
 * H's ceiling (132) — see RESPONSIVE_AUDIT.md for the full measurement,
 * including why full parity with desktop's visible density isn't reachable
 * within that cost ceiling.
 */

export interface SmokeFieldOptions {
  /** [stop 0..1, hex] pairs, ascending. Maps field intensity to colour. */
  ramp: Array<[number, string]>;
  /** Simulation grid. Larger = more detail, more work. gridWidth is fixed
   *  regardless of viewport. gridHeight is the *floor* — the row count at
   *  or below the reference aspect ratio (desktop-shaped containers); it
   *  scales up from there at taller aspects, up to a cost-bounded ceiling.
   *  See resize()'s own comment for the derivation. */
  gridWidth?: number;
  gridHeight?: number;
  /** Backing store width the field is rendered at, before cover-scaling.
   *  Height follows gridHeight/gridWidth's own ratio. */
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

  // H, the offscreen buffer, and its ImageData are all re-derived in
  // resize() below, not fixed at setup — see that function's comment.
  let H = baseH;
  const offscreen = document.createElement('canvas');
  offscreen.width = W;
  offscreen.height = H;
  const octx = offscreen.getContext('2d')!;
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

  // REFERENCE_ASPECT: the ceiling of every desktop aspect ratio this hero
  // actually renders at (measured, not guessed — RESPONSIVE_AUDIT.md has
  // the survey: 0.666-0.706 across 1280/1366/1440 at this content length),
  // with headroom so a future copy edit that changes hero height by a line
  // or two doesn't silently cross it. At or below this aspect, H stays at
  // its floor (baseH) — desktop is untouched, not just "close": the exact
  // same 108x66 field it always rendered.
  //
  // MAX_GRID_HEIGHT: the cost ceiling. Rows scale H linearly with cost, so
  // this is chosen directly from the cost budget (~2x desktop's measured
  // ~1.67ms baseline) rather than from any visual target — see
  // RESPONSIVE_AUDIT.md for the measurement this was checked against.
  const REFERENCE_ASPECT = 0.75;
  const MAX_GRID_HEIGHT = 132;

  // Cover-scales the canvas's CSS box over its container, the same
  // algorithm `background-size: cover` uses — see the horizontal-centring
  // note in Hero.jsx for why the overflow this produces is safe. Layered
  // on top: H (row count) scales with the container's own aspect ratio,
  // floored at baseH and capped at MAX_GRID_HEIGHT. The reason this one
  // parameter is safe to make aspect-dependent where the geometry itself
  // isn't: with height always the binding axis (true at every aspect this
  // hero renders at, phone through desktop — confirmed by measurement, not
  // assumed), the number of grid *columns* that survive "cover" clipping
  // works out to exactly `H / aspect`, independent of W. Holding that
  // product roughly constant is what keeps visible detail comparable
  // across aspect ratios; the algebra is in RESPONSIVE_AUDIT.md. `H` is
  // read directly by paint() below via closure, so no other plumbing is
  // needed when it changes here.
  const resize = () => {
    const container = canvas.parentElement;
    const rect = (container ?? canvas).getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const aspect = rect.height / rect.width;

    const newH = Math.round(Math.min(MAX_GRID_HEIGHT, baseH * Math.max(1, aspect / REFERENCE_ASPECT)));
    if (newH !== H) {
      H = newH;
      canvas.height = Math.round(outputWidth * (H / W));
      offscreen.height = H;
      image = octx.createImageData(W, H);
      data = image.data;
    }

    const scale = Math.max(rect.width / canvas.width, rect.height / canvas.height);
    canvas.style.width = `${canvas.width * scale}px`;
    canvas.style.height = `${canvas.height * scale}px`;
  };
  canvas.width = outputWidth;
  canvas.height = Math.round(outputWidth * (H / W));
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

  // ResizeObserver on the container, not a window 'resize' listener — H
  // now depends on the container's aspect ratio, and that aspect ratio
  // changes for reasons a window resize listener never fires for: web
  // fonts swapping in after first paint (fallback-font line wraps differ
  // from the real font's, changing hero height with no window resize
  // event at all) being the one that actually bit this — confirmed live,
  // not theoretical: landscape's H settled at ~68 instead of the correct
  // 75 because resize() only ever ran once, against the pre-font-swap
  // layout. A ResizeObserver fires for any of these, window resize
  // included, so it replaces that listener rather than supplementing it.
  //
  // repaint after resize() when reducedMotion: resize() clears the canvas
  // whenever H changes (a fresh createImageData, and setting canvas.height
  // resets the bitmap per spec) but never repaints it itself — fine in the
  // normal animated path, where the next rAF frame repaints regardless, but
  // reducedMotion has no next frame, since paint() only ever runs once, at
  // setup. Without this, a font swap arriving after that one paint() (a
  // real, live race, not hypothetical — confirmed via two independent
  // screenshots of identical code: one showed the plume, one showed a
  // blank hero, differing only in whether the font had already swapped in
  // when this effect first ran) leaves the canvas permanently blank for
  // any reduced-motion visitor unlucky enough to hit it — a screenshot
  // harness comparing two page loads is just the first thing that made an
  // intermittent-in-production bug visible and reproducible.
  const ro = new ResizeObserver(() => {
    resize();
    if (reducedMotion) paint();
  });
  if (canvas.parentElement) ro.observe(canvas.parentElement);

  return {
    destroy() {
      cancelAnimationFrame(raf);
      ro.disconnect();
      document.removeEventListener('visibilitychange', onVisibility);
    },
  };
}
