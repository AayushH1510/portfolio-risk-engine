# Handoff: Varense Landing Page

## Overview

A marketing landing page for **Varense**, a portfolio risk analysis engine whose
application already lives at `/app`. The page sells the engine to four audiences
at once — retail investors, prosumer traders, RIAs, and institutional risk teams —
in a quiet, institutional register. It is deliberately not a growth-hacked SaaS
page: no urgency, no exclamation marks, no invented social proof.

Structure, in order: sticky nav → full-viewport hero over an animated espresso
plume → a tilted replica of the dashboard → stats band → what it is → why it
matters → how it works → four feature deep-dives with data visuals → method &
trust → pricing → FAQ → closing CTA → footer.

## About the design files

**The files in `reference/` are design references created in HTML. They are not
production code to copy.** They demonstrate intended appearance, layout, motion,
and copy. Your task is to recreate them in the target codebase — the existing
**Vite + React** app in `portfolio-risk-ui/` that already serves the dashboard at
`/app` — using its established patterns, component library, and styling approach.

The files in `src/` are different: those **are** intended for production. They are
framework-agnostic TypeScript, extracted from the prototype and rewritten to be
configuration-driven. Drop them in, wire them up.

## Fidelity

**High-fidelity.** Colors, type scale, spacing, shadows, and motion timings are
final and are enumerated in `tokens.json`. Recreate them faithfully. Copy is
final and lives in `src/content.ts`.

Two things are explicitly **placeholder** and must be replaced before launch:

1. **Pricing.** The Free / $19 / $79 tiers and their feature lists are invented.
2. **"a licensed market data provider"** in the Method section and FAQ. Replace
   with the real vendor name.

---

## Where this goes in the repo

The target is `portfolio-risk-ui/` — Vite + React, `.jsx` components, routed from
`App.jsx`, deployed to Vercel. The landing page becomes `/`; the dashboard stays
at `/app`.

```
portfolio-risk-ui/
├── design/
│   └── tokens.json                 ← from this bundle; build input for the theme
├── src/
│   ├── content/
│   │   └── landing.ts              ← from this bundle (src/content.ts)
│   ├── lib/motion/
│   │   ├── smoke-field.ts          ← from this bundle
│   │   └── scroll-motion.ts        ← from this bundle
│   ├── hooks/
│   │   ├── useSmokeField.js        ← you write; thin useEffect wrappers
│   │   └── useScrollMotion.js
│   ├── components/landing/         ← you write; one file per block + registry
│   └── pages/
│       └── Landing.jsx             ← you write
└── .env.example                    ← add VITE_APP_URL
```

`README.md` and `reference/` stay in the handoff folder as documentation.

**Two Vite specifics:**

1. **TypeScript.** Vite transpiles `.ts` with no configuration, so `content.ts`,
   `smoke-field.ts`, and `scroll-motion.ts` work as-is next to `.jsx` files. But
   with no `tsconfig.json` in the project, nothing type-checks them — the types
   are documentation, not a build gate. If you want them enforced, add a
   `tsconfig.json` with `"allowJs": true` and `"checkJs": false` and run `tsc
   --noEmit` in CI. Otherwise leave them; they still guide editors.
2. **Environment variables.** `content.ts` ships with
   `process.env.NEXT_PUBLIC_APP_URL`. Change it to
   `import.meta.env.VITE_APP_URL ?? '/app'` and add `VITE_APP_URL` to
   `.env.example`. Vite exposes nothing to the client without the `VITE_` prefix.

---

## The two rules that make this maintainable

The brief for this handoff was "nothing hardcoded, sustainable to scale." Two
constraints deliver that. Everything below is downstream of them.

### Rule 1 — All visual values come from `tokens.json`

`tokens.json` is the single source of truth for color, type, spacing, radius,
shadow, motion, and the smoke-field parameters. No component may contain a hex
code, a font stack, or a magic pixel value.

Generate your styling layer from it at build time rather than transcribing it.
Pick whichever fits the existing app:

**If the app uses Tailwind** — map the token file into `tailwind.config.js`:

```ts
import tokens from './design/tokens.json';
// Vite resolves JSON imports natively; no plugin needed.

const flat = (group: Record<string, any>) =>
  Object.fromEntries(Object.entries(group).map(([k, v]) => [k, v.value ?? v]));

export default {
  theme: {
    extend: {
      colors: {
        bg: flat(tokens.color.bg),
        line: flat(tokens.color.line),
        ink: flat(tokens.color.text),
        accent: flat(tokens.color.accent),
        signal: flat(tokens.color.signal),
      },
      fontFamily: {
        display: [tokens.font.display.family, tokens.font.display.fallback],
        sans: [tokens.font.body.family, tokens.font.body.fallback],
        mono: [tokens.font.mono.family, tokens.font.mono.fallback],
      },
      borderRadius: tokens.radius,
      boxShadow: tokens.shadow,
      maxWidth: { shell: tokens.layout.maxWidth },
    },
  },
};
```

**If the app uses plain CSS** (which `index.css` suggests) — emit custom
properties with a prebuild script and import the result from `index.css`. Add
`"prebuild": "node scripts/tokens-to-css.mjs > src/theme.css"` to `package.json`
scripts, plus the same command under `predev` so local runs stay in sync:

```js
import tokens from '../design/tokens.json' with { type: 'json' };
// Node 20.10+. On older Node use: assert { type: 'json' }

const lines = [];
const walk = (node, path = []) => {
  for (const [k, v] of Object.entries(node)) {
    if (k.startsWith('$')) continue;
    if (v && typeof v === 'object' && 'value' in v) {
      lines.push(`  --${[...path, k].join('-')}: ${v.value};`);
    } else if (v && typeof v === 'object' && !Array.isArray(v)) {
      walk(v, [...path, k]);
    } else {
      lines.push(`  --${[...path, k].join('-')}: ${v};`);
    }
  }
};
walk(tokens.color, ['color']);
walk(tokens.space, ['space']);
walk(tokens.radius, ['radius']);
walk(tokens.shadow, ['shadow']);
walk(tokens.layout, ['layout']);
console.log(`:root {\n${lines.join('\n')}\n}`);
```

Yielding `--color-accent-mint`, `--color-bg-panel`, `--space-9`, and so on. Add
the script to `prebuild` so a token edit can never drift from the stylesheet.

Do not hand-maintain a parallel copy of these values in a `constants.ts`.

### Rule 2 — All content comes from `src/content.ts`

The page is a **list of typed blocks**, not a hardcoded sequence of JSX. Copy,
metrics, prices, FAQ items, nav links, and section order all live in the
`landingPage` object. Components receive a block and render it.

```jsx
// src/pages/Landing.jsx
import { landingPage } from '../content/landing';
import { BLOCK_RENDERERS } from '../components/landing/registry';
import { useScrollMotion } from '../hooks/useScrollMotion';

export default function Landing() {
  const { blocks, motionIntensity } = landingPage;
  useScrollMotion({ intensity: motionIntensity / 100 });

  return blocks.map((block, i) => {
    const Renderer = BLOCK_RENDERERS[block.type];
    if (!Renderer) {
      if (import.meta.env.DEV) console.error(`No renderer for block type "${block.type}"`);
      return null;
    }
    return <Renderer key={block.id ?? `${block.type}-${i}`} block={block} id={block.id} />;
  });
}
```

`useScrollMotion` must run **after** the blocks have mounted, since
`initScrollMotion` queries `[data-parallax]` and `[data-reveal]` at call time.
A `useEffect` in the page component satisfies that — effects run bottom-up, so
children are in the DOM by then.

```js
// src/components/landing/registry.js
export const BLOCK_RENDERERS = {
  hero: Hero,
  productPlate: ProductPlate,
  statsBand: StatsBand,
  prose: Prose,
  cards: Cards,
  steps: Steps,
  features: Features,
  pillars: Pillars,
  pricing: Pricing,
  faq: Faq,
  cta: Cta,
};
```

In TypeScript this registry would carry
`satisfies Record<Block['type'], ComponentType<…>>` and the build would fail on
an unhandled variant. This project is `.jsx`, so add the guard yourself — either
the `import.meta.env.DEV` check above, or a one-line test that asserts every
`block.type` in `landingPage.blocks` has a key in `BLOCK_RENDERERS`. Without one,
a missing renderer silently drops a section.

**Adding a section six months from now** is therefore: add an interface to
`content.ts`, add it to the `Block` union, write one component, register it,
append an entry to `blocks`. Zero edits to existing sections.

**Moving to a CMS** is a drop-in: mirror the `content.ts` types as a zod schema,
parse the CMS payload at the fetch boundary, hand the result to the same
renderer. The component layer never learns where the content came from.

---

## Screens / Views

One page, twelve blocks. Every block sits inside the same shell:

```
max-width: 1240px; margin-inline: auto; padding-inline: clamp(20px, 3vw, 40px)
```

Sections alternate between `bg.base` (#0C0908) and `bg.raised` (#0E0B09), with a
1px `line.subtle` (#1C1611) rule at each boundary. Never more than two background
values on the page.

Six of the blocks share one **two-column section frame**:

```
grid-template-columns: clamp(150px, 18vw, 300px) minmax(0, 1fr);
gap: clamp(28px, 4.5vw, 90px);
align-items: start;
```

Left column: a mono, uppercase, letter-spaced `01 — What it is` label with a 1px
top rule and 16px top padding. Right column: the content. Build this once as
`<SectionFrame index label>` and reuse it — Prose, Cards, Steps, Pillars,
Pricing, and FAQ all use it.

### 1. Nav (sticky)

- `position: fixed`, full width, `z-index: 50`, min-height 72px, `flex-wrap: wrap`.
- At rest: transparent background, transparent bottom border.
- Scrolled past 24px: background `rgba(12,9,8,0.78)`, `backdrop-filter: blur(20px) saturate(1.2)`, bottom border `line.subtle`. Transition both over 300ms ease.
- Left: 26px rounded-6px logo mark, `linear-gradient(145deg, #4FD8A0, #2E9B72)`, `box-shadow: 0 0 22px rgba(79,216,160,0.35)`, then wordmark in mono 15px/500.
- Centre: five `monoNav` links, `text.ghost` → `text.body` on hover, gap `clamp(16px, 2vw, 34px)`.
- Right: outlined mint button, 10px/20px padding, radius 4px, mint border and text; on hover the fill becomes mint and the text becomes `bg.base`.

### 2. Hero

- `min-height: 100vh`, flex column, centred, `padding: 150px 40px 0`.
- **Background:** absolutely-positioned `<canvas>` at `inset: -10% -5% 0`, sized 110% wide and offset `left: -5%` so the plume bleeds past the viewport edge. `filter: saturate(1.15)`. Above it, two gradient scrims (values in the Interactions section — do not darken them further; the plume was clipped to black twice during design).
- **Eyebrow:** blinking 6px mint dot (2.8s ease-in-out), then `Portfolio risk engine / v1.0` in `monoEyebrow`, `text.ghost`. Bottom margin 34px.
- **Headline:** `displayXL`, `text.primary`, max-width 880px, `text-wrap: balance`. "before" is italic in `text.secondary`. Bottom margin 34px.
- **Lead:** `lead` type, `text.muted`, max-width 640px, bottom margin 46px.
- **Actions:** flex, gap 16px, wrap. Primary is mint fill on `bg.base` text with `shadow.action`, lifting 2px on hover. Secondary is `line.interactive` border → `line.hover`. Then the reassurance line in `monoCaption`, `text.ghost`.
- **Metric strip** (92px below): five cells, `repeat(auto-fit, minmax(140px, 1fr))`, 1px gap, container border `line.default`, radius 6px, `overflow: hidden`. Each cell: `bg.panel`, `shadow.hairline`, 20px/22px padding, `monoTile` label in `text.faint`, then a 26px mono value tinted by tone. The `/100` suffix drops to 16px `text.ghost`.
- Headline group carries `data-parallax="0.18"`; metric strip carries `0.08`.

### 3. Product plate

A replica of the real dashboard, rotated `rotateX(11deg) scale(0.985)` with
`transform-origin: 50% 0%` inside a `perspective: 2200px` parent, wrapped in a
`data-parallax="0.22"` container. Border `line.strong`, radius 12px, background
`bg.panel`, `shadow.plate`.

- **Chrome bar:** 46px, `bg.raised`, three 9px `#2E2620` dots, then five tab labels in mono 11px/0.13em uppercase. Active tab is `text.primary` with a 2px mint bottom border.
- **Body:** `minmax(0, 232px) minmax(0, 1fr)`, min-height 460px.
  - Sidebar: numbered field groups (`1. STOCKS`, `2. PORTFOLIO VALUE`, `3. WEIGHTS`) with `bg.inset` inputs bordered `line.strong`; three weight rows each with a mono ticker, mint percentage, and a 3px track (`line.default`) filled mint; a full-width mint `RUN ANALYSIS` button pinned to the bottom with `margin-top: auto`.
  - Main: four metric tiles (`repeat(auto-fit, minmax(112px, 1fr))`, 12px gap) — Annual Return and Max Drawdown carry a 1px accent top border in mint and amber respectively. Then a growth chart panel: mono header with a legend, and an SVG (`viewBox="0 0 800 230"`, `preserveAspectRatio="none"`) drawing a mint area fill under a 2px mint line plus a dashed `text.ghost` S&P benchmark, both with `vector-effect="non-scaling-stroke"`. Then an insight callout: 2px mint left border, `alpha.mintWash` background, mono kicker, 12.5px body with figures in `text.body`.

**Roadmap note:** this is a hand-built replica and will drift from the product.
The block already carries `source: 'mock' | 'live'` and a `fixture` id. When the
dashboard component can render from a fixture without auth, render the real one
inside a `pointer-events: none` shell and flip the flag. Do not delete the mock —
keep it as the SSR/no-JS fallback.

### 4. Stats band

`bg.raised`, rules top and bottom. `repeat(auto-fit, minmax(170px, 1fr))`, 46px
vertical padding per cell. Figure in `statM`; the `d` in `252d` drops to 24px
`text.ghost`. Caption in `monoLabel`, `text.faint`, 12px above.

### 5. What it is — `prose`

Section frame. `displayM` heading, then `lead`-sized paragraphs in `text.muted`,
max-width 700px, 24px apart. Bold spans (`**…**` in the content file) render as
`text.body` — weight is unchanged, only color.

### 6. Why it matters — `cards`

Section frame. Heading, 60px gap, then three cards:
`repeat(auto-fit, minmax(230px, 1fr))`, 1px gap, container border `line.default`,
radius 8px, `overflow: hidden`. Each card: `bg.panel`, `shadow.hairline`,
38px/32px padding, mono ordinal tinted by the card's tone, `headingS` title,
`bodyS` copy in `text.dim`. Hover raises the background to `bg.inset` over 300ms.

### 7. How it works — `steps`

Section frame. Three rows, each `64px minmax(0, 1fr) minmax(0, 260px)` with
`gap: clamp(18px, 2.5vw, 40px)`, `align-items: baseline`, 40px vertical padding,
1px top rule (last row also gets a bottom rule). Ordinal in display 44px/200 mint;
`headingM` title; `bodyM` copy capped at 460px; annotation column in mono 12px
`text.ghost`, `line-height: 1.9`, 1px left border with 24px padding — its final
line is mint.

### 8. The engine — `features`

Four alternating rows: `repeat(auto-fit, minmax(330px, 1fr))`,
`gap: clamp(40px, 5vw, 90px)`, `align-items: center`, 80px vertical padding, 1px
bottom rule between rows. `visualSide` decides which side the panel sits on (use
CSS `order`, not duplicated markup). Copy side: mono kicker tinted by
`kickerTone`, `displayS` heading, `bodyL` copy at 500px, then mono 12px tags with
34px gaps. Visual side: `bg.panel` panel, `line.default` border, radius 8px,
34px padding, `shadow.card`, `data-parallax="0.06"`.

The four visuals:

- **`distribution`** — a 15-bar flex histogram, 190px tall, 4px gaps, bar heights `6 11 19 31 48 66 84 100 92 74 55 38 24 14 7` percent. The left three bars are `signal.negative`, the fourth `signal.warning`, the peak `chart.neutral100`, the rest `chart.neutral200`, the right three `chart.neutral300`. Below it, a two-cell VaR/CVaR readout.
- **`simulationPaths`** — `viewBox="0 0 520 250"`. 24 seeded random walks stroked `text.body` at 0.7px with opacity 0.07–0.18, then a 2.2px mint median and a 1.4px dashed red 5th-percentile. **Generate the walks from a fixed seed** (an LCG, as in the prototype) so server and client markup match and the panel never hydration-mismatches. Legend row underneath, 1px top rule.
- **`stressBars`** — three labelled rows (2008 −48.2%, 2020 −31.7%, 2022 −24.1%), each a 6px `line.subtle` track filled `linear-gradient(90deg, deep → signal)` at 96% / 63% / 48%. Red for the first two, amber for 2022. Footer line: recovery times, 1px top rule.
- **`frontier`** — `viewBox="0 0 460 260"`. Axis lines in `line.default`, a mint cubic frontier curve, eight `chart.neutral100` scatter dots, an amber "YOU" marker and a mint "OPTIMAL" marker (each a 6px dot inside a 12px 40%-opacity ring) with mono 10px labels.

### 9. Method & trust — `pillars`

Section frame. `displayM` heading, 50px gap, then a
`repeat(auto-fit, minmax(255px, 1fr))` grid, `gap: 44px clamp(30px, 4vw, 70px)`.
Each pillar: mono 13px/0.12em uppercase mint title, `bodyM` copy in `text.dim`.

### 10. Pricing

Section frame with a full-width heading above the grid.
`repeat(auto-fit, minmax(255px, 1fr))`, 20px gap. Tier card: `bg.panel`,
`line.default` border, radius 10px, 40px/34px padding, flex column, 26px gap,
CTA pinned bottom with `margin-top: auto`. Featured tier differs: border
`accent.mintBorder`, background `linear-gradient(180deg, #12100D, #100C0A)`,
`shadow.tier`, a mint primary CTA, and a 1px `linear-gradient(90deg, transparent,
mint, transparent)` hairline inset 34px along its top edge.

Price in `statL`; the `/mo` period drops to mono 15px `text.ghost`.

### 11. FAQ

Section frame. Native `<details>` / `<summary>` — no JS, no state, and the
browser's own disclosure marker rotates for free. Set `summary { color: text.ghost }`
so the marker inherits a muted tint, and put the question inside a `headingXS`
span in `text.primary`. Each item: 28px vertical padding, 1px bottom rule (omit
on the last). Answer: 16.5px/1.75 `text.dim`, max-width 640px, 18px top margin.

### 12. Closing CTA + footer

CTA: `bg.raised`, 170px vertical padding, centred, max-width 800px, plus a
`radial-gradient(70% 130% at 50% 100%, alpha.mintGlow, transparent 62%)` bloom.
`displayL` heading, 18.5px lead, two buttons centred.

Footer: 1px top rule, 60px/40px/46px padding. Left: logo mark, wordmark, version,
then the disclaimer in `bodyXS` `text.ghost` at 380px. Right: three link columns,
`gap: clamp(32px, 5vw, 70px)`, wrapping; column headings in mono 10.5px/0.16em
`text.quiet`, links 14px `text.dim` → `text.body`. Bottom bar: 1px top rule, 26px
above, mono 11.5px `text.quiet`, copyright left and tagline right.

---

## Interactions & behavior

All motion is progressive enhancement. **Nothing may start at `opacity: 0` in
CSS** — the reveal module applies the hidden state in script so that no-JS and
pre-hydration users see a complete page.

### Reveal on scroll

`data-reveal` on any element. IntersectionObserver at `threshold: 0.08`,
`rootMargin: '0px 0px -12% 0px'`. On intersect: opacity 0 → 1 and
`translateY(22px)` → 0 over 900ms `cubic-bezier(.16,1,.3,1)`, with a 90ms stagger
against reveal siblings, capped at 360ms.

A **scroll-tick fallback** re-checks anything still pending and reveals whatever
is already above 94% of the viewport. This is required, not optional: a scrollbar
drag, `Cmd+End`, or a deep hash link outruns the observer and otherwise strands
blocks invisible. Already implemented in `src/scroll-motion.ts`.

### Parallax

`data-parallax="<speed>"` on any element. Travel is
`-centreOffsetRatio × speed × 190px × intensity`, where `centreOffsetRatio` is the
element centre's distance from viewport centre in viewport heights. Intensity is
`landingPage.motionIntensity / 100`, default 0.35.

Speeds in use: hero copy `0.18`, hero metric strip `0.08`, product plate `0.22`,
feature visual panels `0.06`. Elements more than 300px outside the viewport are
skipped. One rAF-throttled listener serves the whole page.

The plate's static `rotateX` tilt must live on an **inner** element — parallax
overwrites `transform` on the element it targets.

### Nav

Crosses at `scrollY > 24`. Background, `backdrop-filter`, and border-color all
transition over 300ms ease. Driven from the same scroll tick via the `onScroll`
callback, not a second listener.

### Hero smoke field

Domain-warped fractal-noise plume, animated on a `<canvas>`. Rendered at
108×66 into an offscreen buffer, upscaled onto a 520px-wide canvas with a 9px
blur, then CSS-stretched — so the visible resolution is nearly free. Capped at
24fps and paused when the tab is hidden.

Two scrims sit over it. **These are tuned to the edge; darkening them kills the
plume:**

```css
radial-gradient(125% 95% at 50% 6%,
  rgba(12,9,8,0) 0%, rgba(12,9,8,0.14) 54%, rgba(12,9,8,0.82) 86%, #0C0908 100%)

linear-gradient(to bottom,
  rgba(12,9,8,0.32) 0%, rgba(12,9,8,0) 24%, rgba(12,9,8,0) 64%, #0C0908 100%)
```

**Acceptance test.** After any change to the plume or the scrims, sample the
canvas and assert it is actually visible:

```js
const c = document.querySelector('canvas');
const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
let max = 0, sum = 0, above = 0, n = 0;
for (let i = 0; i < d.length; i += 4) {
  const v = Math.max(d[i], d[i + 1], d[i + 2]);
  max = Math.max(max, v); sum += v; if (v > 40) above++; n++;
}
// Expect roughly: max ≈ 176, avg ≈ 37, pctAbove40 ≈ 29%
```

If `max` is under 60 the field has collapsed to the darkest ramp stop. See
Known trap #1.

### Reduced motion

`prefers-reduced-motion: reduce` must disable parallax, skip the reveal
transition (render everything visible immediately), and paint a single static
smoke frame. `scroll-motion.ts` handles the first two; pass
`reducedMotion: true` to `createSmokeField` for the third.

### Hover states

- Primary button: `translateY(-2px)` + `shadow.action` → `shadow.actionHover`, 200ms.
- Secondary button: border `line.interactive` → `line.hover`, text `text.secondary` → `text.primary`.
- Nav / footer links: `text.ghost`|`text.dim` → `text.body`.
- Why-it-matters card: background `bg.panel` → `bg.inset`, 300ms.

---

## Responsive rules

The page has no media queries. It is fluid by construction, and must stay that
way — every grid is intrinsic.

1. **Never a bare fixed `px` track in `grid-template-columns`.** Use
   `clamp()` for label columns and `minmax(0, …)` for content columns. A bare
   `300px 1fr` overflows below ~1100px because grid items default to
   `min-width: auto` and refuse to shrink.
2. **Card grids use `repeat(auto-fit, minmax(<floor>, 1fr))`.** Floors in use:
   140px (metric strip), 170px (stats), 230px (why cards), 255px (pillars,
   pricing), 330px (feature rows), 112px (plate tiles).
3. **Section padding and gaps are `clamp()`,** never fixed.

Verified clean at 909px content width. Sanity-check 375px, 768px, 1024px,
1440px, and 1920px.

---

## State management

Almost none — this is a static marketing page.

- **No React state for reveal or parallax.** Both mutate DOM style directly from a rAF loop. Putting scroll position in state re-renders the tree every frame.
- **No state for the FAQ.** Native `<details>`.
- Two thin hooks are the only stateful code. Each calls its module in a `useEffect` with an empty dep array and returns the `destroy` fn for cleanup:

```js
// src/hooks/useSmokeField.js
export function useSmokeField(canvasRef, options) {
  useEffect(() => {
    if (!canvasRef.current) return;
    const field = createSmokeField(canvasRef.current, options);
    return field.destroy;
  }, []);
}
```

  Strict mode double-invokes effects in dev — both modules are idempotent and clean up fully, so this is safe, but keep it that way if you modify them.
- No data fetching, no router state, no context. If content later moves to a CMS, fetch it once at the page level and pass it down — the block renderers stay pure.

---

## Files

```
design_handoff_varense_landing/
├── README.md                  ← this file
├── tokens.json                ← all design values; generate CSS/Tailwind from it
├── src/
│   ├── content.ts             ← every string, price, and metric on the page
│   ├── smoke-field.ts         ← hero canvas engine (framework-agnostic)
│   └── scroll-motion.ts       ← parallax + reveal (framework-agnostic)
└── reference/
    ├── Varense Landing.dc.html  ← the design prototype
    └── support.js               ← runtime the prototype needs to render
```

Open `reference/Varense Landing.dc.html` in a browser to see the design running,
including all motion. Read it for anything this README leaves ambiguous — but
implement from the README and `src/`, not by copying its markup.

## Assets

None to transfer. Every graphic is drawn in code — the logo mark is a CSS
gradient, the charts are inline SVG, the histogram and stress bars are flex
`<div>`s, and the hero texture is procedural. No image, icon set, or font file
needs to be hosted.

**Fonts** are all Google Fonts. Prefer `@fontsource` so they are bundled and
self-hosted — no render-blocking third-party request, and no layout shift:

```bash
npm i @fontsource-variable/newsreader @fontsource/ibm-plex-sans @fontsource/ibm-plex-mono
```

```js
// src/main.jsx — import before index.css
import '@fontsource-variable/newsreader';
import '@fontsource-variable/newsreader/wght-italic.css';
import '@fontsource/ibm-plex-sans/300.css';
import '@fontsource/ibm-plex-sans/400.css';
import '@fontsource/ibm-plex-sans/500.css';
import '@fontsource/ibm-plex-mono/400.css';
import '@fontsource/ibm-plex-mono/500.css';
```

Only load the weights listed in `tokens.json > font` — 200 and 300 for Newsreader
(plus italic 300 for the hero's emphasised word), 300/400/500 for Plex Sans,
400/500 for Plex Mono. Loading the full families costs ~400KB for nothing.

If you use the Google Fonts `<link>` in `index.html` instead, include the
`preconnect` pair to `fonts.googleapis.com` and `fonts.gstatic.com` (crossorigin)
and accept the FOUT.

---

## Known traps

Three real bugs were hit building this. They will recur if the reasons aren't known.

**1. The noise hash must use logical shifts.** `n ^ (n >> 16)` with an
*arithmetic* `>>` propagates the sign bit, so bit 31 of the result is always 0,
the hash caps at 0.5 instead of 1.0, and the entire plume collapses to the
darkest ramp stop — a black rectangle. Use `>>>`. `smoke-field.ts` has this right
and carries a comment; don't "simplify" it.

**2. Never fake grid dividers with a container background.** Setting the
container to the line color with `gap: 1px` over opaque children looks correct
only while the last row is full. The moment `auto-fit` wraps and leaves an empty
cell, the container background shows through as a lit rectangle. Instead, keep
the container transparent and give each cell `box-shadow: 0 0 0 1px <line>`
(`shadow.hairline`) — a missing cell then renders as nothing.

**3. Parallax overwrites `transform`.** Any element with `data-parallax` cannot
also carry a static transform. Put the tilt, scale, or rotation on a child.

## Definition of done

- [ ] `tokens.json` generates the theme; zero hex codes in component files.
- [ ] `content.ts` drives the page; zero user-facing strings in component files.
- [ ] The block registry `satisfies Record<Block['type'], …>` and the build fails on an unhandled variant.
- [ ] Smoke canvas passes the brightness assertion (max > 60).
- [ ] No horizontal overflow at 375 / 768 / 1024 / 1440 / 1920.
- [ ] Reveals all fire after a jump to the page bottom.
- [ ] `prefers-reduced-motion` disables parallax and reveal transitions.
- [ ] Page renders complete and readable with JavaScript disabled.
- [ ] Lighthouse: no CLS from font loading; the canvas is not the LCP element.
- [ ] Placeholder pricing and the market-data vendor name replaced.
- [ ] `VITE_APP_URL` added to `.env.example`; no `process.env` left in `content.ts`.
- [ ] `/` serves the landing page and `/app` still serves the dashboard unchanged.
- [ ] A missing block renderer fails loudly (dev guard or test), not silently.
- [ ] `npm run build` produces no new warnings; token generation runs in `prebuild`.
