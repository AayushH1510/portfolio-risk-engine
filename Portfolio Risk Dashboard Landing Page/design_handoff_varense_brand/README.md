# Handoff: Varense Brand System v1.0

## Overview

The visual identity for **Varense**: one espresso luminance ramp, exactly two
reserved accent colours, two typefaces, and a logo derived from what the engine
computes — a domed outcome envelope with the realised path climbing through it.

The governing idea: **in a risk product, colour is data.** So colour is spent
almost nowhere. Everything structural is carried by luminance, hairlines, and
type scale, which leaves two hues free to mean something exact.

## Fidelity

**High-fidelity and authoritative.** Every value in `brand-tokens.json` is final.
The logo geometry in particular is exact — see "Logo traps" below before touching
a path.

`reference/Varense Brand.dc.html` is the living spec. Open it in a browser to see
every lockup, size step, specimen, and swatch. It is a **reference**, not code to
copy; implement from `brand-tokens.json` and `src/Logo.jsx`.

---

## ⚠ Reconciling with the landing-page handoff

This brand **supersedes parts of** `design_handoff_varense_landing/tokens.json`.
If that page is already built, these are breaking changes:

| Thing | Landing page v1 | Brand v1.0 |
|---|---|---|
| Display face | Newsreader (serif, 200/300) | **Archivo** (grotesk, 500) |
| Body face | IBM Plex Sans | **Archivo** — Plex Sans retired |
| Mono | IBM Plex Mono | IBM Plex Mono (unchanged) |
| Interaction accent | `#4FD8A0` | **`#5CC49B`** (muted) |
| Positive figures | `#4FD8A0` mint | **`#F6F1EA` bone** |
| Loss | `#E0555A` | **`#C9544B`** (warmed to hue) |
| Warning/amber | `#D9A441` | **retired** — fold into loss or bone |
| Chart neutrals | 3 steps | use the ramp directly |

Three families become two, and three signal colours become two. The type
**scale** is renamed too (`displayXL` → `display-xl`, and `lead`/`body-*` now
resolve to Archivo). Migrate `tokens.json` to the brand values rather than
running both — a page with Newsreader headlines and Archivo body copy is neither
system.

---

## What to build

### 1. Tokens → theme

Generate the theme from `brand-tokens.json` exactly as the landing-page handoff
describes (prebuild script → CSS custom properties, imported by `index.css`).
Same rule: **no hex codes, font stacks, or magic pixel values in components.**

The `type` block should emit utility classes or CSS custom properties keyed by
the 18 step names, so `display-m` is a single token rather than four properties
re-typed per component.

### 2. Logo component

Drop `src/Logo.jsx` in as-is (`src/components/Logo.jsx`). It is the only place
logo geometry may exist. It handles all three lockups, all three detail levels,
and takes `ink` so it inherits `currentColor` by default.

Replace the current gradient-square placeholder in the landing nav and footer
with `<Logo />` and `<Logo size={38} />`.

### 3. Favicons and app icons

`assets/favicon-16.svg` and `assets/mark.svg` are ready. You still need raster
fallbacks — generate them, don't hand-draw:

```bash
npm i -D sharp
# 180 (apple-touch), 32, 16 — each on the #16110D field with 22% corner radius
```

```html
<link rel="icon" href="/favicon.svg" type="image/svg+xml" />
<link rel="icon" href="/favicon-32.png" sizes="32x32" />
<link rel="apple-touch-icon" href="/apple-touch-icon.png" />
<meta name="theme-color" content="#0C0908" />
```

Use the **solid** detail level for 16px and the **simple** level for 32px — the
full path turns to mud below 64px. `Logo.jsx` picks correctly if you pass
`detail`; the ladder is in `brand-tokens.json > logo.detailLadder`.

### 4. Fonts

```bash
npm i @fontsource-variable/archivo @fontsource/ibm-plex-mono
```

```js
// src/main.jsx — before index.css
import '@fontsource-variable/archivo';
import '@fontsource/ibm-plex-mono/400.css';
import '@fontsource/ibm-plex-mono/500.css';
```

Remove every Newsreader and IBM Plex Sans import. Archivo variable covers
400/500/600 in one file.

### 5. Grain overlay

One fixed element, mounted once at the app root. Full CSS is in
`brand-tokens.json > texture.grain.css`. It is not optional — it carries most of
the perceived depth, and the design reads flat and cheap without it.

---

## Logo traps

**1. The envelope uses `large-arc-flag=1`.** That puts the ellipse centre at
cy≈40.95, so the dome is **37 units tall, not 30**. Rebuilding a plain
rx27/ry30 half-ellipse gives a visibly shallower dome that no longer matches the
logo. Copy the path string; do not re-derive it.

**2. Content spans y 10.95 → 48, not 14 → 48.** Any `viewBox` starting at y=14
clips the apex. Use `3 9 58 41` (or `2 5 60 45` for the 16px solid variant).
This was a real shipped bug.

**3. The mark is wide, not square** — 58 × 41, aspect 1.415. Always set width
and height as a matched pair, or use `Logo.jsx`, which derives width from height.

**4. Detail strips the path, never the envelope.** The dome and baseline are
constant at every size. Dropping the baseline turns the mark into the WiFi glyph;
dropping the envelope leaves a generic sparkline.

---

## Colour policy — the part most likely to be violated

Two accents, each with exactly one job:

- **`#5CC49B` interaction** — buttons, links, active states, focus rings. Never on
  a chart, never on a figure. *If it appears next to a number, it is wrong.*
- **`#C9544B` loss** — VaR, CVaR, drawdown, the left tail. Never a button, never a
  border, never decoration. Large text only; it fails AA at body sizes on ground.

**Positive figures are bone, not green.** Sharpe 1.48 renders `#F6F1EA`; VaR −$180
renders `#C9544B`. Loss being the only hue in the data layer is what makes it
land. Reintroducing green for gains destroys the entire policy.

Everything else — every border, every background, every label, every chart line —
comes from the 12-step ramp.

---

## Files

```
design_handoff_varense_brand/
├── README.md
├── brand-tokens.json          ← authoritative; generate the theme from this
├── src/
│   └── Logo.jsx               ← production component; the only home for geometry
├── assets/
│   ├── mark.svg               ← 58×41, full detail
│   ├── mark-inverse.svg       ← dark ink for light fields
│   ├── mark-32.svg            ← simple detail
│   ├── favicon-16.svg         ← solid detail
│   ├── logo-horizontal.svg    ← lockup (convert text to outlines before external use)
│   └── logo-stacked.svg
└── reference/
    └── Varense Brand.dc.html  ← the living spec; open in a browser
```

The two lockup SVGs use a `<text>` element, so they render correctly only where
Archivo is available. **Convert text to outlines** before sending them anywhere
external (press, partner sites, email signatures).

## Definition of done

- [ ] `brand-tokens.json` generates the theme; no hex codes in components.
- [ ] All 18 type steps exist as named tokens.
- [ ] `<Logo />` replaces every hand-rolled mark; no logo paths outside `Logo.jsx`.
- [ ] 16 / 32 / 180 icons use the correct detail level; nothing is clipped.
- [ ] Newsreader and IBM Plex Sans fully removed.
- [ ] Grain overlay mounted at the app root.
- [ ] No green on any figure; no accent on any chart.
- [ ] Landing-page `tokens.json` migrated, not run in parallel.
