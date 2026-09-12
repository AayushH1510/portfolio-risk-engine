# Responsive Audit — Varense (`portfolio-risk-ui/`)

Read-only audit. No files modified, no build run. `CLAUDE.md` and `DESIGN.md` read in full first (both quoted below where relevant).

---

## 1. Styling ground truth

**Plain CSS custom properties only. No Tailwind, anywhere, as of the current working tree.**

- `git status` is clean; `git log` shows Tailwind was wired in at some point, flagged as contradicting `CLAUDE.md`, and fully removed in commit `f5572fe2` ("chore: remove unused Tailwind CSS dependency (zero usage found, build 9s->4.2s, CSS bundle -23%)").
- Confirmed directly: `portfolio-risk-ui/src/index.css` has no `@import "tailwindcss"`; `vite.config.js` has no `@tailwindcss/vite` plugin (`plugins: [react()]` only); `package.json` has no `tailwindcss` / `@tailwindcss/vite` entries; there is no `tailwind.config.js` anywhere in `portfolio-risk-ui/`.
- So there is no Tailwind version to report and no `tailwind.config.js` vs. v4 `@theme` question to resolve — the question is moot for this codebase's current state.
- Styling is 100% inline `style={{...}}` objects reading CSS custom properties defined in `src/index.css`'s `:root` block (136 distinct `--*` custom properties, confirmed by direct grep), plus a small number of literal semantic class names (`card`, `btn-primary`, `spin`, `grain-canvas`, `grain-surface`, `vignette-layer`, `fade-up`) whose rules also live in `index.css`. This matches `CLAUDE.md` exactly: *"All inline styles — no Tailwind utility classes in component JSX."*

**Media queries: zero, confirmed.**

`grep -r "@media"` across all of `portfolio-risk-ui/` returns exactly one hit: [ExportPDF.jsx:465](portfolio-risk-ui/src/components/ExportPDF.jsx#L465), and it's `@media print` inside a hand-built HTML string used for the PDF-export report (`@page { margin: 0; size: A4 landscape; }`) — a print stylesheet for a generated document, not a responsive/viewport breakpoint. There is no `@media (max-width:...)` or `(min-width:...)` anywhere in the app. Confirmed as expected.

**DESIGN.md's token naming convention, quoted verbatim** (so new breakpoint tokens match it):

> `--space-1` | 4px
> `--space-2` | 8px
> `--space-3` | 12px
> ... (through `--space-20` | 80px)

> ### Layout
> | Token | Value |
> |-------|-------|
> | `--page-max-width` | 1440px |
> | `--sidebar-width` | 260px |
> | `--section-gap` | 48px |
> | `--card-padding` | 16px 20px |
> | `--element-gap` | 12px |

And from the Type Scale section — the pattern for a *named, not numbered* token when the value carries semantic meaning rather than a linear scale position:

> | Role | Size | Weight | Line Height | Tracking | Token |
> |------|------|--------|-------------|----------|-------|
> | caption | 10px | 500 | 1.4 | `0.08em` | `--text-caption` |
> | body-sm | 12px | 400 | 1.5 | `0` | `--text-body-sm` |

**Convention to follow for new breakpoint tokens:** kebab-case, no unit suffix in the name, grouped under a semantic prefix the way `--space-N` (numbered scale) and `--sidebar-width`/`--page-max-width` (named layout values) already are. A breakpoint set would most naturally read `--breakpoint-sm` / `--breakpoint-md` / `--breakpoint-lg` (naming the role, not a raw pixel value) sitting in the existing `## Layout` table alongside `--sidebar-width` and `--page-max-width`, since those are the only other viewport-scale layout tokens defined.

**Tailwind default scale vs. DESIGN.md — no live conflict, but worth flagging for later:** Tailwind v4's default breakpoints (`sm:640px md:768px lg:1024px xl:1280px 2xl:1536px`) don't exist in this codebase today, so there's nothing to actually collide with right now. If Tailwind (or its breakpoint *values*, even without the framework) is ever reintroduced, the one real number to reconcile is `--sidebar-width: 260px` plus its live drag range `SIDEBAR_MIN 220` / `SIDEBAR_MAX 400` ([Sidebar.jsx:15-17](portfolio-risk-ui/src/components/Sidebar.jsx#L15-L17)) against `--page-max-width: 1440px` — Tailwind's `md` (768px) sits well below where a 260-400px sidebar + usable content stops making sense as a two-column layout, so **DESIGN.md's own layout tokens should win**; a mobile breakpoint here needs to be chosen around where the sidebar-as-drawer transition actually has to happen (see §2), not borrowed from Tailwind's generic scale.

---

## 2. App shell

**Owner:** [App.jsx](portfolio-risk-ui/src/App.jsx) owns the whole shell. Structure (lines 128-331):

```
<div className="grain-canvas" style={{height:'100vh', overflow:'hidden', ...}}>   ← page root, 100vh, no scroll
  <div className="vignette-layer" />
  <div style={{display:'flex', height:'100%', ...}}>                              ← flex row: sidebar + content
    <Sidebar .../>                                                                ← flex item #1
    <div style={{flex:1, display:'flex', flexDirection:'column', overflow:'hidden', minWidth:0}}>  ← flex item #2
      <header>...tab bar...</header>                                              ← 48px fixed height
      <main style={{flex:1, overflow:'hidden', padding:16, minHeight:0}}>         ← the active tab renders here
      <div>...footer disclaimer + Privacy/Terms links...</div>
```

**Offset mechanism: flexbox, not margin/padding/grid.** The sidebar is a normal flex item with its own explicit width; the content column is `flex:1` and simply takes whatever space is left — there is no margin-left offset trick or grid-template-columns split anywhere in this shell. The sidebar's width comes from `<aside style={{width:'var(--sidebar-width)', minWidth:'var(--sidebar-width)', flexShrink:0, ...}}>` ([Sidebar.jsx:178-179](portfolio-risk-ui/src/components/Sidebar.jsx#L178-L179)) — a CSS custom property, not a literal number, which is exactly what makes the live-resize handle work (it just rewrites that one custom property).

**Resize handle:** [Sidebar.jsx:444-455](portfolio-risk-ui/src/components/Sidebar.jsx#L444-L455) — an absolutely-positioned 6px-wide invisible strip at `right:-3` on the `<aside>`, `cursor:col-resize`, highlighted green on hover. `onMouseDown` starts a drag (`handleResizeStart`, lines 68-95): tracks `clientX` deltas, clamps to `SIDEBAR_MIN=220` / `SIDEBAR_MAX=400` (lines 15-16), and writes the live value directly onto `document.documentElement.style.setProperty('--sidebar-width', ...)` on every `mousemove` (via a `sidebarWidth` state + a `useEffect` at lines 64-66). It is **mouse-only** — bound via `onMouseDown` / `window.addEventListener('mousemove'/'mouseup')`, no `onTouchStart`/`onPointerDown` equivalent, so it isn't draggable at all on a touchscreen (not dangerous, since a touch device would presumably not have this sidebar-as-column layout in the first place, but worth knowing if the drawer redesign ever keeps a resizable column on tablet-width).

**localStorage key:** `'varense-sidebar-width'` ([Sidebar.jsx:17](portfolio-risk-ui/src/components/Sidebar.jsx#L17), constant `SIDEBAR_STORAGE_KEY`). Read once on mount via `getInitialSidebarWidth()` (lines 22-28): checks `localStorage` first, clamps it into `[SIDEBAR_MIN, SIDEBAR_MAX]`, and only falls back to reading the live computed `--sidebar-width` CSS value (then a hardcoded `260`) if nothing is stored. **Written** only at drag-end (`onUp`, lines 82-88), not on every intermediate `mousemove` — a deliberate choice noted in a comment to avoid write-storms. Nothing else in the codebase reads `'varense-sidebar-width'` (confirmed via grep — the only other sidebar-related storage keys are `OnboardingTour`'s `'varense_has_visited'` and `FirstResultCallout`'s `'varense_has_seen_first_result'`, both unrelated).

---

## 3. Sidebar contents inventory

Every interactive element in [Sidebar.jsx](portfolio-risk-ui/src/components/Sidebar.jsx), in DOM order:

| # | Element | Classification |
|---|---------|----------------|
| 1 | Logo (`<Link to="/">`) | **Navigation** (leaves the app entirely, back to landing page) |
| 2 | Saved Portfolios header (click to expand/collapse) | Action (UI state, not portfolio data) |
| 3 | "+ Save current portfolio" button → name input → Save/Cancel | Action (persists current portfolio) |
| 4 | Saved-portfolio list rows (click to load) | Action (loads a saved portfolio's data-entry state) |
| 5 | Saved-portfolio delete (×) → confirm Delete/Keep | Action |
| 6 | **"1. Stocks"** ticker text input | **Portfolio data-entry** (tickers) |
| 7 | **"2. Portfolio value"** number input ($) | **Portfolio data-entry** (value) |
| 8 | **"3. Weights"** — "% Split" / "$ Amount" mode toggle | **Portfolio data-entry** (mode) |
| 9 | Per-ticker `TickerLabel` (click → opens Stock Drawer) | Action (drawer trigger), embedded inside data-entry rows |
| 10 | Per-ticker weight `<input type="range">` sliders (one per non-last ticker, in both pct and dollar mode) | **Portfolio data-entry** (weights) |
| 11 | **"4. Period"** — 1M/3M/6M/1Y/3Y/5Y/Max button grid | **Portfolio data-entry** (period) |
| 12 | "Custom dates" toggle button | **Portfolio data-entry** (mode) |
| 13 | 6× `<select>` (start/end × day/month/year), shown only if custom dates on | **Portfolio data-entry** (dates) |
| 14 | **"5. Risk window"** `<select>` (20d/30d/60d/90d) | **Portfolio data-entry** (analysis parameter) |
| 15 | **"6. Benchmark"** `<select>` (8 benchmarks + "None") | **Benchmark selection** |
| 16 | "Run Analysis" button | **Action button** (fires the actual API call) |
| 17 | Resize handle (drag strip) | Layout control, not portfolio input |

**The sidebar is not navigation.** Tab-to-tab navigation (Dashboard / Risk Analysis / Monte Carlo / Efficient Frontier / Valuation / Compare / Backtest / Learn) lives entirely in `App.jsx`'s header tab bar ([App.jsx:174-197](portfolio-risk-ui/src/App.jsx#L174-L197)), not in the sidebar. Apart from the one logo link back to `/`, every single sidebar element is either constructing the portfolio to analyze (tickers, weights, value, period, risk window, benchmark) or acting on that construction (save/load/delete/run). **The sidebar is the primary input surface for running an analysis** — closing it off entirely on mobile (rather than making it reachable as a drawer) would make the app unusable, not just less convenient.

---

## 4. Hardcoded dimensions

Grep for fixed px widths/heights, `minWidth`/`minHeight`/`maxWidth`, fixed `gridTemplateColumns` counts, and arbitrary Tailwind-style `w-[...]`/`h-[...]` values (none found — confirmed zero, consistent with §1) across the 9 named tab files, plus the shared components they use.

| File : Line | Value | What breaks below ~400px |
|---|---|---|
| [Dashboard.jsx:228](portfolio-risk-ui/src/pages/Dashboard.jsx#L228) | `gridTemplateColumns: '1fr 220px'` | **The single worst offender.** A 220px column is fixed regardless of container width. At ~390px viewport width (minus 32px `<main>` padding, minus 10px grid gap = ~348px available), the `1fr` track for the *primary growth chart* is squeezed to roughly 100-110px — far too narrow for the chart, its legend, or its Combined/By Holding toggle. Below ~330px total available width, the grid can no longer satisfy both tracks and forces horizontal overflow of the whole row. |
| [Dashboard.jsx:198](portfolio-risk-ui/src/pages/Dashboard.jsx#L198) | `gridTemplateColumns: 'repeat(5, 1fr)'` | 5 equal columns, no `minmax()` floor. At ~350px available width each card gets ~62-64px — narrower than `MetricCard`'s own padding (`14px 16px` non-small, [MetricCard.jsx:15](portfolio-risk-ui/src/components/MetricCard.jsx#L15)) plus a 22px mono value can comfortably hold. Labels wrap to 2 lines, cards grow taller instead of narrower, but there is no reflow to fewer columns. |
| [Dashboard.jsx:209](portfolio-risk-ui/src/pages/Dashboard.jsx#L209) | `gridTemplateColumns: secondCols` → `repeat(N, 1fr)`, N = 2-5 | Same failure mode as above, one row down (VaR/CVaR/Diversification/Beta/Alpha). |
| [MonteCarlo.jsx:124](portfolio-risk-ui/src/pages/MonteCarlo.jsx#L124) | `gridTemplateColumns: 'repeat(4, 1fr)'` | 4 metric tiles (`small` MetricCard, 18px value font) — tight but survivable to ~320px given short values (`$10,000`, `92%`). |
| [Backtest.jsx:169](portfolio-risk-ui/src/pages/Backtest.jsx#L169) | `gridTemplateColumns: 'repeat(3, 1fr)'` | 3 strategy summary cards — same category, moderate risk. |
| [Frontier.jsx:353](portfolio-risk-ui/src/pages/Frontier.jsx#L353) | `gridTemplateColumns: '1fr 1fr 1fr'` | Sharpe-ratio insight panel + 2 `WeightCard`s (one per ticker inside each, so this compounds with ticker count — see below). |
| [Comparison.jsx:30](portfolio-risk-ui/src/pages/Comparison.jsx#L30) / [:101](portfolio-risk-ui/src/pages/Comparison.jsx#L101) | `gridTemplateColumns: '1fr 80px 1fr'` | Used for **every** metric row (Return, Volatility, Sharpe, Sortino, Drawdown, VaR, CVaR, Beta, Alpha) and the A/VS/B header. The 80px fixed center column is less severe than Dashboard's 220px but still eats a fixed chunk from a value column that needs to show e.g. `34.3%` + a `WIN` badge. |
| [Comparison.jsx:136](portfolio-risk-ui/src/pages/Comparison.jsx#L136) | `gridTemplateColumns: '1fr 1fr'` | Metrics table + growth chart side by side — at mobile width this needs to become a single column or the chart gets crushed the same way Dashboard's does. |
| [CompareWrapper.jsx:153](portfolio-risk-ui/src/components/CompareWrapper.jsx#L153) | `gridTemplateColumns: '1fr 1fr 1fr auto'` | Portfolio-B config row: ticker text input, period button grid, portfolio-value number input, and a Run button all forced onto one row. This is the tightest single row in the whole app below ~400px. |
| [CompareWrapper.jsx:171](portfolio-risk-ui/src/components/CompareWrapper.jsx#L171) | `gridTemplateColumns: 'repeat(4, 1fr)'` | Period button grid inside that same config panel — duplicate of Sidebar's own period grid, same risk profile. |
| [CompareWrapper.jsx:206](portfolio-risk-ui/src/components/CompareWrapper.jsx#L206) | `` gridTemplateColumns: `repeat(${tickers.length}, 1fr)` `` | Weight sliders, 3-5 columns depending on Portfolio B's ticker count — same failure mode as Sidebar's own weight grid, just for the comparison portfolio. |
| [Sidebar.jsx:347](portfolio-risk-ui/src/components/Sidebar.jsx#L347) | `gridTemplateColumns: 'repeat(4, 1fr)'` | Period button row (1M/3M/6M/1Y). Inside a 220-400px-wide sidebar this is already routinely narrow *by design* and the labels are short (`1M`, `3M`...), so it degrades gracefully — genuinely low risk. |
| [Sidebar.jsx:378](portfolio-risk-ui/src/components/Sidebar.jsx#L378) | `gridTemplateColumns: '2fr 2fr 3fr'` | Custom-date day/month/year selects — same low-risk profile as above, already built for a narrow column. |
| [RiskAnalysis.jsx:223](portfolio-risk-ui/src/pages/RiskAnalysis.jsx#L223), [:318](portfolio-risk-ui/src/pages/RiskAnalysis.jsx#L318), [:327](portfolio-risk-ui/src/pages/RiskAnalysis.jsx#L327), [:343](portfolio-risk-ui/src/pages/RiskAnalysis.jsx#L343), [:359](portfolio-risk-ui/src/pages/RiskAnalysis.jsx#L359), [:297](portfolio-risk-ui/src/pages/RiskAnalysis.jsx#L297) | Six `1fr 1fr` / `repeat(2,1fr)` rows | 2-column layout throughout (rolling charts, VaR/CVaR + correlation, confidence headers). Least severe fixed-grid case in the file set — 2 columns is usually the *correct* mobile answer already, it just needs to become 1 column below some width rather than staying 2. |
| [ReturnHistogram.jsx:320](portfolio-risk-ui/src/components/ReturnHistogram.jsx#L320) | `gridTemplateColumns: 'repeat(4, 1fr)'` | Mean/Std-dev/Positive-days/Observations stat row under the histogram. |
| [AuthModal.jsx:35](portfolio-risk-ui/src/components/AuthModal.jsx#L35) | `width: 360` (fixed, no `max-width` safety) | Fits with only ~15px margin at 390/375px viewports; **overflows horizontally below ~360px** viewport width (common on Android). No responsive sizing at all on the sign-in modal. |

**A distinct, more fundamental finding, not a "hardcoded dimension" but the same family of bug:** three of the nine tab root containers have **no scroll fallback whatsoever** — Dashboard ([Dashboard.jsx:179](portfolio-risk-ui/src/pages/Dashboard.jsx#L179), `height:'100%'`, no `overflowY`), Monte Carlo ([MonteCarlo.jsx:111](portfolio-risk-ui/src/pages/MonteCarlo.jsx#L111), same), and Efficient Frontier ([Frontier.jsx:292](portfolio-risk-ui/src/pages/Frontier.jsx#L292), same) all omit `overflowY:'auto'`, while sitting inside `App.jsx`'s `<main style={{overflow:'hidden', ...}}>` ([App.jsx:254](portfolio-risk-ui/src/App.jsx#L254)). Every other tab (Risk Analysis, Valuation, Backtest, Compare, Learn) explicitly opts into `overflowY:'auto'` on its own root. On these three tabs, if the fixed-height rows above (metric grids, toggles) grow taller than expected at a narrow viewport, there is **no scrollbar to fall back to — content is silently clipped**, not just squeezed. This matters most for Dashboard, whose `1fr 220px` chart row is the single riskiest grid in the app (see above).

---

## 5. Canvas components

### RiskGauge.jsx

- **Sizing:** [RiskGauge.jsx:33-37](portfolio-risk-ui/src/components/RiskGauge.jsx#L33-L37) — `canvas.offsetWidth`/`offsetHeight` read once inside a `useEffect` keyed on `[score]` only (line 135). **Not** a `ResizeObserver`, no `window.resize` listener. CSS gives the canvas `width:'100%', aspectRatio:'1.4/1'` ([RiskGauge.jsx:157](portfolio-risk-ui/src/components/RiskGauge.jsx#L157)) so it visually rescales with its container, but the raster (`canvas.width/height`) and all drawing math are only recomputed when `score` changes — a pure container resize (viewport rotate, sidebar drag, drawer open/close) with an unchanged score leaves the gauge stretched/blurry until the next data refresh.
- **DPR handling:** yes — `canvas.width = W * window.devicePixelRatio`, `ctx.scale(dpr, dpr)` ([RiskGauge.jsx:35-37](portfolio-risk-ui/src/components/RiskGauge.jsx#L35-L37)).
- **Hardcoded drawing constants:** radius `R = min(W*0.44, H*0.82)` (proportional, not fixed px); arc `lineWidth = R*0.10`; zone-band opacity `0.18`; tick loop `for t=0; t<=100; t+=25` → exactly 5 markers; needle length `R*0.70`; pivot radii `R*0.06` / `R*0.03`; score label font `` `500 ${R*0.14}px ...` ``. Everything scales off `R` (itself off `W`/`H`), so the drawing itself is genuinely responsive — only the *redraw trigger* isn't.
- **Hit-testing:** none. `onMouseEnter`/`onMouseLeave` are bound to the wrapping `<div>`, not the `<canvas>`, and only toggle a border-color highlight ([RiskGauge.jsx:145-153](portfolio-risk-ui/src/components/RiskGauge.jsx#L145-L153)). No coordinate math, no tooltip.
- **Reads dimensions dynamically:** yes, but only at mount/score-change, not live.

### ReturnHistogram.jsx

- **Sizing:** [ReturnHistogram.jsx:32-36](portfolio-risk-ui/src/components/ReturnHistogram.jsx#L32-L36) — same pattern, `offsetWidth`/`offsetHeight` read in a `useEffect` keyed on `[portfolioReturns, varPct, cvarPct]` (line 190). No `ResizeObserver`. CSS: `width:'100%', height:200` (line 300) — a **fixed 200px height**, fluid width.
- **DPR handling:** yes, identical pattern to RiskGauge.
- **Hardcoded drawing constants:** `PAD = {top:16, right:16, bottom:36, left:36}` ([ReturnHistogram.jsx:38](portfolio-risk-ui/src/components/ReturnHistogram.jsx#L38)) — **fixed px, not proportional** (unlike RiskGauge's `R`-relative math); `N_BINS = 38` fixed bin count (line 42); axis/label fonts fixed at `9px` (lines 143, 162); `xTicks=6`, `yTicks=3`; bar top-cap height `1.5px`; line widths `1`/`1.5px`; hit tolerance `LINE_TOL = 5` (line 220).
- **Hit-testing:** yes. Binds `onMouseMove` + `onMouseLeave` on the `<canvas>` itself (lines 298-299). Coordinate conversion: `canvas.getBoundingClientRect()` then `e.clientX/clientY - rect.left/top` (lines 196-198) — explicit rect math, not `offsetX/offsetY`. Priority order: bottom/left axis-label strips first, then a `LINE_TOL=5px` window around the VaR/CVaR/Mean vertical lines, then a fallback continuous test against whichever histogram bar's `[x, x+width)` range contains the cursor (lines 200-226) — bars have no minimum tolerance beyond their own width, which shrinks as `N_BINS=38` divides a narrowing `PW`. No touch events bound anywhere.
- **Reads dimensions dynamically:** yes, at mount/prop-change only — same "correct at draw-time, stale between renders on pure resize" caveat as RiskGauge.

### Frontier.jsx scatter (the big one)

- **Sizing:** [Frontier.jsx:125-128](portfolio-risk-ui/src/pages/Frontier.jsx#L125-L128) — `canvas.offsetWidth`/`offsetHeight` read inside a `useEffect` keyed on `[data]` (line 258), i.e. redraws on every new analysis run, not on resize. It additionally pins `canvas.style.width/height` explicitly to the measured `W`/`H` in px (line 128) — so if the container resizes *after* this effect runs and *before* the next `data` change, the canvas's CSS box and its drawing buffer can visibly disagree (blur/stretch), the same class of bug as the other two canvases but slightly worse because it's actively overriding the CSS size rather than leaving it at `100%`.
- **DPR handling:** yes (`dpr = window.devicePixelRatio || 1`, lines 124-129).
- **Point count:** the raw "5,000 simulated portfolios" (label at [Frontier.jsx:305](portfolio-risk-ui/src/pages/Frontier.jsx#L305)) are **deduplicated before drawing**, not plotted 1:1. Lines 172-181 bucket every simulated `(vol, return)` pair into a grid at `CELL = 0.002` resolution, keeping only the single highest-Sharpe point per occupied cell (`if (!grid[key] || sharpes[i] > grid[key].s) grid[key] = ...`). `ptsRef.current` — the array actually drawn *and* hit-tested — is therefore some smaller number of unique grid cells, not 5,000.
- **Drawing constants:** fixed padding `PAD = {top:32, right:32, bottom:48, left:54}` (line 131, **not proportional to canvas size**); grid: 5 horizontal / 6 vertical lines (lines 158-159); axis font fixed `9px` (lines 162, 166); dot radius `1.6 + pct*1.2` (≈1.6-2.8px) plus a `+0.6px` outer glow ring (line 184); min-vol marker radius `5` (line 216); "Your portfolio" marker radius `8` core + pulsing rings `18→12` (lines 223-228); "Optimal" marker: diamond size `6`, crosshair size `5`, animated pulse ring `10±2` (lines 244-256); frontier-curve bucket size `max(0.002, volRange/60)` → up to ~60 buckets (line 173).
- **Hit-testing — the specific question:** [Frontier.jsx:275-287](portfolio-risk-ui/src/pages/Frontier.jsx#L275-L287). Binds `onMouseMove`/`onMouseLeave` on the `<canvas>` ([Frontier.jsx:326](portfolio-risk-ui/src/pages/Frontier.jsx#L326)). Coordinates via `getBoundingClientRect()` + `clientX/Y` delta (same pattern as ReturnHistogram). Two-tier pick:
  1. **Special markers first** — loop over `specRef.current` (only "Your Portfolio" and "Optimal", **not** min-vol), each carrying its own `radius: 12`; first one where `Math.sqrt((s.px-mx)**2+(s.py-my)**2) < s.radius` wins outright.
  2. **Otherwise, nearest-point search over the deduplicated cloud** — `let nearest = null, minDist = 12`, then every point in `ptsRef.current` is checked and the closest one *within* that 12px radius replaces `nearest`; if nothing is within 12px, the tooltip clears.
  So: it never "picks among 5,000" — it picks among the deduplicated grid-cell set, by nearest-Euclidean-distance-within-12px, with the two named markers getting a slightly larger 12px radius checked first (in practice the same 12px, just checked before the generic cloud).
- **What a ~44px fingertip does to this logic:** nothing good, for two independent reasons. First, **there is no touch path at all** — only `onMouseMove`/`onMouseLeave` are bound, no `onTouchStart`/`onTouchMove`/`onPointerMove`; most mobile browsers don't synthesize a continuous `mousemove` from a touch-drag, so the entire hover/tooltip system is simply inert on a touchscreen as written (the chart still renders, it just can't be queried). Second, *if* it were wired to touch input, the fixed 12px hit radius is already tuned for a ~1px mouse cursor with sub-pixel precision; a real fingertip contact patch is roughly **44px**, i.e. ~3.7× that radius, while the fixed `PAD` (32+32+54=118px eaten before any data plots) shrinks the usable plot width `PW` at narrow viewports, pushing point density *up* right as touch precision goes *down*. With potentially hundreds of deduplicated points inside a plot area that could be squeezed to ~150-200px wide on a phone, average inter-point spacing would routinely be well under 12px, let alone 44px — a single touch would sit ambiguously on top of many candidate points, the user's own finger would occlude the exact marker they meant to hit, and the current "closest single point wins" logic has no way to disambiguate or offer a larger dedicated target for the two named markers the way it already does versus the generic cloud.
- **Reads dimensions dynamically:** yes, at `data`-change time only (same class of staleness caveat as the other two canvases, compounded by the explicit `canvas.style.width/height` pin noted above).

---

## 6. Tables

| Table | Column count | Fixed or data-driven | Narrowest readable width (estimate) | Scroll vs. stack recommendation |
|---|---|---|---|---|
| **Correlation matrix** ([RiskAnalysis.jsx:130-181](portfolio-risk-ui/src/pages/RiskAnalysis.jsx#L130-L181)) | `n+1` (row-header + one column per ticker), `n` = 3-5 | **Data-driven**, and already has a home-grown density adaptation: `cellPad`/`cellFont`/`headFont`/`rowWidth` are all keyed off `n` ([RiskAnalysis.jsx:132-135](portfolio-risk-ui/src/pages/RiskAnalysis.jsx#L132-L135)) — 9px font / 32px row header at n≥5, up to 11px / 44px at n=3. | ~150-200px total at n=5 (32px header + 5×~30px cells) — genuinely narrow already. | **Neither, really — it doesn't need one.** It already reflows by *ticker count*, just not by viewport. The actual mobile problem is that it currently shares a fixed `1fr 1fr` row with the VaR/CVaR card ([RiskAnalysis.jsx:318](portfolio-risk-ui/src/pages/RiskAnalysis.jsx#L318)) — stack that outer grid to 1 column below the mobile breakpoint and the matrix itself will have all the room it needs without any table-specific scroll wrapper. |
| **Valuation table** ([Valuation.jsx:185-286](portfolio-risk-ui/src/pages/Valuation.jsx), read earlier this session) | 8 fixed columns (#, Ticker, P/S, EV/EBITDA, Gross Margin, Rev Growth, Mkt Cap, V/G Score) | **Fixed** — always all 8, never fewer; cell padding is a flat `'12px 14px'` regardless of ticker count. | ≥ 600-650px to stay legible at that padding (8 columns × ~70-90px). Far wider than any phone viewport. | **Horizontal scroll — and it's currently missing.** The table's card wrapper is `overflow:'hidden'` ([Valuation.jsx:196](portfolio-risk-ui/src/pages/Valuation.jsx#L196)), not `overflowX:'auto'` — at mobile width this **clips columns invisibly** rather than making them reachable by scroll. This is the one table in the set that needs an actual code change, not just a layout breakpoint. |
| **Backtest year-by-year** ([Backtest.jsx:66-113](portfolio-risk-ui/src/pages/Backtest.jsx#L66-L113)) | 4 fixed columns (Year, Your Portfolio, Equal Weight, Benchmark) | **Fixed** at exactly 4 — never per-ticker, always the 3 strategies + Year. | ~280-360px (4 columns × 70-100px) — borderline at a 375px phone, comfortably fine once scrollable. | **Already has horizontal scroll.** [Backtest.jsx:75](portfolio-risk-ui/src/pages/Backtest.jsx#L75): `<div style={{ overflowX: 'auto' }}><table>...` — this is the one table in the app that already does the right thing, and is the pattern to copy onto Valuation's table wrapper. |

---

## 7. Proven mobile patterns already in the codebase

**[MetricTooltip.jsx](portfolio-risk-ui/src/components/MetricTooltip.jsx) is the only place in the entire codebase that does real touch/hover-capability detection**, and it's a genuinely solid, reusable pattern. Quoted in full for reuse elsewhere:

Hover-capability check (module scope, computed once):
```js
// Touchscreens synthesize mouseenter/mouseleave around a tap alongside the
// click event — with both hover and click handlers wired to the same
// element, that synthetic pair can toggle `open` shut again right after the
// click opens it. Only bind the hover handlers on devices that actually
// support hover, so touch is left to the click-toggle alone.
const supportsHover = typeof window !== 'undefined' && window.matchMedia('(hover: hover)').matches
```

Conditional handler binding on the trigger element:
```jsx
<span
  ref={triggerRef}
  {...(supportsHover ? {
    onMouseEnter: () => setOpen(true),
    onMouseLeave: () => setOpen(false),
  } : {})}
  onClick={e => { e.stopPropagation(); setOpen(o => !o) }}
  style={{ ... cursor: 'help' ... }}
>
  {children}
</span>
```

Portal target + viewport-fixed positioning with clamping (the reusable part):
```js
const updatePosition = () => {
  if (!triggerRef.current) return
  const rect = triggerRef.current.getBoundingClientRect()
  const spaceBelow = window.innerHeight - rect.bottom
  const placement = spaceBelow < 100 ? 'top' : 'bottom'
  let left = rect.left + rect.width / 2 - POPOVER_WIDTH / 2
  left = Math.max(8, Math.min(left, window.innerWidth - POPOVER_WIDTH - 8))   // ← horizontal clamp to viewport
  setCoords({
    left,
    placement,
    y: placement === 'bottom' ? rect.bottom + GAP : window.innerHeight - rect.top + GAP,
  })
}
```
```jsx
{open && coords && createPortal(
  <div style={{
    position: 'fixed',
    [coords.placement === 'bottom' ? 'top' : 'bottom']: coords.y,
    left: coords.left,
    zIndex: 9999,
    width: POPOVER_WIDTH,
    /* ...surface-elevated card styling... */
  }}>
    {explanation}
  </div>,
  document.body   // ← portal target: escapes any ancestor `overflow:hidden`
)}
```
Plus outside-click-to-close and reposition-on-scroll/resize while open ([MetricTooltip.jsx:52-66](portfolio-risk-ui/src/components/MetricTooltip.jsx#L52-L66)):
```js
useEffect(() => {
  if (!open) return
  const handleOutside = e => {
    if (triggerRef.current && !triggerRef.current.contains(e.target)) setOpen(false)
  }
  document.addEventListener('click', handleOutside)
  window.addEventListener('scroll', updatePosition, true)
  window.addEventListener('resize', updatePosition)
  return () => { /* ...remove all three... */ }
}, [open])
```

Why it's the right template to copy: it (1) detects real hover-capability rather than screen width or user-agent sniffing, (2) falls back cleanly to tap-to-toggle without a second code path fighting the first, (3) portals to `document.body` specifically because `App.jsx`'s `<main>` is `overflow:'hidden'` and would silently clip a relatively-positioned popover, and (4) clamps its own horizontal position to the viewport rather than assuming it always has room. This last point is exactly what §8 shows the tour/callout components are missing.

**Every other place in the codebase that does any touch/pointer detection: none.** A full-tree grep for `matchMedia|ontouchstart|onTouchStart|onTouchMove|onPointerDown|onPointerMove|pointerType|maxTouchPoints|TouchEvent|hover:\s*hover|any-pointer|any-hover` returns exactly three hits total — `MetricTooltip.jsx`'s `(hover: hover)` above, and two unrelated `(prefers-reduced-motion: reduce)` checks ([lib/motion/scroll-motion.ts:53](portfolio-risk-ui/src/lib/motion/scroll-motion.ts#L53), [components/landing/Hero.jsx:26](portfolio-risk-ui/src/components/landing/Hero.jsx#L26)) that detect motion preference, not input type. Nothing else in the app knows whether it's being used with a finger or a cursor.

---

## 8. Tour and callout anchors

**[OnboardingTour.jsx](portfolio-risk-ui/src/components/OnboardingTour.jsx)** — 4 steps, each anchored via `document.querySelector(anchor).getBoundingClientRect()` ([OnboardingTour.jsx:26-27](portfolio-risk-ui/src/components/OnboardingTour.jsx#L26-L27)):

| Anchor selector | Lives in the sidebar? |
|---|---|
| `[data-tour="stocks"]` | **Yes** — [Sidebar.jsx:217](portfolio-risk-ui/src/components/Sidebar.jsx#L217), the "1. Stocks" `Section` |
| `[data-tour="weights"]` | **Yes** — [Sidebar.jsx:244](portfolio-risk-ui/src/components/Sidebar.jsx#L244), the "3. Weights" `Section` |
| `[data-tour="period"]` | **Yes** — [Sidebar.jsx:346](portfolio-risk-ui/src/components/Sidebar.jsx#L346), the "4. Period" `Section` |
| `[data-tour="run-analysis"]` | **Yes** — [Sidebar.jsx:424](portfolio-risk-ui/src/components/Sidebar.jsx#L424), the Run Analysis button |

**All four** onboarding-tour anchors live inside the sidebar. The component's own gate is `if (!visible || !rect) return null` ([OnboardingTour.jsx:44](portfolio-risk-ui/src/components/OnboardingTour.jsx#L44)) — if the sidebar becomes a closed-by-default off-canvas drawer on mobile, `querySelector` returns nothing for step 1 and the entire tour silently never appears (not a crash, just permanently invisible for first-time mobile visitors, unless the drawer is forced open during onboarding). Separately, the tour's own popover math (`left: rect.right + 16`, fixed `width: 236`, [OnboardingTour.jsx:47-53](portfolio-risk-ui/src/components/OnboardingTour.jsx#L47-L53)) assumes there is room to the right of the sidebar — if the drawer covers the full viewport width when open, `rect.right` is at or past the screen edge and the tour popover would render off-screen. **It has no horizontal viewport clamp**, unlike MetricTooltip (§7).

**[FirstResultCallout.jsx](portfolio-risk-ui/src/components/FirstResultCallout.jsx)** — 1 anchor: `[data-tour="sharpe-card"]`, which is **not** in the sidebar — it's the Sharpe Ratio `MetricCard` wrapper on Dashboard's top metrics row ([Dashboard.jsx:201-203](portfolio-risk-ui/src/pages/Dashboard.jsx#L201-L203)). Same positioning gap as OnboardingTour: `top: rect.bottom+10, left: rect.left` ([FirstResultCallout.jsx:40](portfolio-risk-ui/src/components/FirstResultCallout.jsx#L40)) with a fixed `width:250` and **no clamp against `window.innerWidth`** — at a narrow single-column mobile layout, a Sharpe card sitting anywhere past ~140px from the right edge would push this callout off-screen.

**Summary:** 4 of 5 total tour/callout anchors are inside the sidebar; only FirstResultCallout's is in main content. Both components lack the viewport-clamping that MetricTooltip already proves out — that clamp logic is the single most reusable fix for both.

---

## 9. Content density

**`InsightBox` usage per tab**, and its two size variants ([InsightBox.jsx](portfolio-risk-ui/src/components/InsightBox.jsx)): `compact` → `padding:'8px 14px'`, `marginTop:6`, label `marginBottom:3`; default → `padding:'12px 16px'`, `marginTop:8`, label `marginBottom:5`. Body text is always `--text-body-sm` (12px) at `line-height:1.6`.

| Tab | InsightBox count | Variant(s) | Rough vertical space per instance |
|---|---|---|---|
| Dashboard | 1 ("Return", [Dashboard.jsx:310](portfolio-risk-ui/src/pages/Dashboard.jsx#L310)) | compact | ~70-90px (short 1-2 line body) |
| Risk Analysis | 3 ("Why risk-adjusted metrics matter", "Why the daily returns distribution matters", "Why stress testing matters") | **all non-compact** | ~90-140px each — the returns-distribution one runs 4-5 sentences and is the longest InsightBox in the app |
| Monte Carlo | 2 ("Why three scenarios, not one", "Monte Carlo insight") | both compact | ~70-90px each |
| Efficient Frontier | 1 ("Why the mix matters") | compact | ~70-90px |
| Valuation | 1, mutually exclusive: "Why fundamentals, not just price" *or* "Not applicable" (all-fund case) | compact / non-compact respectively | ~70-90px / ~100-120px |
| Backtest | 1 ("Why run the backtest") | compact | ~70-90px |
| Compare | 2 — "Why the comparison matters" ([CompareWrapper.jsx:58](portfolio-risk-ui/src/components/CompareWrapper.jsx#L58)) + "Comparison summary" ([Comparison.jsx:208](portfolio-risk-ui/src/pages/Comparison.jsx#L208)) | compact + **non-compact** | ~70-90px + ~90-120px |
| Learn | 0 | — | — |

**Risk Analysis is by far the most InsightBox-dense tab** (3 instances, all full-size, easily 300-400px combined) — it's also the tab most likely to need real content triage (not just reflow) on mobile, separate from any layout fix.

**Dashboard above-the-fold at 390×844, sidebar as a drawer:** Available height ≈ 844 − 48px header/tab-bar ([App.jsx:169](portfolio-risk-ui/src/App.jsx#L169)) − ~32-40px footer ([App.jsx:317](portfolio-risk-ui/src/App.jsx#L317)) − 32px `<main>` padding (16px × 2, [App.jsx:254](portfolio-risk-ui/src/App.jsx#L254)) ≈ **~724px** of usable vertical space, ~358px usable width. Stacking Dashboard's fixed-height rows top to bottom: period pills (~28px, and note — this row has no `flexWrap` set either, [Dashboard.jsx:184](portfolio-risk-ui/src/pages/Dashboard.jsx#L184) — 4 pills in an un-wrapped flex row is itself a minor overflow risk) + top metrics row (5 cards squeezed to ~64px wide each, labels wrapping ⇒ ~100-130px tall) + second metrics row (2-5 `small` cards, long `sub` text like *"Well diversified · avg corr 0.14 · 1yr window"* wrapping 2-3 lines on a ~70px-wide card ⇒ ~90-110px) + SectorChart (~96-150px, or its own compact InsightBox ~70-90px for an all-ETF portfolio) ≈ **~300-420px** consumed before the charts row even starts. That leaves roughly 300-420px of height for the `flex:1` charts row — enough *height*, but its `1fr 220px` grid (§4) squeezes the primary growth chart to ~100-110px of *width*, well before vertical space runs out. **Net: Dashboard is very unlikely to actually clip vertically at this viewport** (the flex layout has enough room), but the growth chart — the single most important visual on the page — is squeezed to the point of being close to unreadable, and it would be squeezed *above the fold*, not below it. Given §4's finding that Dashboard has no `overflowY:'auto'` at all, if any of these fixed rows ever grow taller than this estimate (a longer diversification `sub` string, a 5th secondRowItem, wrapped ticker labels), there is no scroll fallback to catch the overflow.

**Benchmark selector:** native `<select>` ([Sidebar.jsx:413-416](portfolio-risk-ui/src/components/Sidebar.jsx#L413-L416)):
```jsx
<select value={benchmark ?? 'none'} onChange={e => setBenchmark(e.target.value === 'none' ? null : e.target.value)}>
  {BENCHMARKS.map(b => <option key={b.ticker} value={b.ticker}>{b.label} ({b.ticker})</option>)}
  <option value="none">None (no benchmark comparison)</option>
</select>
```
8 real benchmarks (SPY/QQQ/VTI/VXUS/GLD/SLV/USO/AGG, [lib/benchmarks.js](portfolio-risk-ui/src/lib/benchmarks.js)) + "None" = 9 `<option>` elements total, matching the "9-option" framing. **This is genuinely fine on mobile as-is** — it's a plain native `<select>`, which every mobile OS renders as its own full-screen picker UI; no custom dropdown component to rebuild. The same is true of every other `<select>` in the app (Risk Window, and the 6 custom-date day/month/year selects) — all native, none custom-built.

---

## 10. Landing pages and auth

- **`/`** → [Landing.jsx](portfolio-risk-ui/src/pages/Landing.jsx) — a thin (51-line) wrapper that renders a data-driven `landingPage` content config through a `BLOCK_RENDERERS` registry ([components/landing/registry](portfolio-risk-ui/src/components/landing/registry.jsx)) into components like `Hero`, `Cards`, `Features`, `Pillars`, `Pricing`, `StatsBand`, `Steps`, `ProductPlate`.
- **`/pro`** → [LandingPro.jsx](portfolio-risk-ui/src/pages/LandingPro.jsx) — same pattern, different content variant. There is also an unrouted `LandingRetail.jsx` (same 51-line shape) that isn't reachable from any current route in `main.jsx` — dead code, not a live page.
- **Sign-in:** [AuthModal.jsx](portfolio-risk-ui/src/components/AuthModal.jsx) — rendered as a `position:fixed` overlay from `App.jsx` (i.e. only reachable from inside `/app`, not a standalone route).

**Already more responsive than the app itself, in one specific way:** every one of the `components/landing/*` block renderers already uses CSS Grid's `repeat(auto-fit, minmax(Npx, 1fr))` — a viewport-responsive technique requiring **zero media queries** — confirmed in [Cards.jsx:42](portfolio-risk-ui/src/components/landing/Cards.jsx#L42), [Features.jsx:14](portfolio-risk-ui/src/components/landing/Features.jsx#L14), [Hero.jsx:130](portfolio-risk-ui/src/components/landing/Hero.jsx#L130), [Pillars.jsx:33](portfolio-risk-ui/src/components/landing/Pillars.jsx#L33), [Pricing.jsx:136](portfolio-risk-ui/src/components/landing/Pricing.jsx#L136), [StatsBand.jsx:12](portfolio-risk-ui/src/components/landing/StatsBand.jsx#L12), [LandingLayout.jsx:81](portfolio-risk-ui/src/components/LandingLayout.jsx#L81), [:107](portfolio-risk-ui/src/components/LandingLayout.jsx#L107). This is a real, working, already-proven-in-this-codebase alternative to the app's fixed `repeat(N, 1fr)` grids (§4) that the app tabs never use anywhere. `Nav.jsx` also uses `flexWrap:'wrap'` and a fluid `clamp(16px, 2vw, 34px)` gap on its own nav-item row ([Nav.jsx:99](portfolio-risk-ui/src/components/landing/Nav.jsx#L99)) rather than a dedicated hamburger/mobile-menu component — nav items reflow onto additional lines on narrow viewports; there is no mobile nav-toggle component anywhere in the codebase (confirmed by grep for "menu"/"hamburger", zero hits).
- **Sign-in is the one landing-adjacent surface that is genuinely desktop-only-assumption:** AuthModal's fixed `width: 360` with no `max-width` safety net (§4) is a real gap the auto-fit grid trick above doesn't cover, since it's a literal pixel width, not a grid track.

Per `DESIGN.md`'s own explicit scope note, this is expected to be a *separate* visual system from the app (Newsreader/IBM Plex, rounded corners, its own palette) — but "separate design system" and "separate responsiveness posture" turn out to be two different axes here: the landing page's responsiveness is actually *ahead* of the app's, not behind it.

---

## Risks

- **The app has never had to reflow anything, ever.** Zero media queries, zero `ResizeObserver` usage anywhere in `portfolio-risk-ui/src` (confirmed — ResizeObserver appears nowhere in the tree), and every one of the app's own grids is a literal `repeat(N, 1fr)` or a hardcoded `1fr 220px` track list. The landing page already knows the `repeat(auto-fit, minmax(...))` trick (§10); the app tabs have simply never needed it, which means introducing responsive breakpoints here is closer to a first attempt than a refinement.
- **Three tabs have no scroll fallback at all** (Dashboard, Monte Carlo, Efficient Frontier — §4/§9). Any mobile layout work on these three needs to either add `overflowY:'auto'` defensively or be extremely confident every fixed-height row actually fits, because right now there is no safety net if it doesn't.
- **The canvases don't redraw on resize** (§5) — RiskGauge, ReturnHistogram, and Frontier's scatter all measure their container exactly once per data-change, never on a pure container-resize/orientation-change. A sidebar-to-drawer transition, or a phone rotating from portrait to landscape, will leave all three canvases visually stale (wrong aspect, blurry, or — for Frontier, which explicitly pins `canvas.style.width/height` — actively mismatched) until the next `Run Analysis`. This is arguably a bigger risk than any single fixed-width grid, because it's invisible in a one-shot screenshot test and only shows up on an actual device being rotated or resized live.
- **Touch hit-testing doesn't exist for any canvas**, and Frontier's scatter is the sharpest version of that problem (§5) — a 12px mouse-tuned hit radius against a ~44px fingertip, on a point cloud that gets *denser* exactly as the plot area gets *narrower*. This one probably needs a real interaction redesign for touch (larger tap targets for the two named markers at minimum, likely a fundamentally different disclosure pattern — e.g. tap-to-select-then-detail rather than hover-to-preview) rather than a CSS-only fix.
- **DESIGN.md's constraints actively cut against several standard mobile patterns**, which is the part of this brief the visual system doesn't yet have an answer for:
  - *Zero border radius, no shadows* (Core Rules 1 & 3) means the usual "elevated sheet slides up from the bottom" mobile drawer/sheet convention — which almost always leans on a rounded top edge and/or a shadow to read as "lifted above the page" — has no vocabulary here. An off-canvas sidebar drawer can still work (a hard-edged panel sliding in over a scrim, which is what `StockDrawer` already does on desktop, per `CLAUDE.md`), but a bottom sheet in the iOS/Android convention doesn't fit the system as documented without either bending Core Rule 1/3 or finding a hairline-only way to signal "this is a temporary overlay, not part of the page."
  - *"Never more than one signal colour per component"* (DESIGN.md Signal table) is already flagged in-code as bent once, deliberately, for the Frontier scatter's 5-stop Sharpe gradient legend ([Frontier.jsx:309-311](portfolio-risk-ui/src/pages/Frontier.jsx#L309-L311): *"flagged exception to the single-signal-colour rule... needs a 5-stop ramp to communicate a continuous metric"*) — worth knowing a precedent for a deliberate, documented exception already exists, in case mobile-specific affordances (e.g. a two-tone "swipe" hint) need one too.
  - *One typeface for UI, negative tracking everywhere ≥18px, uppercase captions* (Type Scale) means the usual mobile trick of simply shrinking a font a step or two to fit more in has a hard floor — `--text-caption` (10px) is already the smallest role in the scale and is already tuned for legibility at that size; there is no smaller token to reach for, so mobile density has to come from *removing/collapsing* content (per §9's Risk Analysis finding) rather than shrinking it further.
  - Conversely, the system's *density-as-a-feature* stance ("Let cards sit tight, this is a data product," DESIGN.md Do's) is actually well-aligned with mobile once given the right column count — the existing `MetricCard`/`Section`/hairline-bordered-card vocabulary doesn't need reinventing for mobile, it needs a second, narrower column count to target, which is squarely a `gridTemplateColumns` problem (§4), not a visual-language problem.
