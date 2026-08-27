# THEMING.md — how to retheme Varense

Every color, font, spacing, and radius value in the app routes through a CSS
custom property defined once in `portfolio-risk-ui/src/index.css`'s `:root`
block. This is what makes "change the palette" a token edit in one file
instead of a hunt across ~30 component files. This doc lists exactly which
variables to touch for common retheme requests.

Full token reference and the rules behind them: `DESIGN.md` at the project
root. This file is the "where do I actually go to change X" index.

---

## To change the accent brown to a different hue

Edit these together — they're the warm-brown family and should move as a set:

```
--surface-canvas    (page background)
--surface-sidebar    (sidebar background — one step above canvas)
--surface-card       (card/panel background)
--surface-elevated   (hover states, tooltips, input wells)
--vignette-center    (canvas vignette — keep it a lighter version of --surface-canvas)
--vignette-edge      (canvas vignette — keep it a darker version of --surface-canvas)
```

Keep the same *relative* lightness relationship between these five (canvas
darkest → sidebar → card → elevated lightest, vignette center/edge bracketing
canvas) or the depth hierarchy the whole UI relies on disappears.

## To change the hairline/border color

```
--line-hairline       (default 1px border — card sides/bottom, dividers, table cells)
--line-hairline-top    (card top edge only — kept slightly lighter than --line-hairline
                         on purpose, simulates light catching a physical edge)
--line-emphasis        (focused inputs, hovered rows, active edges)
--line-faint            (chart gridlines, low-emphasis separators)
```

## To change the signal colors (positive/negative/caution)

```
--signal-positive, --signal-positive-rgb, --signal-positive-wash
--signal-negative, --signal-negative-rgb, --signal-negative-wash
--signal-caution,  --signal-caution-rgb,  --signal-caution-wash
```

Each color has three forms: the solid hex, an `R, G, B` triple (for
`rgba(var(--signal-positive-rgb), 0.15)`-style washes), and a pre-mixed 10%
wash. When you change the hex, recompute the `-rgb` triple to match — nothing
derives it automatically — and the wash tokens if you want their alpha level
to stay the same 10%.

**Where signal colors show up beyond cards:** chart line series (portfolio
line = positive, benchmark = muted, comparison series = caution, drawdown =
negative), the vignette-adjacent `--text-on-accent`/`--surface-canvas` pairing
used for dark text sitting on a positive-filled button.

## To change the typeface

```
--font-primary   (UI text — headings, labels, body)
--font-mono      (every number — prices, percentages, ratios, dates, tickers)
```

Both are loaded globally via `<link>` tags in `portfolio-risk-ui/index.html`
(currently Geist Sans / Geist Mono from the jsDelivr Fontsource CDN). Swap the
font by (1) changing those `<link>` tags to the new family's CDN URLs and (2)
updating the two `--font-*` values to reference the new family name as the
first item in the stack — keep the existing fallback stack after it.

## To change the border radius

There is one value: `--radius: 0`, enforced globally by a single rule near
the top of `index.css`:

```css
*, *::before, *::after { border-radius: 0 !important; }
```

This is intentionally `!important` — it's the single source of truth so
individual components never need their own zero-radius override. If Varense
ever grows actual rounded corners, remove the `!important` and the universal
selector, set `--radius` to the new value, and route `border-radius:
var(--radius)` through components deliberately, rather than reintroducing
one-off `borderRadius: 'Npx'` calls per component.

## To change spacing

```
--space-1 through --space-20   (4px base unit, see DESIGN.md for the full scale)
--card-padding, --element-gap, --section-gap, --sidebar-width, --page-max-width
```

## To change the type scale (sizes, tracking, weights)

Nine roles, each bundling size + tracking + weight + line-height as a set:

```
--text-caption / --tracking-caption / --weight-caption / --leading-caption
--text-micro ... --text-body-sm ... --text-body ... --text-heading-sm ...
--text-heading ... --text-heading-lg ... --text-display ... --text-display-lg
```

Components reference these by role (e.g. `fontSize: 'var(--text-caption)'`),
never a bare pixel number, so retuning the whole type scale is nine variable
edits, not a search-and-replace across every component.

## To change the vignette or grain texture

```
--vignette-center, --vignette-edge, --vignette-shape, --vignette-position, --vignette-stop
--grain-opacity-canvas, --grain-opacity-surface, --grain-tile-size
```

The vignette gradient string and the grain SVG data-URI both live in
`index.css`'s `.vignette-layer` / `.grain-canvas` / `.grain-surface` /
`.card` rules — the five vignette tokens above are exactly what those rules
interpolate, so retuning the vignette is a variable edit there, not a string
hunt through the gradient itself.

---

## Known exceptions — not driven by the token system above

A handful of places intentionally don't route through the standard 3-signal
palette, because the component structurally needs more than 3 colors to
convey the information it shows. Each is a real, deliberate exception, not
an oversight:

- **`Frontier.jsx`** — the efficient-frontier scatter plot's Sharpe-ratio
  heat-map (`sharpeColor()`, a 5-stop red→teal ramp) and its legend swatch
  (a real CSS `linear-gradient`, DESIGN.md's only other permitted gradient
  besides the canvas vignette). Collapsing this to one signal color would
  make all 5,000 simulated portfolios visually indistinguishable — the whole
  point of the chart.
- **`RiskGauge.jsx`** — the risk-o-meter's 4-zone arc gradient reuses the
  legacy `--chart-highlight` token for its "Moderate" stop, since the
  3-signal palette doesn't define a 4th tier between positive and caution.
- **`MonteCarlo.jsx`** — the "Bull" scenario / 95th-percentile line uses the
  legacy `--accent-light` token as a 4th chart color (Bear/Base/Bull each
  need a distinct hue; signal-negative and signal-positive are already
  spoken for by Bear and Base).
- **`ExportPDF.jsx`**'s printable report — deliberately stays on its own
  separate light `--report-*` token set (`--report-bg`, `--report-text-dark`,
  etc., defined in both `index.css` and duplicated into the popup document's
  own injected `<style>` tag, since that popup is a separate `document` that
  doesn't inherit `index.css` at all). This is a print artifact meant to
  read on white paper, not a dark-mode UI surface — retheme it separately
  from the app if you ever want to, it won't affect anything above.
- **Google's "G" logo** in `AuthModal.jsx` keeps its 4 official brand hex
  colors (`#4285F4` etc.) — those are fixed by Google's brand guidelines,
  not a themeable app color.

If you want any of these collapsed into the 3-signal system, that's a
content/information-density decision (do you lose the ability to tell
"Moderate" from "Elevated" risk, or a 5,000-dot cloud from a flat color),
not a mechanical retheme — flag it explicitly rather than trying to fix it
via a token edit.
