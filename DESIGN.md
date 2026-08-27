# Varense — Style Reference
> Warm brown terminal, sharp-edged, typographically compressed

**Theme:** dark

Varense operates as a precision instrument rendered in warm dark tones. The canvas is a deep brown-black rather than neutral black, giving the interface weight and warmth that pure black cannot. Every edge is sharp: zero border radius, everywhere, without exception. Structure comes entirely from 1px hairlines and disciplined spacing, never from shadows, gradients, or elevation tricks. A single typeface carries every role from 10px captions to 64px display, with aggressive negative tracking compressing headlines into confident blocks. Numbers are always monospace, always tabular. Color appears only as signal, never as decoration, and never more than one accent per element. The result reads as an engineered tool built by someone with taste, not a dashboard assembled from a component library.

---

## Core Rules

These are non-negotiable and define the system:

1. **Zero border radius.** Every element, every card, every button, every input, every tag. `border-radius: 0`. No exceptions.
2. **No gradients.** Anywhere. Not on backgrounds, not on charts, not on buttons, not as atmospheric bands.
3. **No shadows.** Depth comes from surface tone and hairline borders alone.
4. **One typeface for UI, one for numbers.** No third font.
5. **Color is signal.** If a color does not carry meaning, it should not be there.
6. **Hairlines do the structural work.** 1px borders define every boundary.

---

## Tokens — Colors

### Surfaces

| Name | Value | Token | Role |
|------|-------|-------|------|
| Canvas | `#1d1a18` | `--surface-canvas` | Page background. The warm brown-black everything sits on. |
| Elevated | `#262321` | `--surface-elevated` | Nested cards, input wells, inset panels. One step up from canvas. |
| Card | `#3d3a39` | `--surface-card` | Primary card and panel surface. |
| Raised | `#4d4947` | `--surface-raised` | Hover states on interactive surfaces, active tab backgrounds. |

### Lines

| Name | Value | Token | Role |
|------|-------|-------|------|
| Hairline | `#4d4947` | `--line-hairline` | Default 1px border on every card, divider, table cell, chart gridline. |
| Hairline Emphasis | `#5f5a57` | `--line-emphasis` | Focused inputs, active card edges, hovered rows. |
| Hairline Faint | `#2e2b29` | `--line-faint` | Chart gridlines, very low-emphasis separators inside dense data. |

### Text

| Name | Value | Token | Role |
|------|-------|-------|------|
| Primary | `#f2efec` | `--text-primary` | Headlines, metric values, primary labels. Warm off-white, never pure `#ffffff`. |
| Secondary | `#a39d98` | `--text-secondary` | Body copy, descriptions, supporting text. |
| Muted | `#7a736e` | `--text-muted` | Captions, axis labels, timestamps, placeholder text, inactive nav. |
| Faint | `#5a544f` | `--text-faint` | Disabled states, watermark-level text. |

### Signal

| Name | Value | Token | Role |
|------|-------|-------|------|
| Positive | `#52b788` | `--signal-positive` | Gains, favourable metrics, brand accent, active navigation. |
| Negative | `#e0574f` | `--signal-negative` | Losses, drawdown, VaR/CVaR, risk warnings. |
| Caution | `#d99a3c` | `--signal-caution` | Moderate risk, elevated volatility, neutral-but-watch states. |
| Positive Wash | `#52b7881a` | `--signal-positive-wash` | 10% opacity fill behind positive callouts. |
| Negative Wash | `#e0574f1a` | `--signal-negative-wash` | 10% opacity fill behind risk callouts. |
| Caution Wash | `#d99a3c1a` | `--signal-caution-wash` | 10% opacity fill behind caution callouts. |

**Rule:** never more than one signal colour per component. A card is positive, negative, or caution, never a mix.

---

## Tokens — Typography

### Primary — Geist

`--font-primary`

Geist is a neo-grotesque built for technical products. It has the engineered neutrality Swiss design demands without the ubiquity of Inter. Weight 400 carries body and most UI; 500 for labels and nav; 600 reserved for display headlines.

- **Substitute:** Satoshi, General Sans, Inter
- **Weights:** 400, 500, 600
- **Source:** free, `vercel/geist-font`

### Numeric — Geist Mono

`--font-mono`

Every number in the application is monospace with tabular figures. Prices, percentages, ratios, dates, tickers. This is what makes a data product feel like an instrument rather than a webpage.

- **Substitute:** JetBrains Mono, IBM Plex Mono
- **Weights:** 400, 500, 600
- **OpenType:** `"tnum" on, "zero" on`

### Type Scale

| Role | Size | Weight | Line Height | Tracking | Token |
|------|------|--------|-------------|----------|-------|
| caption | 10px | 500 | 1.4 | `0.08em` | `--text-caption` |
| micro | 11px | 400 | 1.45 | `0.04em` | `--text-micro` |
| body-sm | 12px | 400 | 1.5 | `0` | `--text-body-sm` |
| body | 14px | 400 | 1.55 | `-0.005em` | `--text-body` |
| heading-sm | 18px | 500 | 1.3 | `-0.02em` | `--text-heading-sm` |
| heading | 24px | 500 | 1.2 | `-0.035em` | `--text-heading` |
| heading-lg | 32px | 600 | 1.12 | `-0.045em` | `--text-heading-lg` |
| display | 48px | 600 | 1.04 | `-0.055em` | `--text-display` |
| display-lg | 64px | 600 | 1.0 | `-0.06em` | `--text-display-lg` |

**Caption uses positive tracking.** All-caps 10px labels need letter-spacing to breathe. Everything 18px and above uses negative tracking, and the larger the size the tighter it gets. This is the signature typographic move.

---

## Tokens — Spacing & Shape

**Base unit:** 4px
**Density:** compact (this is a data product, not a marketing page)

### Spacing Scale

| Token | Value |
|-------|-------|
| `--space-1` | 4px |
| `--space-2` | 8px |
| `--space-3` | 12px |
| `--space-4` | 16px |
| `--space-5` | 20px |
| `--space-6` | 24px |
| `--space-8` | 32px |
| `--space-10` | 40px |
| `--space-12` | 48px |
| `--space-16` | 64px |
| `--space-20` | 80px |

### Border Radius

| Element | Value |
|---------|-------|
| **everything** | **0px** |

There is one radius token and its value is zero. This is a defining constraint, not a default to override.

### Borders

| Token | Value |
|-------|-------|
| `--border-default` | `1px solid var(--line-hairline)` |
| `--border-emphasis` | `1px solid var(--line-emphasis)` |
| `--border-faint` | `1px solid var(--line-faint)` |
| `--border-signal` | `1px solid var(--signal-*)` |

### Layout

| Token | Value |
|-------|-------|
| `--page-max-width` | 1440px |
| `--sidebar-width` | 260px |
| `--section-gap` | 48px |
| `--card-padding` | 16px 20px |
| `--element-gap` | 12px |

---

## Components

### Card
Surface `--surface-card`, `--border-default`, 0 radius, padding `16px 20px`. No shadow. Header label in caption style (10px, 500, uppercase, `0.08em` tracking, `--text-muted`) with `12px` bottom margin. Cards sit directly adjacent with `1px` gaps, or `12px` apart, never both.

### Metric Card
Extends Card. Label in caption style. Value in `--font-mono`, `heading-lg` size, `--text-primary` by default or a signal colour when the metric carries direction. Optional sub-label in `micro` style, `--text-muted`. When a metric is in a warning or negative state, the card gets a `2px` top border in that signal colour, no fill, no glow.

### Button — Primary
`--signal-positive` background, `--surface-canvas` text, 0 radius, padding `10px 20px`, `body-sm` size weight 500, tracking `0.02em`, uppercase. No border, no shadow. Hover: brightness increases 8%, no movement, no scale.

### Button — Ghost
Transparent background, `--border-default`, `--text-primary` text, same padding and type as Primary. Hover: border becomes `--line-emphasis`, background becomes `--surface-elevated`.

### Button — Text
No border, no background, `--text-secondary`, `body-sm`. Hover: `--text-primary` with a 1px underline offset 4px.

### Input
`--surface-elevated` background, `--border-default`, 0 radius, padding `10px 12px`, `--font-mono` for anything numeric or ticker-based, `--font-primary` otherwise. Focus: border becomes `--signal-positive`, no glow, no ring.

### Tab Bar
Horizontal row, no background fill. Inactive tabs in `--text-muted`, `body-sm`, weight 500, uppercase, `0.06em` tracking. Active tab in `--text-primary` with a `2px` bottom border in `--signal-positive`. The row itself sits on a `1px` bottom hairline that the active indicator overlaps.

### Table
Full-width, `border-collapse: collapse`. Header row: caption style, `--text-muted`, `--border-default` bottom. Body rows: `--border-faint` bottom, `body-sm`, numeric cells in `--font-mono` right-aligned. Row hover: `--surface-elevated`. No zebra striping, no rounded container.

### Tooltip / Popover
`--surface-elevated` background, `--border-emphasis`, 0 radius, padding `10px 12px`, max-width `280px`. Body in `body-sm`, `--text-secondary`. Title in `caption` style, `--text-primary`. No arrow, no shadow, the border alone defines it.

### Callout / Insight Block
Signal wash background (`--signal-*-wash`), `3px` left border in the matching signal colour, no other borders, 0 radius, padding `12px 16px`. Label in caption style in the signal colour. Body in `body-sm`, `--text-secondary`.

### Sidebar
`--surface-canvas` background, `1px` right hairline. Section labels in caption style, `--text-muted`, `20px` top margin. Fixed `260px` width.

---

## Charts

Charts follow the same rules as everything else: sharp, flat, hairline-defined.

- **Gridlines:** `--line-faint`, 1px, solid. No dashed grids.
- **Axis labels:** `--font-mono`, 10px, `--text-muted`.
- **Axis lines:** hidden. Gridlines imply the axis.
- **Series colours:** `--signal-positive` for the user's portfolio, `--text-muted` for benchmarks, `--signal-caution` for comparison series, `--signal-negative` for drawdown.
- **Line weight:** 1.5px. No thicker.
- **Area fills:** flat colour at 8% opacity. **Never a gradient fill.**
- **Dots:** hidden by default, 3px square (not circle) on hover.
- **Tooltips:** use the Tooltip component spec above.
- **Chart containers:** Card spec, no radius, no inner padding beyond `12px`.

---

## Motion

Animation is permitted and encouraged on data, not on chrome. The rule: motion should reveal information, never decorate the interface.

| Element | Animation | Duration | Easing |
|---------|-----------|----------|--------|
| Chart line draw-on | Left-to-right path reveal | 700ms | `cubic-bezier(0.4, 0, 0.2, 1)` |
| Chart area fill | Fade in after line completes | 300ms | `ease-out` |
| Metric value | Count up from 0 to final value | 600ms | `ease-out` |
| Card entrance | Fade + 8px upward translate, staggered 40ms per card | 300ms | `ease-out` |
| Tab switch | Active indicator slides horizontally | 200ms | `cubic-bezier(0.4, 0, 0.2, 1)` |
| Tooltip | Fade only, no scale, no slide | 120ms | `ease-out` |
| Drawer | Slide from right | 220ms | `cubic-bezier(0.4, 0, 0.2, 1)` |

**Never animate:** hover scale, button press bounce, rotating icons, pulsing glows, looping ambient motion. Respect `prefers-reduced-motion` by disabling all of the above except opacity fades.

---

## Do's and Don'ts

### Do
- Set every `border-radius` to 0. This is the single most defining rule of the system.
- Use `--font-mono` with tabular figures for every number without exception, prices, percentages, ratios, dates, tickers.
- Apply negative tracking to everything 18px and up, tightening as size increases. This compression is the typographic signature.
- Use warm off-white `#f2efec` for primary text, never pure white, it will look cold and detached against the brown canvas.
- Define every boundary with a 1px hairline. Structure comes from lines, not shadows.
- Restrict signal colours to elements where direction or risk has actual meaning.
- Use uppercase with positive tracking (`0.08em`) for all 10px caption labels.
- Let cards sit tight, this is a data product and density is a feature.

### Don't
- Don't introduce border radius anywhere, including "just slightly rounded" 2px or 4px values.
- Don't use gradients, on backgrounds, charts, buttons, or as decorative bands.
- Don't add box-shadows for depth. Use `--surface-elevated` or a hairline instead.
- Don't use pure `#000000` or pure `#ffffff` anywhere. The warm palette is the point.
- Don't apply more than one signal colour to a single component.
- Don't animate chrome, hover scales, bouncing buttons, and pulsing glows all break the instrument feel.
- Don't use a third typeface. Geist and Geist Mono cover every role.
- Don't use circular dots or rounded caps on chart elements, square dots, butt line caps.
- Don't set positive letter-spacing on anything above 12px.

---

## Agent Prompt Guide

**Quick Reference**
- canvas: `#1d1a18` · elevated: `#262321` · card: `#3d3a39`
- hairline: `#4d4947` · faint line: `#2e2b29`
- text: `#f2efec` primary · `#a39d98` secondary · `#7a736e` muted
- positive: `#52b788` · negative: `#e0574f` · caution: `#d99a3c`
- radius: `0` everywhere · borders: `1px solid` · shadows: none
- fonts: Geist (UI), Geist Mono (all numbers)

**Example Component Prompts**

1. **Metric Card**: `#3d3a39` background, `1px solid #4d4947` border, 0 radius, padding 16px 20px. Label "SHARPE RATIO" at 10px weight 500 uppercase, letter-spacing 0.08em, colour `#7a736e`. Value "2.52" in Geist Mono 32px weight 600, letter-spacing -0.045em, colour `#f2efec`. No shadow.

2. **Risk Metric Card (negative state)**: Same as Metric Card plus a `2px solid #e0574f` top border. Value colour becomes `#e0574f`. No fill tint, no glow.

3. **Tab Bar**: Horizontal row on a `1px solid #4d4947` bottom border. Inactive tab: 12px weight 500 uppercase, letter-spacing 0.06em, colour `#7a736e`, padding 12px 16px. Active tab: colour `#f2efec` with a `2px solid #52b788` bottom border overlapping the row hairline. Indicator slides 200ms on change.

4. **Chart**: Recharts LineChart, no radius on container. Gridlines `#2e2b29` 1px solid horizontal only. Axis text Geist Mono 10px `#7a736e`, axis lines hidden. Portfolio series `#52b788` 1.5px, benchmark series `#7a736e` 1.5px. Area fill flat `#52b788` at 8% opacity, never a gradient. Dots hidden, 3px squares on hover. Line draws left-to-right over 700ms on mount.

5. **Insight Callout**: Background `#52b7881a`, `3px solid #52b788` left border, no other borders, 0 radius, padding 12px 16px. Label "WHY THIS MATTERS" at 10px weight 500 uppercase letter-spacing 0.08em colour `#52b788`. Body at 12px weight 400 line-height 1.5 colour `#a39d98`.

---

## Quick Start

```css
:root {
  /* Surfaces */
  --surface-canvas:    #1d1a18;
  --surface-elevated:  #262321;
  --surface-card:      #3d3a39;
  --surface-raised:    #4d4947;

  /* Lines */
  --line-hairline:     #4d4947;
  --line-emphasis:     #5f5a57;
  --line-faint:        #2e2b29;

  /* Text */
  --text-primary:      #f2efec;
  --text-secondary:    #a39d98;
  --text-muted:        #7a736e;
  --text-faint:        #5a544f;

  /* Signal */
  --signal-positive:       #52b788;
  --signal-negative:       #e0574f;
  --signal-caution:        #d99a3c;
  --signal-positive-wash:  #52b7881a;
  --signal-negative-wash:  #e0574f1a;
  --signal-caution-wash:   #d99a3c1a;

  /* Fonts */
  --font-primary: 'Geist', ui-sans-serif, system-ui, -apple-system, sans-serif;
  --font-mono:    'Geist Mono', ui-monospace, 'SF Mono', Menlo, monospace;

  /* Type scale */
  --text-caption:      10px;  --tracking-caption:      0.08em;
  --text-micro:        11px;  --tracking-micro:        0.04em;
  --text-body-sm:      12px;  --tracking-body-sm:      0;
  --text-body:         14px;  --tracking-body:        -0.005em;
  --text-heading-sm:   18px;  --tracking-heading-sm:  -0.02em;
  --text-heading:      24px;  --tracking-heading:     -0.035em;
  --text-heading-lg:   32px;  --tracking-heading-lg:  -0.045em;
  --text-display:      48px;  --tracking-display:     -0.055em;
  --text-display-lg:   64px;  --tracking-display-lg:  -0.06em;

  /* Weights */
  --weight-regular:  400;
  --weight-medium:   500;
  --weight-semibold: 600;

  /* Spacing */
  --space-1: 4px;   --space-2: 8px;   --space-3: 12px;
  --space-4: 16px;  --space-5: 20px;  --space-6: 24px;
  --space-8: 32px;  --space-10: 40px; --space-12: 48px;
  --space-16: 64px; --space-20: 80px;

  /* Shape */
  --radius: 0;
  --border-default:  1px solid var(--line-hairline);
  --border-emphasis: 1px solid var(--line-emphasis);
  --border-faint:    1px solid var(--line-faint);

  /* Layout */
  --page-max-width: 1440px;
  --sidebar-width:  260px;
  --section-gap:    48px;
  --card-padding:   16px 20px;
  --element-gap:    12px;

  /* Motion */
  --ease-standard: cubic-bezier(0.4, 0, 0.2, 1);
  --duration-fast:     120ms;
  --duration-standard: 220ms;
  --duration-slow:     300ms;
  --duration-draw:     700ms;
}
```
