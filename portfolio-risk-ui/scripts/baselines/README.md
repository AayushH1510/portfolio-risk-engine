# Desktop baselines

"Known good" screenshots, checked automatically by `npm run shots` (no
separate flag needed — see `shots.mjs`'s own top comment and its
`BASELINE_*` constants for the full rationale and the noise floor's
derivation).

```
desktop/<1280|1366|1440|1366x650|1536x730|1920x970>/<view>.png
```

13 views × 6 viewports = 78 files: every landing/legal route (`home`, `pro`,
`methodology`, `privacy`, `terms`) and every app tab (`app-dashboard`,
`app-risk`, `app-montecarlo`, `app-frontier`, `app-valuation`,
`app-compare`, `app-backtest`, `app-learn`). Three of the six viewports
(`1280`, `1366`, `1440`) are full device heights with no browser chrome;
the other three (`1366x650`, `1536x730`, `1920x970`) subtract realistic
chrome — see `shots.mjs`'s `DESKTOP_CHROME_VIEWPORTS` for exactly why both
kinds are needed, not just the chrome-free ones.

**Why this exists:** three real regressions shipped clean through every
other check this harness runs (overflow measurement, scroll-reachability,
tab-switch verification, touch taps) — a universal padding that shrank
every app tab's content, an avatar that rendered when it shouldn't have,
and a compact-viewport media query keyed to window *height* that fired on
any laptop window under 768px tall (common, once real browser chrome and
display scaling are in the picture — see `lib/breakpoints.js`'s own
postmortem on `COMPACT_VIEWPORT_QUERY`). None of the three moved a number
any existing check was looking at; they only ever looked wrong. This is
the check for "looks different," independent of whatever metric a future
regression happens to avoid moving — and the chrome-realistic viewports
specifically exist because the first two regressions were caught by
pixel-diffing, while the third slipped past pixel-diffing too, since every
desktop viewport being diffed was a full, chrome-free device height no
real browser window has.

## Re-captured 2026-09-27 — the previous set had locked in a regression

The baseline set was fully regenerated on this date because the *previous*
one wasn't actually "known good" — it was captured after desktop had
already drifted from its intended layout, so the check was comparing new
renders against an already-wrong reference and passing. Three concrete
desktop bugs shipped underneath a clean `npm run shots`:

- Metric-card-style content aside, the **feedback icon crowded the header
  tab bar** enough that "Compare" and every tab after it were reachable
  only by touch-scrolling `nav.tab-bar-scroll` (a hidden-scrollbar,
  touch-only affordance — a mouse wheel over it scrolls the page, not the
  nav) — no visible sign on desktop that the row scrolled at all. Fixed by
  moving the trigger to a footer link on desktop (`.footer-feedback-link`)
  and, since that alone didn't fully close the gap at 1366px, a modest
  tab-button padding reduction (`.app-tab-btn`, 10px → 6px a side,
  desktop-only — phone and landscape keep the original 10px).
- The **Drawdown chart could render at zero height** on a real laptop
  *window* (chrome included) short enough to starve `.dashboard-grid`'s
  `1fr` final row — confirmed present identically on the pre-Phase-B
  reference commit (`db3b8f7e`) at the same viewports, so this was never a
  Phase B regression, just a gap nobody had a short-enough test viewport
  to expose before. Fixed with a real `min-height` on
  `.dashboard-right-col__drawdown` (see that rule's own comment for the
  190px derivation).
- Restored the desktop tab-content layout to match `db3b8f7e` (the last
  commit before any tab-layout work) wherever it had drifted, while
  keeping every deliberate improvement made since (chart fullscreen-expand
  buttons, InsightBox collapse toggles, the signed-in-only avatar fix,
  hidden zero-weight sectors, Compare's `btn-primary` width fix).

Every PNG under `desktop/` reflects the corrected layout as of this date.
If `npm run shots` starts failing against this set, treat it the same way
this file's "If a check fails" section below always has — the point of
today's recapture wasn't to lower the bar, it was to make sure the bar was
actually measuring the right thing again.

## Re-captured 2026-09-27, again — 1280px nav fit + mouse-wheel reachability

A same-day follow-up to the recapture above, not a second regression —
this one is a deliberate header change, not a fix for drift the baselines
missed. The previous recapture closed the nav-overflow gap at 1366px
exactly (742/742px) but explicitly left 1280px short (147px → 83px,
flagged at the time as out of scope) and left mouse users with no way to
reach an overflowed tab at all if the row ever did overflow (the nav's
horizontal scroll only ever worked by touch-drag — a hidden scrollbar with
no other affordance). Two changes:

- **Export PDF and Export CSV combined into one "Export" menu**
  (`ExportMenu.jsx`) — a single trigger that opens a small, keyboard-
  accessible, zero-radius/hairline-border menu with both options, applied
  at every desktop width (not gated behind its own breakpoint, so the
  header reads the same way at 1280px as at 1920px). This alone recovers
  enough header width that **1280px now fits all eight tabs with zero
  nav overflow** — confirmed by direct measurement
  (`scrollWidth === clientWidth`), not just visually. `ExportPDF.jsx` and
  `ExportCSV.jsx` keep their original full-button exports for the sidebar
  drawer (`.sidebar-export-actions`, phone/tablet) — that surface was
  never the width-constrained one, so there was no reason to collapse it
  there too. Both files now also export a plain `runExportPDF`/
  `runExportCSV` function (the PDF-popup / CSV-download logic itself,
  unchanged) so the menu and the standalone buttons share one
  implementation instead of two.
- **Mouse-wheel-to-horizontal-scroll on the tab bar, `(pointer: fine)`
  only** — a vertical wheel gesture over `nav.tab-bar-scroll` now scrolls
  the row itself instead of the page, and the row gets a real, thin,
  visible scrollbar whenever it overflows (`::-webkit-scrollbar`/
  `scrollbar-width: thin`, gated the same way) instead of the permanently
  hidden one every other viewport still correctly keeps. This is the
  reachability *guarantee* the fit fix above doesn't cover on its own —
  verified by deliberately shrinking to a width that still overflows even
  after the Export-menu fix (1024px) and confirming a wheel gesture moves
  `scrollLeft`. Touch is completely untouched: `(pointer: fine)` doesn't
  match a touchscreen, so phone and 852×393 landscape keep the exact same
  hidden-scrollbar, drag-to-scroll behaviour they had before this change,
  confirmed both by computed-style assertions and by screenshot.
- Also verified at 1536×730 with the browser zoomed to 125% (an effective
  ~1229×584 viewport — how a real 1536px laptop reports itself under
  Windows display scaling): still zero nav overflow, no different from
  the un-zoomed case.

A real bug was caught and fixed *before* it ever reached this baseline
set, not shipped and found later: the menu's initial "click outside to
close" check only tested against the trigger button, so clicking either
menu item was itself misclassified as an outside click and closed the
menu on `mousedown`, before the item's own `onClick` (and therefore
`runExportPDF`/`runExportCSV`) ever ran — worth naming here since it's
exactly the shape of bug pixel-diffing can't catch (the menu still closed,
which looked correct), only interaction testing did.

Every PNG under `desktop/` reflects the combined-Export-button header as
of this date.

## Re-captured 2026-09-27, a third time — chart fullscreen-expand enabled on desktop

Another same-day deliberate header change, not a regression fix. The
chart-expand-to-fullscreen control (`useOverlay`, `.chart-expand-btn`,
`.chart-fullscreen-expanded`, portalled to `document.body`) previously
rendered only on a compact viewport — phone or landscape-phone — gated by
`isCompactViewport` in each of its six consumers (Dashboard's Portfolio
Growth, Backtest's cumulative-return chart, Monte Carlo's simulated-futures
chart, Efficient Frontier's scatter, and both of Compare's charts). It's now
shown at every width: the `isCompactViewport &&` gate around the button
itself is gone (`.chart-expand-btn`'s own hairline border was already
visible at rest, not hover-only, so nothing new was needed there), and each
`isXExpanded` state is now a plain `expandRequested` boolean instead of
`expandRequested && isCompactViewport` — desktop can now genuinely reach
the expanded state, not just render a button that could never do anything.
No new mechanism: same `useOverlay`, same CSS classes, same portal, reused
exactly as they already worked on phone.

One behavioural addition alongside it: the four Recharts line/area charts
among the six (Dashboard, Backtest, Monte Carlo, Compare's two) now apply
`computeYDomain`'s tight Y-axis domain whenever `isXExpanded` is true,
regardless of viewport — previously that tight domain only applied at
compact viewport (Dashboard, Compare's Monte Carlo chart) or not at all
(Backtest, Compare's Monte Carlo... Compare's own Cumulative Returns chart
never had one). Without this, a chart expanded to fill a 1920×970 viewport
would still use Recharts' auto-scaled "nice number" domain, sized for a
~65-200px inline plot — the line would sit in a thin band in the middle of
a much taller box instead of filling it, the same legibility problem this
project already fixed for the inline compact-viewport case. The *inline*
chart's domain is untouched at every width, desktop and phone alike — the
domain prop is only added or its condition widened for the *expanded*
state; verified directly (Y-axis tick labels read identically inline before
and after this change, and visibly tighter only once expanded). Efficient
Frontier's scatter is canvas-based with its own axis-drawing logic, not
Recharts — no domain change applies there, only the button.

Every affected chart's header row grew slightly once the button became
unconditional there (previously absent from desktop's layout entirely) —
this is what the baseline diff below actually is, not a broken chart: a
solid-color button appearing in five headers pushes each header row a few
pixels taller, which shifts the chart plot area beneath it by that same
few pixels, and pixel-diffing correctly (if noisily) flags nearly every
pixel along a shifted line or scatter cloud as "different" even though the
underlying data and geometry are identical. Confirmed by inspecting the
diff images directly, not just trusting the percentage: `app-dashboard`,
`app-backtest`, `app-montecarlo`, `app-compare`, and `app-frontier` all
diffed in the 0.5%-3.9% range at every desktop baseline width — `app-risk`,
`app-valuation`, `app-learn`, and every landing route stayed at 0% (or
within the existing floor), confirming the change is isolated to exactly
the five card headers this pass touched and nothing else. Efficient
Frontier's own diff was the largest (~3.3-3.9%) because its canvas's
Y-axis gridlines are computed from the canvas's own measured height
(`useCanvasSize`, scaled padding — see that file's own comments); a few
fewer pixels of height from the taller header nudged its "nice round
number" gridline search to a different top tick (confirmed by comparing
the two screenshots directly: same cloud shape, same marker positions,
different axis label) — a real, deterministic, and expected consequence of
the header change, not noise or a bug.

Verified at 1280×720, 1366×650, 1536×730, and 1920×970: all six expand
buttons visible and clickable, each expands to fill the exact viewport
(`0,0,width,height`), a real chart (SVG or canvas) renders inside every
one, and Escape closes all six back to the inline card. Phone (384×854)
and 852×393 landscape with touch emulation confirmed unchanged — the
button still works by tap exactly as before, and the inline chart's own
Y-axis ticks are byte-identical to what they were before this pass (still
gated on `isCompactViewport`, now just also true when expanded).

Every PNG under `desktop/` reflects the six always-visible expand buttons
as of this date.

## Re-captured 2026-09-27, a fourth time — font-display fix changed landing-page text layout

A fourth same-day recapture, and — like the previous three — a deliberate
frontend change, not a regression: two pre-launch landing-page performance
fixes (code-splitting `/app` out of the bundle `/` no longer needs to
download, and fixing the font-swap-driven layout shift Lighthouse's CLS
audit flagged on `/`). Neither one was expected to touch a single rendered
pixel of any desktop screenshot — the `/app` code-splitting only changes
*when* JS is fetched, not what anything looks like, and the CLS fix only
changes *which* font renders under a specific timing condition. It was the
second one that did, and is worth recording exactly why.

**The fix:** `scripts/fix-font-display.mjs` (new, runs in prebuild/predev
alongside the existing `tokens-to-css.mjs`) rewrites every landing
`@fontsource` face's `font-display: swap` to `font-display: optional`.
`swap` paints the fallback font immediately and then reflows the page once
the real webfont finishes loading — a page-wide layout shift happening
*after* first paint, which is exactly what CLS measures, and exactly what
was showing up as 0.162 on `/`. `optional` gives the browser a short (~100ms)
window to use the font if it's already available (cached from an earlier
visit) and otherwise commits to the fallback font for that page view, with
no later swap and therefore no later reflow — confirmed directly, not just
by the audit's own number: a `PerformanceObserver` for `layout-shift`
entries measured 0 shift, 0 entries, at every one of four different
conditions (normal load, fonts artificially delayed, canvas disabled,
reduced-motion) after this fix, against a real, reproducible shift before
it.

**Why `home` and `methodology` specifically changed height, and nothing
else did:** `scripts/shots.mjs` always launches a fresh browser context
with no font cache — exactly the "first visit" condition `optional` is
designed around, so every capture of a landing route now renders body text
in the *fallback* font, not Newsreader/Lora, unless the page has no real
body copy set in those families to begin with. `home` and `methodology`
are the two routes with substantial Newsreader/Lora prose; `pro`/`privacy`/
`terms` show 0.0000% precisely because they don't lean on those families
enough for a fallback-vs-real-font metrics difference to move a pixel. The
fallback font's line-height and character widths aren't identical to
Newsreader/Lora's, so the same text wraps and stacks slightly differently —
`methodology` (the most text-heavy route) shortened by ~300-400px out of
~20,000px total (under 2%); `home` moved by tens of px in either direction
depending on width. Checked directly against the previous baseline, not
assumed from the numbers alone: both pages render fully, correctly, with no
overlapping or cut-off text at any width — this is a real but cosmetically
minor font substitution, not a layout break.

This is also the more representative capture for what "before a public
launch" actually means: a first-time visitor with no cached fonts is
precisely the case these two fixes were aimed at, more so than a returning
visitor whose browser already has Newsreader/Lora cached and would see the
real fonts immediately (still within `optional`'s ~100ms window, still no
reflow) on every page after their first.

Every PNG under `desktop/` for `home` and `methodology` reflects the
fallback-font-rendered, cold-cache layout as of this date; every other
route's PNGs are unchanged (confirmed 0.0000%/within-floor across the
board) since neither fix touches anything else on the page.

## Re-captured 2026-09-28 — Compare tab's WIN-badge direction bug fixed

Deliberate change, not drift: fixed a correctness bug where Drawdown, VaR
95%, and CVaR 95% (all stored as negative numbers — a loss) awarded WIN to
the *worse* side. `Comparison.jsx`'s old `winner()` applied a bare
`lowerBetter` boolean straight to the raw signed value — for these three
metrics that meant "more negative" (a bigger loss) read as "lower" and
therefore "better," the opposite of correct (e.g. a -6.6% drawdown used to
beat a -4.7% one). Replaced with `lib/compareMetrics.js`, a single explicit
per-metric direction table every consumer (the win-count array and each
`MetricRow`) now reads from — see that file's own header comment.

That same fix changed how an exact **tie** resolves: the old `winner()` had
no tie case at all (`aVal < bVal ? 'A' : 'B'` and `aVal > bVal ? 'A' : 'B'`
both evaluate to `'B'` when the two values are equal), so it silently
declared Portfolio B the winner of every metric whenever A and B scored
identically. The fix returns `null` (no winner, no WIN badge, no
highlight) on a genuine tie instead. This harness's own fixture feeds
`/api/analyse`, `/api/analyse-summary`, and `/api/analyse-full` the exact
same captured data for both Portfolio A and Portfolio B — so every metric
in `app-compare`'s screenshot ties, and the old baseline showed Portfolio B
winning all 6 metrics it counted, with a "Portfolio B wins on 6 of 6
metrics" banner and a WIN badge/highlight on every single row. The new
baseline correctly shows no winner banner and no WIN badges anywhere on
that screenshot (confirmed by inspecting the diff image directly, not just
the percentage) — a real, larger-looking diff (2.8%-3.9% across the six
desktop widths) that is entirely explained by this, not a layout break.

Verified against the reported bug's own numbers separately (a disposable
Playwright script with genuinely different A/B fixtures, deleted after
use): drawdown -6.6% vs -4.7%, VaR 95% -1.5% vs -1.4%, and CVaR 95% -2.2%
vs -1.6% now all correctly award WIN to the smaller-loss side, and the
win-count banner/Comparison-summary text recompute from the same corrected
per-metric results (now including CVaR and, when present, Beta/Alpha in
the counted set — the old win-count array omitted CVaR entirely and had
already special-cased Drawdown/VaR with an ad-hoc `Math.abs()`, a second,
silently-diverged copy of the same direction logic `MetricRow` used raw and
unabs'd).

Every PNG under `desktop/` for the eight `app-*` views reflects this fix
as of this date (all eight were recaptured together via `--routes=/app`;
only `app-compare` actually changed pixel content — the other seven
recaptured identical to what they already were).

## Local-only — not committed

`desktop/` is gitignored (this README isn't — it's the one file in this
directory git tracks). There's no CI running this repo, so nothing outside
your own machine ever needs these images, and `npm run shots:update-baseline`
re-writes every one of the 78 PNGs (~71MB) on every intentional desktop
change — committed, that's the full ~71MB added again, in full, each time,
permanently bloating history for a benefit (a `git diff` you can eyeball)
that has no one to serve here.

The real, and only, cost of that choice: baselines don't travel with the
repo. A fresh clone, or switching machines, starts with none — see below.

## First-time setup (no local baselines yet)

`npm run shots` fails loudly if a view has no baseline — it does not skip
the check silently, since "missing" is the normal state right after a
clone and silently skipping it would mean the desktop check simply never
ran, with nothing in the output calling that out. To generate a real
baseline set, capture it from code you trust, not from whatever you're
mid-edit on:

```bash
git checkout main          # or any other known-good commit
npm run shots:update-baseline
git checkout your-branch   # back to the branch you're actually working on
```

The generated PNGs stay on disk across the `git checkout` (they're
untracked, so checkout doesn't touch them) — you now have a local baseline
to check your branch's changes against.

## Updating after an intentional change

No checkout needed here — you're already on the branch with the change,
and confirming *that* render is the new truth:

```bash
npm run shots:update-baseline
```

Look at the resulting PNGs (or at least the ones for the views you
touched) before trusting them — that look is the actual point: a
desktop-visible change is now something that has to be seen and
deliberately accepted, not something that can slip through silently the
way the last two regressions did.

To update just one route/tab or width, scope it the same way any other
`shots.mjs` run would be scoped, e.g.:

```bash
npm run shots -- --update-baseline --routes=/ --widths=1440
```

## If a check fails

`npm run shots` prints which view(s) exceeded the noise floor (or have no
baseline at all) and, for a real diff, where to find a
`<view>-baseline-diff.png` (in that run's output directory, next to the
view's own screenshot) showing exactly which pixels differ. From there:

- **No baseline** — see "First-time setup" above.
- **Unintentional diff** — a real regression. Fix the code, not the baseline.
- **Intentional diff** — a deliberate desktop change. Confirm it looks
  right, then `npm run shots:update-baseline` (see above).
