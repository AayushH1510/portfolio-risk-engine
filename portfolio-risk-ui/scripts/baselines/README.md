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
