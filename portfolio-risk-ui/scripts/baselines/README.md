# Desktop baselines

"Known good" screenshots, checked automatically by `npm run shots` (no
separate flag needed — see `shots.mjs`'s own top comment and its
`BASELINE_*` constants for the full rationale and the noise floor's
derivation).

```
desktop/<1280|1366|1440>/<view>.png
```

13 views × 3 widths = 39 files: every landing/legal route (`home`, `pro`,
`methodology`, `privacy`, `terms`) and every app tab (`app-dashboard`,
`app-risk`, `app-montecarlo`, `app-frontier`, `app-valuation`,
`app-compare`, `app-backtest`, `app-learn`).

**Why this exists:** two real regressions — a universal padding that
shrank every app tab's content, and an avatar that rendered when it
shouldn't have — both shipped clean through every other check this harness
runs (overflow measurement, scroll-reachability, tab-switch verification,
touch taps). Neither one moved a number any existing check was looking at;
they only ever looked wrong. This is the check for "looks different,"
independent of whatever metric a future regression happens to avoid moving.

## Local-only — not committed

`desktop/` is gitignored (this README isn't — it's the one file in this
directory git tracks). There's no CI running this repo, so nothing outside
your own machine ever needs these images, and `npm run shots:update-baseline`
re-writes every one of the 39 PNGs (~34MB) on every intentional desktop
change — committed, that's the full 34MB added again, in full, each time,
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
