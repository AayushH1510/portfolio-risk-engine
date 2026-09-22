// Responsive screenshot harness — captures both static routes and in-app
// views (driven past the Run Analysis / sign-in gate via mocked API fixtures)
// at a fixed set of viewports, against the already-running local dev server.
// Writes full-page PNGs to screenshots/<viewport>/<view>.png (gitignored,
// not build output) plus a manifest.json recording per-view horizontal
// overflow (document.documentElement.scrollWidth vs clientWidth), so
// "does this view overflow at 320/360" is a manifest query, not a rescreenshot.
//
// Assumes `npm run dev` is already running (see CLAUDE.md > How to run) —
// this script does not spawn or manage the dev server itself, same
// assumption every ad hoc Playwright check in this repo has made so far.
//
// Determinism: every context sets reducedMotion:'reduce', which drives the
// same `matchMedia('(prefers-reduced-motion: reduce)')` check already used
// in lib/motion/scroll-motion.ts and components/landing/Hero.jsx — reused
// there and in Frontier.jsx's canvas (see that file), not reinvented here.
// In-app views use committed JSON fixtures (scripts/fixtures/) instead of
// hitting the real backend, so results.json/full/fundamentals never change
// between two runs of this script.
//
// Usage:
//   npm run shots
//   npm run shots -- --widths=393,1440
//   npm run shots -- --routes=/,/app --base=http://localhost:5173
//   npm run shots -- --out=screenshots-before
import { chromium } from 'playwright'
import { mkdirSync, writeFileSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT      = path.resolve(__dirname, '..')
const FIXTURES_DIR = path.join(__dirname, 'fixtures')

// ─── Static routes (client-side SPA paths reachable cold, no interaction) ──
const DEFAULT_ROUTES = ['/', '/pro', '/methodology', '/app', '/privacy', '/terms', '/style-preview']

// ─── Viewports ──────────────────────────────────────────────────────────────
// Width -> a representative device height (not a strict aspect ratio, just a
// reasonable window to render full-page screenshots in before scrolling).
const HEIGHT_FOR_WIDTH = {
  320: 568,   // iPhone SE (1st gen) — narrowest width still seen in the wild
  360: 740,   // common Android (e.g. Galaxy S8)
  384: 854,   // Samsung Galaxy A54, Chrome, DPR 2.8125 — real device in QA rotation
  393: 852,   // iPhone 14/15/16 portrait
  440: 956,   // large Android (e.g. Pixel 7 Pro-class)
  768: 1024,  // iPad portrait
  1024: 768,  // iPad landscape / small laptop
  1280: 800,  // common laptop
  1366: 768,  // most common laptop width in the wild
  1440: 900,  // desktop
}
const DEFAULT_WIDTHS = [320, 360, 384, 393, 440, 768, 1024, 1280, 1366, 1440]
// One explicit landscape viewport — the portrait/landscape pair of the same
// physical phone (393x852 portrait above), since width-only sweeps never
// catch a phone rotated sideways into a short, wide viewport.
const LANDSCAPE_VIEWPORTS = [{ label: '852x393-landscape', width: 852, height: 393 }]

function viewportsFor(widths) {
  return [
    ...widths.map(w => ({ label: String(w), width: w, height: HEIGHT_FOR_WIDTH[w] ?? 900 })),
    ...LANDSCAPE_VIEWPORTS,
  ]
}

// ─── In-app named views ─────────────────────────────────────────────────────
// Mirrors App.jsx's own TABS array (id + header label) so the harness clicks
// through the real tab bar rather than guessing selectors.
const APP_TABS = [
  { id: 'dashboard',  label: 'Dashboard' },
  { id: 'risk',       label: 'Risk Analysis' },
  { id: 'montecarlo', label: 'Monte Carlo' },
  { id: 'frontier',   label: 'Efficient Frontier' },
  { id: 'valuation',  label: 'Valuation' },
  { id: 'compare',    label: 'Compare' },
  { id: 'backtest',   label: 'Backtest' },
  { id: 'learn',      label: 'Learn' },
]

// ─── Fixtures ───────────────────────────────────────────────────────────────
// Captured from a real 4-ticker (AAPL/MSFT/GOOGL/AMZN) + QQQ-benchmark run —
// see scripts/fixtures/ for provenance notes. analyse.json intentionally
// reuses the "full" shape: /api/analyse (used by Compare's Portfolio B) has
// always returned the same superset shape as /api/analyse-full.
function loadFixtures() {
  const read = name => JSON.parse(readFileSync(path.join(FIXTURES_DIR, name), 'utf8'))
  const summary = read('analyse-summary.json')
  const full    = read('analyse-full.json')
  const fundamentals = read('fundamentals.json')
  const stressTest    = read('stress-test.json')
  return {
    summary, full, fundamentals, stressTest,
    tickers: Object.keys(full.per_ticker_cumulative_returns),
    benchmark: full.benchmark,
  }
}

async function mockApiRoutes(page, fixtures) {
  await page.route('**/api/analyse-summary', r =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(fixtures.summary) }))
  await page.route('**/api/analyse-full', r =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(fixtures.full) }))
  // Plain /api/analyse — used directly by Compare's Portfolio B (useComparison.js).
  await page.route('**/api/analyse', r =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(fixtures.full) }))
  await page.route('**/api/fundamentals*', r =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(fixtures.fundamentals) }))
  // /api/stress-test (RiskAnalysis.jsx's Historical Scenarios section) — see
  // fixtures/README.md's postmortem: this call was left unmocked for a while,
  // which meant it hit a live (slow, non-deterministic) backend and stayed on
  // its loading spinner well past every other tab's fixed post-click wait —
  // long enough that the resulting 3-card grid never existed in the DOM by
  // the time capture() measured overflow, and a real repeat(3,1fr) phone-
  // width overflow in it went undetected. Mocking it removes that race
  // instead of just widening the wait (see settleNetwork() below for the
  // other half of that fix, for whatever the next unmocked/slow call is).
  await page.route('**/api/stress-test', r =>
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(fixtures.stressTest) }))
}

function parseArgs(argv) {
  const opts = {}
  for (const arg of argv) {
    const m = /^--([a-z]+)=(.*)$/.exec(arg)
    if (m) opts[m[1]] = m[2]
  }
  return {
    routes: opts.routes ? opts.routes.split(',').map(r => r.trim()) : DEFAULT_ROUTES,
    widths: opts.widths ? opts.widths.split(',').map(Number) : DEFAULT_WIDTHS,
    base:   opts.base || process.env.SHOTS_BASE_URL || 'http://localhost:5173',
    out:    opts.out || 'screenshots',
  }
}

// '/' -> 'home', '/pro' -> 'pro', '/style-preview' -> 'style-preview'
function slugForRoute(route) {
  if (route === '/') return 'home'
  return route.replace(/^\//, '').replace(/\//g, '-') || 'home'
}

async function measureOverflow(page) {
  return page.evaluate(() => {
    const clientWidth = document.documentElement.clientWidth
    const scrollWidth = document.documentElement.scrollWidth

    function selectorFor(el) {
      const cls = typeof el.className === 'string' && el.className.trim()
        ? '.' + el.className.trim().split(/\s+/).join('.') : ''
      return el.tagName.toLowerCase() + cls
    }

    // Every property in this codebase is authored inline (CLAUDE.md: "All
    // inline styles"), so el.style.overflowX/overflowY/overflow — the raw
    // inline declaration, not getComputedStyle — tells us exactly which
    // axis the author actually intended to scroll, as opposed to which axis
    // the browser computes afterward. That distinction matters: per spec,
    // setting only overflow-y to a non-visible value forces the browser to
    // also resolve overflow-x away from 'visible' (to 'auto') — so a plain
    // `overflowY:'auto'` root (Dashboard, Risk Analysis, Valuation, ...)
    // computes an overflow-x of 'auto' it never asked for. A prior version
    // of this function read computed style here and treated that spec
    // side-effect as if it were a deliberately-built horizontal scroll
    // affordance (like the header tab bar or Backtest/Valuation's table
    // wrappers, which explicitly author overflowX) — silently excluding
    // real, unreachable horizontal overflow (e.g. Dashboard's `1fr 220px`
    // chart row) from every count below. Reading the inline declaration
    // directly avoids that: an ancestor only counts as a genuine horizontal
    // scroll container if overflowX (or shorthand overflow) was actually
    // set to auto/scroll on it.
    function explicitAxis(el, axis) {
      const v = el.style[axis]
      return v === 'auto' || v === 'scroll'
    }
    function explicitShorthand(el) {
      const v = el.style.overflow
      return v === 'auto' || v === 'scroll'
    }

    // Classify the nearest actually-clipping ancestor for `el`:
    //   'horizontal' — an ancestor explicitly authored overflowX (or the
    //                  overflow shorthand) auto/scroll, and is genuinely
    //                  overflowing horizontally right now. Content is
    //                  reachable via scroll — a real, working affordance.
    //   'vertical'   — an ancestor explicitly authored ONLY overflowY
    //                  auto/scroll (no explicit overflowX), whose computed
    //                  overflow-x still ends up clipping this element.
    //                  Vertical scroll does not make horizontally-clipped
    //                  content reachable — this is still lost, just hidden
    //                  behind a scrollbar that looks like it "handles" it.
    //   null         — no clipping ancestor at all (or only overflow:hidden
    //                  ones, which were never reachable either way) — this
    //                  is genuine, uncontained page overflow.
    function classify(el) {
      let a = el.parentElement
      while (a) {
        const horizontalAuthored = explicitAxis(a, 'overflowX') || explicitShorthand(a)
        const verticalOnlyAuthored = !horizontalAuthored && explicitAxis(a, 'overflowY')
        if (horizontalAuthored || verticalOnlyAuthored) {
          if (a.scrollWidth > a.clientWidth + 1) {
            const ar = a.getBoundingClientRect()
            const er = el.getBoundingClientRect()
            if (er.right > ar.right + 0.5) {
              return { kind: horizontalAuthored ? 'horizontal' : 'vertical', ancestor: a }
            }
          }
        }
        a = a.parentElement
      }
      return null
    }

    const uncontained = []       // genuine, unreachable overflow
    const verticalOnly = []      // clipped horizontally, "reachable" only by a vertical scrollbar that doesn't help
    let containedHorizontalCount = 0

    for (const el of document.querySelectorAll('body *')) {
      const r = el.getBoundingClientRect()
      if (r.width === 0 || r.right <= clientWidth) continue
      const c = classify(el)
      if (c?.kind === 'horizontal') { containedHorizontalCount++; continue }
      if (c?.kind === 'vertical') { verticalOnly.push({ selector: selectorFor(el), right: Math.round(r.right) }); continue }
      uncontained.push({ selector: selectorFor(el), right: Math.round(r.right) })
    }

    uncontained.sort((a, b) => b.right - a.right)
    verticalOnly.sort((a, b) => b.right - a.right)

    const maxRight = uncontained.length ? uncontained[0].right : clientWidth
    const widestSelector = uncontained.length ? uncontained[0].selector : null

    return {
      scrollWidth, clientWidth, maxRight, widestSelector,
      top5Uncontained: uncontained.slice(0, 5),
      top5VerticalOnlyContained: verticalOnly.slice(0, 5),
      verticalOnlyContainedCount: verticalOnly.length,
      containedOverflowCount: containedHorizontalCount,
    }
  })
}

// Backstop for any tab section whose content depends on a request that
// isn't in mockApiRoutes (either a gap not yet noticed, or a genuinely
// unmocked call this harness doesn't know about yet) — waits for in-flight
// network activity to quiet down before capture() measures overflow, so a
// still-loading section's eventual DOM (and whatever it overflows) isn't
// measured before it exists. This is a backstop, not the primary fix for a
// known-slow call: see fixtures/README.md's stress-test.json entry for the
// actual bug this exists because of — a live, unmocked /api/stress-test
// call routinely outlasted every tab's fixed post-click waitForTimeout, so
// its 3-card grid didn't exist in the DOM yet when overflow was measured,
// and a real phone-width overflow in it shipped undetected. Mocking that
// specific call is the real fix (deterministic, fast, matches every other
// endpoint this harness already mocks); this wait is what keeps the next
// accidentally-unmocked or newly-added slow call from reproducing the same
// blind spot rather than relying on every future contributor remembering to
// widen a fixed timeout by hand. Short timeout and swallowed rejection
// because some pages are never truly idle (the smoke-field canvas's rAF
// loop doesn't count as network activity, but a genuinely open connection —
// a dev-tools-attached long-poll, unrelated to anything this harness drives
// — would otherwise hang capture() indefinitely).
async function settleNetwork(page) {
  await page.waitForLoadState('networkidle', { timeout: 3000 }).catch(() => {})
}

async function capture(page, outDir, slug, viewportLabel, results, extra = {}) {
  // Web fonts (Geist Sans/Mono, index.html) load from a CDN — a screenshot
  // taken before they've swapped in uses a fallback font with different
  // metrics, which can shift wrapped/stacked layout height enough to change
  // what's visible before an internal overflowY:auto region's scroll cuts
  // it off (confirmed: two runs of identical code differed by up to 9% at
  // 1440 on some app tabs, entirely from this, before this wait existed).
  await page.evaluate(() => document.fonts.ready).catch(() => {})
  const {
    scrollWidth, clientWidth, maxRight, widestSelector,
    top5Uncontained, top5VerticalOnlyContained, verticalOnlyContainedCount, containedOverflowCount,
  } = await measureOverflow(page)
  const overflowsHorizontally = scrollWidth > clientWidth + 1 || maxRight > clientWidth + 1
  const file = path.join(outDir, `${slug}.png`)
  await page.screenshot({ path: file, fullPage: true })
  results.push({
    viewport: viewportLabel, view: slug,
    file: path.relative(ROOT, file),
    scrollWidth, clientWidth, maxRight, widestSelector, overflowsHorizontally,
    top5Uncontained, top5VerticalOnlyContained, verticalOnlyContainedCount, containedOverflowCount,
    ...extra,
    ok: true,
  })
  const flags = [overflowsHorizontally && 'OVERFLOW', extra.tabClickForced && 'FORCED-CLICK'].filter(Boolean)
  const flagStr = flags.length ? `  [${flags.join(', ')}]` : ''
  console.log(`  ok    ${viewportLabel.padEnd(20)} ${slug.padEnd(20)} -> ${path.relative(ROOT, file)}${flagStr}`)
}

// Real, REACHABLE horizontal scroll surfaces only — document.documentElement
// plus every element that authors overflowY:'auto' (the pattern every
// scrolling tab root in this app uses, e.g. .dashboard-grid), each measured
// on its own scrollWidth vs clientWidth. Deliberately NOT measureOverflow()'s
// "uncontained" bucket: that bucket also catches elements clipped by a
// plain overflow:hidden ancestor — safe, invisible, unreachable content
// (the same kind of false positive the "Diversification" MetricCard label
// ellipsis already produces there) — which is exactly what the
// overflow:hidden containment fix (Dashboard.jsx, Comparison.jsx) now
// produces on purpose: a tooltip clipped mid-text at a card's edge, still
// present in the DOM at its full natural width so its own
// getBoundingClientRect() still reports past the viewport, but genuinely
// unreachable and invisible past the clip. Confirmed directly: after the
// fix, tapping Comparison's growth chart near its right edge left a
// recharts-tooltip-item's rect legitimately past clientWidth (its own box,
// clipped, never rendered past the card) while every scrollWidth here
// stayed unchanged — measureOverflow() flagged that as new "uncontained"
// overflow, a false positive this function doesn't reproduce.
async function measureReachableOverflow(page) {
  return page.evaluate(() => {
    const doc = { scrollWidth: document.documentElement.scrollWidth, clientWidth: document.documentElement.clientWidth }
    const containers = []
    for (const el of document.querySelectorAll('body *')) {
      if (el.style.overflowY === 'auto' || el.style.overflowY === 'scroll') {
        containers.push({
          selector: el.className && typeof el.className === 'string' ? '.' + el.className.trim().split(/\s+/).join('.') : el.tagName.toLowerCase(),
          scrollWidth: el.scrollWidth, clientWidth: el.clientWidth,
        })
      }
    }
    return { doc, containers }
  })
}

// Touch-triggered chart tooltip overflow — postmortem: Recharts activates
// its tooltip on touch via touchmove, not touchstart (handleTouchStart only
// forwards to the onMouseDown prop; handleTouchMove is what actually calls
// throttleTriggeredAfterMouseMove and sets isTooltipActive), so every check
// above — which only ever moves a mouse — has never exercised this code
// path at all. Confirmed once by forcing the failure mode directly
// (Dashboard's growth chart, 384px): pushing .recharts-tooltip-wrapper's
// translate far outside the chart grew .dashboard-grid's scrollWidth from
// 352px to 1410px while document.documentElement stayed unchanged at
// 384/384 — the escape happens through whichever ancestor authors only
// overflowY:'auto' (every scrolling tab root in this app), which per spec
// forces a computed overflow-x:auto as a side effect; a real, reachable
// horizontal scrollbar a user can act on, invisible to a document-level
// check. Recharts' own internal clamp (getTooltipTranslateXY) turned out
// not to be an absolute safety net — it floors against `viewBox`,
// populated from ResponsiveContainer's last *committed* measured width
// (React state), not a live DOM read, so any transient mismatch between
// that and the chart's actual current size can still produce an unclamped
// position; not reliably reproducible here via scripted taps/swipes/
// post-resize races in headless Chromium (real touch devices have
// different ResizeObserver/rAF timing, and this app deliberately leaves
// pinch-zoom enabled per RESPONSIVE_AUDIT.md's own standing decision —
// both plausible real triggers this harness can't emulate), so this check
// is a regression guard against the known failure *shape*, not a proof it
// will catch every possible trigger. The actual fix is CSS containment
// (overflow:hidden on each chart's wrapper div — Dashboard.jsx,
// Comparison.jsx) — this check exists so a future change that removes
// that containment, or reintroduces the same class of bug somewhere else,
// doesn't go unnoticed again.
async function checkTouchChartOverflow(page, outDir, viewportLabel, slug, results) {
  const wrapperCount = await page.locator('.recharts-wrapper').count()
  if (wrapperCount === 0) return

  const before = await measureReachableOverflow(page)

  const charts = Math.min(wrapperCount, 5)
  for (let i = 0; i < charts; i++) {
    const wrapper = page.locator('.recharts-wrapper').nth(i)
    const box = await wrapper.boundingBox().catch(() => null)
    if (!box || box.width === 0 || box.height === 0) continue
    for (const frac of [0.1, 0.5, 0.9]) {
      await wrapper.evaluate((node, [px, py]) => {
        // pageX/pageY are independent TouchInit fields, not derived from
        // clientX/clientY by the Touch() constructor — a real touch has
        // them set correctly by the browser; an incomplete synthetic one
        // silently no-ops inside Recharts (getMouseInfo reads event.pageX/
        // pageY, and NaN chartX/chartY fails its own inRange check).
        const touch = new Touch({
          identifier: Date.now() + Math.random(), target: node,
          clientX: px, clientY: py, pageX: px + window.scrollX, pageY: py + window.scrollY,
        })
        const mk = (type, t) => new TouchEvent(type, { bubbles: true, cancelable: true, touches: [t], changedTouches: [t], targetTouches: [t] })
        node.dispatchEvent(mk('touchstart', touch))
        node.dispatchEvent(mk('touchmove', touch))
      }, [box.x + box.width * frac, box.y + box.height * 0.5]).catch(() => {})
    }
  }
  await page.waitForTimeout(200)

  const after = await measureReachableOverflow(page)
  const docOverflowed = after.doc.scrollWidth > after.doc.clientWidth + 1 &&
    after.doc.scrollWidth > before.doc.scrollWidth + 1
  // Matched by selector, not array index — the set of overflowY:'auto'
  // elements is stable in practice (one scrolling root per tab), but
  // selector matching is correct even if querySelectorAll's order ever
  // isn't, where a positional zip wouldn't be.
  // Paired by position, not re-queried by selector: the set of
  // overflowY:'auto' elements is identical before and after (tapping a
  // chart adds a position:absolute tooltip node, not another auto-overflow
  // container, so nothing here is inserted, removed, or reordered) — a
  // plain index zip is correct and, critically, keeps every container
  // aligned with its own earlier self even when it ISN'T currently
  // overflowing. An earlier version of this filtered out non-overflowing
  // entries before advancing a per-selector queue, which — for two
  // className-less "div" containers on the same page, one pre-existing
  // and overflowing, one not — could desync the pairing and compare the
  // real one against the wrong baseline, misreporting existing, unrelated
  // overflow (RiskAnalysis's own documented §4 finding, nothing to do
  // with touch) as "new" from the tap. Confirmed via a direct before/after
  // measurement with zero taps involved: the overflow was already there.
  const newContainerOverflow = after.containers
    .map((c, i) => ({ c, b: before.containers[i] }))
    .filter(({ c, b }) => c.scrollWidth > c.clientWidth + 1 && (!b || c.scrollWidth > b.scrollWidth + 1))
    .map(({ c }) => c)
  const touchOverflow = docOverflowed || newContainerOverflow.length > 0

  results.push({
    viewport: viewportLabel, view: `${slug}-touch-check`,
    chartsTapped: charts, touchOverflow,
    before, after, newContainerOverflow,
    ok: true,
  })

  if (touchOverflow) {
    const file = path.join(outDir, `${slug}-touch-overflow.png`)
    await page.screenshot({ path: file, fullPage: true })
    console.log(`  FAIL  ${viewportLabel.padEnd(20)} ${slug.padEnd(20)} -> touch tap on a chart introduced a REACHABLE horizontal scroll surface (see ${path.relative(ROOT, file)})`)
  } else {
    console.log(`  ok    ${viewportLabel.padEnd(20)} ${(slug + '-touch').padEnd(20)} -> ${charts} chart(s) tapped, no new reachable overflow`)
  }
}

async function captureStaticRoutes(page, routes, base, outDir, viewportLabel, results) {
  for (const route of routes) {
    try {
      await page.goto(base + route, { waitUntil: 'networkidle', timeout: 30000 })
      await capture(page, outDir, slugForRoute(route), viewportLabel, results)
    } catch (err) {
      console.error(`  FAIL  ${viewportLabel.padEnd(20)} ${route.padEnd(20)} -> ${err.message}`)
      results.push({ viewport: viewportLabel, view: slugForRoute(route), error: err.message, ok: false })
    }
  }
}

// Clicks a locator normally first (real actionability check — matches
// what an actual pointer could reach), falling back to a forced click only
// if that fails. At narrow widths the header tab bar can be wider than the
// space the fixed-width sidebar leaves it (see captureAppViews), which puts
// a tab's natural click point off-screen — a real click there would fail
// too, so `forced: true` here is itself a finding, not just a workaround,
// and gets carried into the manifest per view.
async function robustClick(locator, timeout = 3000) {
  try {
    await locator.click({ timeout })
    return false
  } catch {
    await locator.click({ force: true, timeout: 2000 })
    return true
  }
}

// Clicking a header tab is not just "did Playwright's click() not throw."
// force:true still aims at the element's normal (mouse-coordinate) center
// point — when the nav container's clientWidth is 0 (narrow viewports, see
// captureAppViews below), that point resolves to nothing or to an unrelated
// element, so the click silently lands on empty space: no exception, no tab
// switch, and every subsequent screenshot quietly re-captures whatever tab
// was already active. Caught via App.jsx's aria-current="page" marker (set
// on the active tab button) — verify after each attempt, and only escalate
// to locator.dispatchEvent('click'), which fires the DOM event directly on
// the node regardless of its screen position, if the state genuinely didn't
// change.
async function clickTab(page, tab) {
  const locator = page.getByRole('button', { name: tab.label, exact: true })
  const isActive = () => locator.getAttribute('aria-current').then(v => v === 'page')

  let forced = false, dispatched = false
  try {
    await locator.click({ timeout: 2000 })
  } catch {
    forced = true
    await locator.click({ force: true, timeout: 2000 })
  }
  await page.waitForTimeout(120)

  if (!(await isActive())) {
    dispatched = true
    await locator.dispatchEvent('click')
    await page.waitForTimeout(120)
  }

  const verified = await isActive()
  return { forced, dispatched, verified }
}

// Drives the app past the Run Analysis gate (and, for Compare, its own
// Run comparison gate), then screenshots each of the 8 tabs by clicking
// through the header tab bar, plus the sign-in modal as its own view.
async function captureAppViews(page, fixtures, base, outDir, viewportLabel, results) {
  try {
    await mockApiRoutes(page, fixtures)
    // Pre-seed both first-visit localStorage flags (OnboardingTour.jsx,
    // FirstResultCallout.jsx) so neither one's fixed-position popover ever
    // mounts — both anchor via getBoundingClientRect() and neither clamps
    // to the viewport (unlike MetricTooltip), so leaving them to their own
    // timing would be a real source of run-to-run flakiness in the
    // screenshots regardless of viewport width.
    await page.addInitScript(() => {
      localStorage.setItem('varense_has_visited', 'true')
      localStorage.setItem('varense_has_seen_first_result', 'true')
    })
    await page.goto(base + '/app', { waitUntil: 'networkidle', timeout: 30000 })

    for (const rx of [/^skip/i, /got it/i]) {
      const btn = page.getByRole('button', { name: rx })
      if (await btn.count()) { try { await btn.first().click({ timeout: 800 }) } catch {} }
    }

    // Phase A: below --breakpoint-tablet (1024px) the sidebar is a
    // closed-by-default off-canvas drawer — its inputs are off-screen
    // (translateX(-100%)) until the hamburger opens it. The hamburger
    // itself is CSS-hidden at 1024px+, so isVisible() here doubles as the
    // "is there a drawer to open" check.
    const hamburger = page.getByRole('button', { name: /open sidebar/i })
    if (await hamburger.isVisible().catch(() => false)) {
      try {
        await hamburger.click({ timeout: 2000 })
      } catch {
        await hamburger.click({ force: true, timeout: 2000 })
      }
      await page.waitForTimeout(350)
      if (await page.locator('aside.sidebar-open').count() === 0) {
        await hamburger.dispatchEvent('click')
        await page.waitForTimeout(350)
      }
      if (await page.locator('aside.sidebar-open').count() === 0) {
        console.error(`  WARN  ${viewportLabel.padEnd(20)} app-views           sidebar drawer did not open — sidebar inputs may be unreachable`)
      }
    }

    // Match the sidebar's portfolio composition to the fixture's, so labels
    // (tickers, benchmark name) are consistent with the mocked data shown.
    const tickerInput = page.locator('input[placeholder="AAPL, MSFT, GOOGL"]')
    if (await tickerInput.count()) {
      await tickerInput.fill(fixtures.tickers.join(', '))
      await tickerInput.blur()
    }
    if (fixtures.benchmark) {
      const benchSelect = page.locator('select').filter({ has: page.locator('option[value="none"]') })
      if (await benchSelect.count()) { try { await benchSelect.selectOption(fixtures.benchmark) } catch {} }
    }

    await robustClick(page.getByRole('button', { name: /run analysis/i }).first())
    await page.waitForTimeout(2000)   // mocked responses resolve near-instantly; this covers React re-render + chart mount

    for (const tab of APP_TABS) {
      // The header tab bar (App.jsx) is wider than the room a fixed 260px
      // sidebar leaves it at narrow viewports — a tab's own click point can
      // end up positioned past the right edge of the viewport entirely, not
      // covered by anything, just off-screen. clickTab() surfaces that as
      // `forced: true`, and — critically — verifies via aria-current that
      // the tab actually switched rather than trusting a non-throwing
      // click(); a forced click at a zero-width nav aims at a point that
      // resolves to nothing, so it can "succeed" while doing nothing at all.
      // clientWidth === 0 here means there is no room for nav at all — the
      // fixed 260px sidebar plus the header's own auth cluster (sign-in +
      // account badge, which must stay visible per design) leave nothing,
      // independent of nav's own scroll mechanics. That's a sidebar-width
      // constraint, not something a header-only fix can close; expected to
      // clear once the sidebar becomes a collapsible drawer (Phase A) —
      // recorded per-view below, not treated as a regression.
      const navClientWidth = await page.locator('nav.tab-bar-scroll').evaluate(el => el.clientWidth).catch(() => null)
      const { forced, dispatched, verified } = await clickTab(page, tab)
      if (!verified) {
        console.error(`  WARN  ${viewportLabel.padEnd(20)} app-${tab.id.padEnd(12)} tab click did not switch activeTab (still showing whatever was active before) — capturing anyway, flagged in manifest`)
      }
      // Heavier canvas/chart tabs get a touch longer to finish drawing.
      await page.waitForTimeout(['montecarlo', 'frontier', 'backtest'].includes(tab.id) ? 1100 : 700)

      if (tab.id === 'compare') {
        const runCompare = page.getByRole('button', { name: 'Run comparison' })
        if (await runCompare.count()) {
          try { await robustClick(runCompare); await page.waitForTimeout(1000) } catch {}
        }
      }

      await settleNetwork(page)
      await capture(page, outDir, `app-${tab.id}`, viewportLabel, results, {
        tabClickForced: forced, tabClickDispatched: dispatched, tabSwitchVerified: verified,
        navClientWidth, blockedBySidebar: navClientWidth === 0,
      })

      await checkTouchChartOverflow(page, outDir, viewportLabel, `app-${tab.id}`, results)
    }

    // Scoped to the header specifically: at least one app tab's own content
    // (e.g. the "Sign in to save your portfolios" callout, or a Compare/Learn
    // CTA) can render a second "Sign in"-labelled control once real tab
    // content is reachable, which makes an unscoped role/name lookup
    // ambiguous (strict-mode violation) even though it never surfaced back
    // when forced clicks were silently failing to reach that content at all.
    //
    // Below the compact-viewport threshold the header's own copy is hidden
    // by design (.header-account-actions, index.css — RESPONSIVE_AUDIT.md's
    // header-account-controls pass) in favour of an icon-only indicator that
    // opens the drawer; the real Sign in control lives in the sidebar there
    // instead (.sidebar-account-actions). Checking window.matchMedia against
    // the same dual-axis condition the app itself uses (rather than parsing
    // viewportLabel) is what keeps this in sync with the app's own
    // definition instead of drifting into a second, hand-maintained copy of
    // "767" — see index.css's own "Compact viewport" banner comment for why
    // that drift is exactly the bug class this project has hit before.
    const isCompactViewport = await page.evaluate(() =>
      window.matchMedia('(max-width: 767px), (max-height: 767px)').matches
    )
    if (isCompactViewport) {
      const hamburger = page.locator('.sidebar-hamburger')
      if (await hamburger.count() && (await hamburger.getAttribute('aria-expanded')) !== 'true') {
        await robustClick(hamburger)
        await page.waitForTimeout(300)
      }
    }
    const signIn = isCompactViewport
      ? page.locator('aside.sidebar-shell').getByRole('button', { name: 'Sign in', exact: true })
      : page.getByRole('banner').getByRole('button', { name: 'Sign in', exact: true })
    if (await signIn.count()) {
      let forced = false, dispatched = false
      try {
        await signIn.click({ timeout: 2000 })
      } catch {
        forced = true
        await signIn.click({ force: true, timeout: 2000 })
      }
      await page.waitForTimeout(300)
      let verified = await page.getByText('Sign in to your account').isVisible().catch(() => false)
      if (!verified) {
        dispatched = true
        await signIn.dispatchEvent('click')
        await page.waitForTimeout(300)
        verified = await page.getByText('Sign in to your account').isVisible().catch(() => false)
      }
      if (!verified) console.error(`  WARN  ${viewportLabel.padEnd(20)} app-auth-modal      sign-in click did not open the modal`)
      // Same header squeeze as the tab bar — the sign-in button lives in the
      // same fixed-260px-sidebar-constrained header row, so it's blocked by
      // the same known cause, not a separate issue.
      const navClientWidth = await page.locator('nav.tab-bar-scroll').evaluate(el => el.clientWidth).catch(() => null)
      await capture(page, outDir, 'app-auth-modal', viewportLabel, results, {
        tabClickForced: forced, tabClickDispatched: dispatched, tabSwitchVerified: verified,
        navClientWidth, blockedBySidebar: navClientWidth === 0,
      })
    }
  } catch (err) {
    console.error(`  FAIL  ${viewportLabel.padEnd(20)} app-views -> ${err.message}`)
    results.push({ viewport: viewportLabel, view: 'app-views', error: err.message, ok: false })
  }
}

async function main() {
  const { routes, widths, base, out } = parseArgs(process.argv.slice(2))
  const viewports = viewportsFor(widths)
  const fixtures  = loadFixtures()

  console.log(`Base: ${base}`)
  console.log(`Routes: ${routes.join(', ')}`)
  console.log(`Viewports: ${viewports.map(v => v.label).join(', ')}`)
  console.log(`Fixture tickers: ${fixtures.tickers.join(', ')}  benchmark: ${fixtures.benchmark}`)

  const browser = await chromium.launch()
  const results = []

  for (const vp of viewports) {
    const outDir = path.join(ROOT, out, vp.label)
    mkdirSync(outDir, { recursive: true })

    // reducedMotion:'reduce' -> matchMedia('(prefers-reduced-motion: reduce)')
    // is true for every page in this context, so the Frontier "Optimal"
    // marker renders its resting frame instead of the rAF pulse.
    const context = await browser.newContext({
      viewport: { width: vp.width, height: vp.height },
      reducedMotion: 'reduce',
    })
    const page = await context.newPage()

    await captureStaticRoutes(page, routes, base, outDir, vp.label, results)
    await captureAppViews(page, fixtures, base, outDir, vp.label, results)

    await context.close()
  }

  await browser.close()

  const failed     = results.filter(r => !r.ok)
  const overflowing = results.filter(r => r.ok && r.overflowsHorizontally)
  const touchOverflowing = results.filter(r => r.ok && r.touchOverflow)
  const forcedClicks = results.filter(r => r.ok && r.tabClickForced)
  const sidebarBlocked = forcedClicks.filter(r => r.blockedBySidebar)
  const unexpectedForced = forcedClicks.filter(r => !r.blockedBySidebar)
  const unverifiedSwitches = results.filter(r => r.ok && r.tabSwitchVerified === false)
  writeFileSync(path.join(ROOT, out, 'manifest.json'), JSON.stringify(results, null, 2))

  console.log(`\n${results.length - failed.length}/${results.length} screenshots written.`)
  if (overflowing.length) {
    console.log(`\n${overflowing.length} view(s) overflow horizontally:`)
    for (const o of overflowing) console.log(`  ${o.viewport.padEnd(20)} ${o.view.padEnd(20)} maxRight=${o.maxRight} > clientWidth=${o.clientWidth}  (widest: ${o.widestSelector || '?'})`)
  }
  if (touchOverflowing.length) {
    console.log(`\n${touchOverflowing.length} view(s) introduced horizontal overflow from a chart TAP that wasn't there before (see checkTouchChartOverflow in this file):`)
    for (const t of touchOverflowing) {
      const docLine = `doc scrollWidth ${t.before.doc.scrollWidth}->${t.after.doc.scrollWidth} (clientWidth ${t.after.doc.clientWidth})`
      const containerLines = t.newContainerOverflow.map(c => `${c.selector} scrollWidth->${c.scrollWidth} (clientWidth ${c.clientWidth})`).join('; ')
      console.log(`  ${t.viewport.padEnd(20)} ${t.view.padEnd(20)} ${docLine}${containerLines ? '  |  ' + containerLines : ''}`)
    }
  }
  if (sidebarBlocked.length) {
    console.log(`\n${sidebarBlocked.length} view(s) needed a forced click, blocked by the fixed 260px sidebar (nav.clientWidth === 0 — no header-only fix closes this) — expected until the sidebar/drawer redesign, not a regression:`)
    for (const f of sidebarBlocked) console.log(`  ${f.viewport.padEnd(20)} ${f.view}`)
  }
  if (unexpectedForced.length) {
    console.log(`\n${unexpectedForced.length} view(s) needed a forced click for an UNEXPECTED reason (nav had nonzero clientWidth — investigate, this is not the known sidebar constraint):`)
    for (const f of unexpectedForced) console.log(`  ${f.viewport.padEnd(20)} ${f.view}`)
  }
  if (unverifiedSwitches.length) {
    console.log(`\n${unverifiedSwitches.length} view(s) may show the WRONG tab's content — click (even forced) never actually switched activeTab, verified via aria-current, even after a direct dispatchEvent('click'):`)
    for (const u of unverifiedSwitches) console.log(`  ${u.viewport.padEnd(20)} ${u.view}`)
  }
  if (failed.length || unexpectedForced.length || unverifiedSwitches.length || touchOverflowing.length) process.exitCode = 1
}

main()
