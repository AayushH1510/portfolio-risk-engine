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
  393: 852,   // iPhone 14/15/16 portrait
  440: 956,   // large Android (e.g. Pixel 7 Pro-class)
  768: 1024,  // iPad portrait
  1024: 768,  // iPad landscape / small laptop
  1440: 900,  // desktop
}
const DEFAULT_WIDTHS = [320, 360, 393, 440, 768, 1024, 1440]
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
  return {
    summary, full, fundamentals,
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
    // document.documentElement.scrollWidth alone misses overflow that's
    // contained by an ancestor's own explicit `overflow` — App.jsx sets
    // overflow:hidden at several shell levels on purpose, so a tab's
    // content can force extra width that's invisibly clipped there and
    // never bubbles up to the document. getBoundingClientRect() still
    // reports an element's true laid-out box even when a clipping ancestor
    // stops it from painting, so walk every element and take the widest
    // right edge — this catches that case too, and names the culprit.
    let maxRight = clientWidth
    let widestSelector = null
    for (const el of document.querySelectorAll('body *')) {
      const r = el.getBoundingClientRect()
      if (r.width > 0 && r.right > maxRight) {
        maxRight = r.right
        const cls = typeof el.className === 'string' && el.className.trim()
          ? '.' + el.className.trim().split(/\s+/).join('.') : ''
        widestSelector = el.tagName.toLowerCase() + cls
      }
    }
    return { scrollWidth, clientWidth, maxRight: Math.round(maxRight), widestSelector }
  })
}

async function capture(page, outDir, slug, viewportLabel, results, extra = {}) {
  const { scrollWidth, clientWidth, maxRight, widestSelector } = await measureOverflow(page)
  const overflowsHorizontally = scrollWidth > clientWidth + 1 || maxRight > clientWidth + 1
  const file = path.join(outDir, `${slug}.png`)
  await page.screenshot({ path: file, fullPage: true })
  results.push({
    viewport: viewportLabel, view: slug,
    file: path.relative(ROOT, file),
    scrollWidth, clientWidth, maxRight, widestSelector, overflowsHorizontally,
    ...extra,
    ok: true,
  })
  const flags = [overflowsHorizontally && 'OVERFLOW', extra.tabClickForced && 'FORCED-CLICK'].filter(Boolean)
  const flagStr = flags.length ? `  [${flags.join(', ')}]` : ''
  console.log(`  ok    ${viewportLabel.padEnd(20)} ${slug.padEnd(20)} -> ${path.relative(ROOT, file)}${flagStr}`)
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
      // covered by anything, just off-screen. robustClick() surfaces that
      // as `forced: true` rather than silently working around it.
      const forced = await robustClick(page.getByRole('button', { name: tab.label, exact: true }))
      // Heavier canvas/chart tabs get a touch longer to finish drawing.
      await page.waitForTimeout(['montecarlo', 'frontier', 'backtest'].includes(tab.id) ? 1100 : 700)

      if (tab.id === 'compare') {
        const runCompare = page.getByRole('button', { name: 'Run comparison' })
        if (await runCompare.count()) {
          try { await robustClick(runCompare); await page.waitForTimeout(1000) } catch {}
        }
      }

      await capture(page, outDir, `app-${tab.id}`, viewportLabel, results, { tabClickForced: forced })
    }

    const signIn = page.getByRole('button', { name: 'Sign in', exact: true })
    if (await signIn.count()) {
      const forced = await robustClick(signIn)
      await page.waitForTimeout(400)
      await capture(page, outDir, 'app-auth-modal', viewportLabel, results, { tabClickForced: forced })
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
  const forcedClicks = results.filter(r => r.ok && r.tabClickForced)
  writeFileSync(path.join(ROOT, out, 'manifest.json'), JSON.stringify(results, null, 2))

  console.log(`\n${results.length - failed.length}/${results.length} screenshots written.`)
  if (overflowing.length) {
    console.log(`\n${overflowing.length} view(s) overflow horizontally:`)
    for (const o of overflowing) console.log(`  ${o.viewport.padEnd(20)} ${o.view.padEnd(20)} maxRight=${o.maxRight} > clientWidth=${o.clientWidth}  (widest: ${o.widestSelector || '?'})`)
  }
  if (forcedClicks.length) {
    console.log(`\n${forcedClicks.length} view(s) needed a forced click to reach (the real click point was off-screen or unreachable):`)
    for (const f of forcedClicks) console.log(`  ${f.viewport.padEnd(20)} ${f.view}`)
  }
  if (failed.length) process.exitCode = 1
}

main()
