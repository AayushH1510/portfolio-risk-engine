// Generates src/landing-fonts.css from the @fontsource packages' own CSS —
// every rule identical to fontsource's, except font-display: swap rewritten
// to font-display: optional. Run via `prebuild`/`predev` (see package.json),
// same convention as tokens-to-css.mjs alongside this file.
//
// Why this exists: `swap` paints the fallback font immediately, then swaps
// to the real webfont whenever it finishes loading — a page-wide reflow
// (every element using that family re-lays-out at the real font's metrics)
// that happens *after* first paint, which is exactly what Cumulative Layout
// Shift measures. `optional` gives the browser a very short window (~100ms)
// to use the font if it's already available (e.g. cached from a previous
// visit) and otherwise commits to the fallback for this page view — no
// later swap, no later reflow. The font still downloads in the background
// and is cached for next time, so this only costs a visual difference (the
// fallback face instead of Newsreader/Lora/IBM Plex) on a first visit on a
// slow connection, not a permanent one. This is the standard fix for
// font-swap-driven CLS (over e.g. reserving blank space for text, which
// hides the symptom without removing the reflow) — see Lighthouse's own
// "cumulative-layout-shift" and "font-display" audit guidance.
//
// Can't just pass a different font-display via @fontsource's own import —
// every weight/subset file it ships hardcodes `swap` — so this rewrites the
// installed CSS text directly rather than hand-copying ~40 @font-face
// blocks (one per weight x style x unicode-range subset) that would drift
// out of sync the next time these packages are updated.
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const NODE_MODULES = path.resolve(__dirname, '../node_modules')

// Exactly the files main.jsx currently imports from @fontsource — kept in
// sync by hand (same as tokens-to-css.mjs's own relationship to
// design/tokens.json): if that import list changes, this list needs the
// matching update, or the newly-added weight ships with font-display:
// swap again, un-fixed.
const SOURCES = [
  '@fontsource/newsreader/200.css',
  '@fontsource/newsreader/200-italic.css',
  '@fontsource/newsreader/300.css',
  '@fontsource/lora/400.css',
  '@fontsource/ibm-plex-sans/300.css',
  '@fontsource/ibm-plex-sans/400.css',
  '@fontsource/ibm-plex-sans/500.css',
  '@fontsource/ibm-plex-mono/400.css',
  '@fontsource/ibm-plex-mono/500.css',
]

const banner = `/* GENERATED FILE — do not edit by hand.
   Source: the @fontsource CSS listed in scripts/fix-font-display.mjs's own
   SOURCES array (runs in prebuild/predev), with font-display: swap
   rewritten to font-display: optional — see that script's comment for why.
   Import this file instead of the individual @fontsource/*.css files. */
`

const chunks = SOURCES.map(specifier => {
  const pkgPath = specifier.replace(/^@fontsource\//, '')
  const [pkg, ...rest] = pkgPath.split('/')
  const absPath = path.join(NODE_MODULES, '@fontsource', pkg, ...rest)
  const css = readFileSync(absPath, 'utf8')

  // Same relative depth every one of these files is at (node_modules/
  // @fontsource/<pkg>/<file>.css, one directory below @fontsource/<pkg>/),
  // so a single fixed '../node_modules/...' prefix (this output lives in
  // src/, one level up from node_modules) is correct for all of them
  // without needing per-file path math.
  const rewrittenUrls = css.replace(
    /url\(\.\/files\//g,
    `url(../node_modules/@fontsource/${pkg}/files/`
  )
  const rewrittenDisplay = rewrittenUrls.replace(/font-display:\s*swap;/g, 'font-display: optional;')

  const swapCount = (css.match(/font-display:\s*swap;/g) || []).length
  if (swapCount === 0) {
    throw new Error(`${specifier}: expected to find "font-display: swap" to rewrite, found none — has @fontsource changed its output format?`)
  }

  return `/* ${specifier} — font-display: swap -> optional */\n${rewrittenDisplay}`
})

console.log(banner + '\n' + chunks.join('\n'))
