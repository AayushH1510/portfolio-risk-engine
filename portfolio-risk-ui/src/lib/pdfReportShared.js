// Shared helpers for both PDF report generators — lib/exportPdfReport.js
// (the one-page Summary PDF) and lib/exportFullReportPdf.js (the new
// multi-page Full report PDF). Both render into a window.open() +
// document.write() pop-up with no access to the app's own index.css, so
// both need the same three things: design/tokens.json walked into CSS
// variables, the same number formatters, and the same pop-up-blocked
// fallback. Defined once here, imported by both, so the two reports can't
// drift from each other the way the Summary report's own colours drifted
// from the brand before this file existed — see exportPdfReport.js's own
// git history for that postmortem. A hand-copied second version of
// tokensToCSSVars in the new module would have been exactly that mistake
// again, just with two call sites instead of one.
import tokens from '../../design/tokens.json'

export const fmt  = v => v != null ? `${(v * 100).toFixed(1)}%` : 'N/A'
export const fmtN = v => v != null ? v.toFixed(2) : 'N/A'
export const fmtD = v => v != null ? `$${Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: 0 })}` : 'N/A'

// Mirrors scripts/tokens-to-css.mjs's own walk() — same recursive
// leaf-with-"value"-key traversal, same `--color-a-b-c` naming — but run
// here at runtime instead of at build time. tokens-to-css.mjs can do its
// walk once and write a committed file because its output (landing-theme
// .css) ships to a document that can <link> a stylesheet; a report pop-up
// is a second document.write()'d document with no build step and no way to
// <link> this project's own generated CSS, so the only way for it to read
// the same source of truth is to walk the same JSON itself, inline, and
// hand the result to the popup as a plain CSS-variables string.
export function tokensToCSSVars(node, path = []) {
  const lines = []
  for (const [k, v] of Object.entries(node)) {
    if (k.startsWith('$')) continue
    if (v && typeof v === 'object' && 'value' in v) {
      lines.push(`--${[...path, k].join('-')}: ${v.value};`)
    } else if (v && typeof v === 'object' && !Array.isArray(v)) {
      lines.push(...tokensToCSSVars(v, [...path, k]))
    }
  }
  return lines
}

// Runs in the MAIN page's context (this module is imported into the app,
// not into the pop-up), so the real app tokens from index.css are already
// on :root here — no need to duplicate them. Used only when window.open()
// is refused (pop-up blocker), by either report.
export function showExportMessage(text) {
  const el = document.createElement('div')
  el.setAttribute('role', 'status')
  el.textContent = text
  el.style.cssText = [
    'position:fixed', 'left:50%', 'bottom:24px', 'transform:translateX(-50%)',
    'z-index:2147483647', 'background:var(--signal-negative, #e05c5c)', 'color:#fff',
    'padding:12px 20px', 'font-family:var(--font-primary, system-ui, sans-serif)',
    'font-size:13px', 'max-width:min(90vw, 420px)', 'text-align:center',
    'box-shadow:0 4px 16px rgba(0,0,0,0.3)', 'border-radius:0',
  ].join(';')
  document.body.appendChild(el)
  setTimeout(() => el.remove(), 5000)
}

// Opens the report pop-up, or shows the pop-up-blocked message and returns
// null. Checked before either report builds its (large) HTML string at
// all — nothing to do with it if the pop-up never opens. A blocked pop-up
// returns null rather than throwing, which used to fail silently (a
// document.write on a null win throws, visible only in the console); both
// callers now just check the return value.
export function openReportPopup() {
  const win = window.open('', '_blank', 'width=1100,height=900')
  if (!win) {
    showExportMessage('Your browser blocked the report pop-up. Please allow pop-ups for this site, then try Export PDF again.')
    return null
  }
  return win
}

// The --color-*/--doc-* variable block shared by every report document's
// <style>:root{}. A light, print-friendly read of the same brand: no light
// mode exists in design/tokens.json (colorMode: "dark-only"), so the --doc-*
// lines reuse existing dark-theme token VALUES in an inverted role — the
// darkest canvas tone (color.bg.base) as ink on paper, the lightest text
// tone (color.text.primary) as the page itself — rather than inventing new,
// undocumented hex. Everything else is a direct var(--color-*) reference,
// nothing new to keep in sync.
export function docTokenCSS() {
  const colorVars = tokensToCSSVars(tokens.color, ['color']).join('\n  ')
  return `${colorVars}
  --doc-bg: var(--color-text-primary);
  --doc-card: var(--color-text-body);
  --doc-track: var(--color-text-body);
  --doc-ink: var(--color-bg-base);
  --doc-ink-secondary: var(--color-line-interactive);
  --doc-ink-muted: var(--color-line-hover);
  --doc-hairline: var(--color-line-default);
  --doc-heat-high: var(--color-signal-negative);
  --doc-heat-mid: var(--color-signal-warning);
  --doc-heat-low: var(--color-accent-mint);
  --doc-font-primary: 'Geist', ui-sans-serif, system-ui, -apple-system, sans-serif;
  --doc-font-mono: 'Geist Mono', ui-monospace, 'SF Mono', Menlo, monospace;`
}

// Same CDN both reports already relied on individually — Fontsource's own
// hosted build, not a Google Fonts link (consistent with the rest of the
// app, which self-hosts Geist via @fontsource).
export const FONT_LINKS = `
  <link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>
  <link href="https://cdn.jsdelivr.net/npm/@fontsource/geist-sans@5.2.5/400.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/@fontsource/geist-sans@5.2.5/500.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/@fontsource/geist-sans@5.2.5/600.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/@fontsource/geist-mono@5.2.5/400.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/@fontsource/geist-mono@5.2.5/500.css" rel="stylesheet">
`

// The real Varense mark's geometry (viewBox "3 9 58 41", two <path>s) —
// matches public/mark.svg / mark-inverse.svg exactly, just with its stroke
// bound to --doc-ink via currentColor's CSS-var equivalent instead of
// literally referencing either file by <img src> (unreliable in an
// about:blank document.write()'d document). Callers wrap this in their own
// <svg viewBox="3 9 58 41" ...> with whatever width/height/aria-label they
// need; kept as inner markup only so neither report hand-copies the path
// data a second time.
export const MARK_PATHS = `
  <path d="M5.19,44.5 A27,30 0 1 1 58.81,44.5" fill="none" stroke="var(--doc-ink)" stroke-width="3.5" stroke-linejoin="miter" stroke-linecap="butt"></path>
  <path d="M5,48 L16,48 L23,41.5 L26.5,44.5 L36,34 L48,48 L59,48" fill="none" stroke="var(--doc-ink)" stroke-width="3.5" stroke-linejoin="miter" stroke-linecap="butt"></path>
`
