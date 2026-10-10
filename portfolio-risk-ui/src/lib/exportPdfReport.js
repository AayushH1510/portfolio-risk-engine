// The actual PDF-report generation logic — split out of components/ExportPDF.jsx
// so it can be dynamically imported (both callers below already do
// `import('./lib/exportPdfReport')` rather than a static top-level import)
// instead of shipping in every /app bundle whether or not Export PDF is
// ever clicked. It's pure string-templating (no Recharts, no heavy
// dependency) but it's still a genuinely large function — see
// buildReportHTML below — and it only ever runs on a click, the textbook
// case for deferring a chunk rather than bundling it eagerly. ExportPDF.jsx
// also preloads this same specifier (useEffect on mount) once a run's
// results are on screen, so by the time someone actually taps Export the
// import below has usually already resolved from cache.
import { getBenchmarkLabel } from './benchmarks'
// Same import Hero.jsx already uses for the landing page's own styling —
// Vite inlines JSON imports as plain objects, so this is live data, not a
// build step. Doing the same thing here is what stops this report's colours
// drifting from the brand again: see tokensToCSSVars below.
import tokens from '../../design/tokens.json'

const fmt  = v => v != null ? `${(v * 100).toFixed(1)}%` : 'N/A'
const fmtN = v => v != null ? v.toFixed(2) : 'N/A'
const fmtD = v => v != null ? `$${Math.abs(v).toLocaleString('en-US', { maximumFractionDigits:0 })}` : 'N/A'

// Mirrors scripts/tokens-to-css.mjs's own walk() — same recursive
// leaf-with-"value"-key traversal, same `--color-a-b-c` naming — but run
// here at runtime instead of at build time. tokens-to-css.mjs can do its
// walk once and write a committed file because its output (landing-theme
// .css) ships to a document that can <link> a stylesheet; this report is a
// second document.write()'d document with no build step and no way to
// <link> this project's own generated CSS, so the only way for it to read
// the same source of truth is to walk the same JSON itself, inline, and
// hand the result to the popup as a plain CSS-variables string.
function tokensToCSSVars(node, path = []) {
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
// on :root here — no need to duplicate them, unlike the report document
// itself below. Used only when window.open() is refused (pop-up blocker);
// nothing else in this file calls it.
function showExportMessage(text) {
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

export function runExportPDF({ data, tickers, weights }) {
  const buildReportHTML = () => {
    const d    = data
    const cum  = d.cumulative_returns
    const mc   = d.monte_carlo
    const vc   = d.var_cvar
    const ba   = d.beta_alpha
    const corr = d.correlation_matrix
    const date = new Date().toLocaleDateString('en-GB', { day:'numeric', month:'long', year:'numeric' })
    const benchmarkLabel = getBenchmarkLabel(d.benchmark)

    const metricRows = [
      { label:'Annual Return',  val: fmt(d.annualised_return),       good: d.annualised_return > 0 },
      { label:'Volatility',     val: fmt(d.annualised_volatility),    good: d.annualised_volatility < 0.25 },
      { label:'Sharpe Ratio',   val: fmtN(d.sharpe_ratio),           good: d.sharpe_ratio > 1 },
      { label:'Sortino Ratio',  val: fmtN(d.sortino_ratio),          good: d.sortino_ratio > 1 },
      { label:'Max Drawdown',   val: fmt(d.max_drawdown),            good: false },
      { label:'VaR 95%',        val: fmt(vc.var_pct),                good: false },
      { label:'CVaR 95%',       val: fmt(vc.cvar_pct),               good: false },
      ...(ba ? [
        { label:'Beta',  val: fmtN(ba.beta),  good: ba.beta < 1.2 },
        { label:'Alpha', val: fmt(ba.alpha),  good: ba.alpha > 0 },
      ] : []),
    ]

    // DESIGN.md's actual Metric Card spec: the value takes the signal colour,
    // but only a warning/negative card gets a coloured top border — a good
    // card keeps the ordinary hairline, so colour reads as signal on the
    // number itself, not as decoration on every card regardless of direction.
    const metricCards = metricRows.map(m => `
      <div class="vr-metric-card" style="background:var(--doc-card);border:1px solid var(--doc-hairline);${m.good ? '' : 'border-top:2px solid var(--color-signal-negative);'}">
        <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:var(--doc-ink-muted)">${m.label}</div>
        <div class="vr-metric-value" style="font-weight:700;font-family:var(--doc-font-mono);color:${m.good ? 'var(--color-accent-mint)':'var(--color-signal-negative)'}">${m.val}</div>
      </div>
    `).join('')

    const tickerPills = tickers.map((t, i) => `
      <span style="display:inline-block;background:var(--doc-card);border:1px solid var(--doc-hairline);padding:2px 9px;font-size:11px;font-weight:700;color:var(--doc-ink);font-family:var(--doc-font-mono);margin-right:5px">${t} ${Math.round(weights[i]*100)}%</span>
    `).join('')

    // Neutral, not a signal — allocation weight isn't a gain, loss, or risk
    // figure, so it stays in the ink scale rather than borrowing the green
    // family decoratively (the old version coloured every bar from a
    // green-ish palette regardless of ticker).
    const allocBars = tickers.map((t, i) => `
      <div class="vr-alloc-row">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
          <span style="font-weight:700;font-family:var(--doc-font-mono);font-size:13px;color:var(--doc-ink)">${t}</span>
          <span style="font-weight:700;color:var(--doc-ink-secondary);font-size:13px">${Math.round(weights[i]*100)}%</span>
        </div>
        <div style="height:6px;background:var(--doc-track)">
          <div style="height:6px;width:${weights[i]*100}%;background:var(--doc-ink-muted)"></div>
        </div>
      </div>
    `).join('')

    const mcItems = [
      { label:'Median',        val: fmtD(mc.p50_final),                       color:'var(--doc-ink)' },
      { label:'Good year',     val: fmtD(mc.p95_final),                       color:'var(--color-accent-mint)' },
      { label:'Bad year',      val: fmtD(mc.p5_final),                        color:'var(--color-signal-negative)' },
      { label:'Profit chance', val: `${Math.round(mc.prob_profit*100)}%`,      color: mc.prob_profit > 0.6 ? 'var(--color-accent-mint)' : 'var(--color-signal-warning)' },
    ].map(m => `
      <div class="vr-mc-item" style="background:var(--doc-card);border:1px solid var(--doc-hairline);">
        <div style="font-size:9px;color:var(--doc-ink-muted);font-weight:700;text-transform:uppercase;margin-bottom:2px">${m.label}</div>
        <div style="font-size:14px;font-weight:700;color:${m.color};font-family:var(--doc-font-mono)">${m.val}</div>
      </div>
    `).join('')

    const riskRows = [
      { label:'VaR 95%',      a: fmt(vc.var_pct),    b:`${fmtD(vc.var_dollar)}/day` },
      { label:'CVaR 95%',     a: fmt(vc.cvar_pct),   b:`${fmtD(vc.cvar_dollar)} avg` },
      { label:'Max Drawdown', a: fmt(d.max_drawdown), b:'peak to trough' },
      ...(ba ? [
        { label:'Beta',  a: fmtN(ba.beta),  b:`vs ${benchmarkLabel}` },
        { label:'Alpha', a: fmt(ba.alpha),  b:"Jensen's" },
      ] : []),
    ].map(r => `
      <div class="vr-risk-row" style="display:flex;justify-content:space-between;border-bottom:1px solid var(--doc-hairline)">
        <span style="color:var(--doc-ink-muted);font-size:12px">${r.label}</span>
        <div style="text-align:right">
          <div style="font-weight:700;color:var(--color-signal-negative);font-family:var(--doc-font-mono);font-size:12px">${r.a}</div>
          <div style="font-size:10px;color:var(--doc-ink-muted)">${r.b}</div>
        </div>
      </div>
    `).join('')

    // Correlation heat: high correlation is the concentration-risk case
    // (negative signal), low is the diversification case (positive signal),
    // same three-tier reasoning as before, now sourced from the same
    // tokens as everything else instead of one-off hex.
    const corrHeaders = corr.tickers.map(t => `<th class="vr-corr-head" style="font-size:10px;color:var(--doc-ink-muted);font-family:var(--doc-font-mono);text-align:center">${t}</th>`).join('')
    const corrRows = corr.tickers.map((row, ri) => {
      const cells = corr.values[ri].map((val, ci) => {
        const bg = val >= 0.7 ? 'var(--doc-heat-high)' : val >= 0.4 ? 'var(--doc-heat-mid)' : 'var(--doc-heat-low)'
        return `<td class="vr-corr-cell" style="text-align:center;font-size:11px;font-family:var(--doc-font-mono);font-weight:600;background:${bg};color:var(--doc-ink)">${val.toFixed(2)}</td>`
      }).join('')
      return `<tr><td style="font-size:10px;color:var(--doc-ink-muted);font-family:var(--doc-font-mono);text-align:right;padding-right:8px">${row}</td>${cells}</tr>`
    }).join('')

    // Growth chart as SVG sparkline
    const cumVals = cum.values
    const minC = Math.min(...cumVals), maxC = Math.max(...cumVals)
    const rngC = maxC - minC || 0.01
    const svgW = 700, svgH = 120
    const points = cumVals.map((v, i) =>
      `${(i / (cumVals.length-1)) * svgW},${svgH - ((v - minC) / rngC) * svgH}`
    ).join(' ')
    const zeroY = svgH - ((0 - minC) / rngC) * svgH

    return `
      <div class="vr-header" style="background:var(--doc-bg);border-bottom:1px solid var(--doc-hairline);display:flex;justify-content:space-between;align-items:flex-start">
        <div>
          <div class="vr-brand" style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
            <svg viewBox="3 9 58 41" width="26" height="18.4" fill="none" role="img" aria-label="Varense">
              <path d="M5.19,44.5 A27,30 0 1 1 58.81,44.5" fill="none" stroke="var(--doc-ink)" stroke-width="3.5" stroke-linejoin="miter" stroke-linecap="butt"></path>
              <path d="M5,48 L16,48 L23,41.5 L26.5,44.5 L36,34 L48,48 L59,48" fill="none" stroke="var(--doc-ink)" stroke-width="3.5" stroke-linejoin="miter" stroke-linecap="butt"></path>
            </svg>
            <span style="font-family:var(--doc-font-primary);font-weight:500;font-size:19px;letter-spacing:-0.035em;color:var(--doc-ink)">Varense</span>
          </div>
          <div>${tickerPills}</div>
        </div>
        <div class="vr-header-right" style="text-align:right">
          <div style="font-size:18px;font-weight:700;color:var(--doc-ink);font-family:var(--doc-font-primary)">Varense Portfolio Report</div>
          <div style="font-size:12px;color:var(--doc-ink-secondary);margin-top:4px">${date}</div>
          <div style="font-size:11px;color:var(--doc-ink-muted);margin-top:5px;font-family:var(--doc-font-mono)">${d.period.start} - ${d.period.end} · ${d.period.n_days} days</div>
        </div>
      </div>

      <div class="vr-body" style="display:flex;flex-direction:column;background:var(--doc-bg)">

        <div class="vr-section">
          <div class="vr-label" style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--doc-ink-muted)">Key Metrics</div>
          <div class="vr-metric-grid" style="display:flex;flex-wrap:wrap">${metricCards}</div>
        </div>

        <div class="vr-section">
          <div class="vr-label" style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--doc-ink-muted)">Portfolio Growth</div>
          <div class="vr-card vr-chart-card" style="background:var(--doc-card);border:1px solid var(--doc-hairline);">
            <div class="vr-chart-meta" style="display:flex;justify-content:space-between;font-size:12px">
              <span style="color:var(--doc-ink-secondary)">Cumulative return from ${d.period.start}</span>
              <span style="font-weight:700;color:${d.annualised_return > 0 ? 'var(--color-accent-mint)':'var(--color-signal-negative)'};font-family:var(--doc-font-mono)">${fmt(cumVals[cumVals.length-1])} total</span>
            </div>
            <svg class="vr-chart-svg" width="100%" height="${svgH}" viewBox="0 0 ${svgW} ${svgH}" preserveAspectRatio="none">
              <defs>
                <linearGradient id="sg1" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="var(--color-accent-mint)" stop-opacity="0.2"/>
                  <stop offset="100%" stop-color="var(--color-accent-mint)" stop-opacity="0"/>
                </linearGradient>
              </defs>
              ${zeroY > 0 && zeroY < svgH ? `<line x1="0" y1="${zeroY}" x2="${svgW}" y2="${zeroY}" stroke="var(--doc-hairline)" stroke-dasharray="6,4" stroke-width="1"/>` : ''}
              <polygon points="${points} ${svgW},${svgH} 0,${svgH}" fill="url(#sg1)"/>
              <polyline points="${points}" fill="none" stroke="var(--color-accent-mint)" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
            </svg>
          </div>
        </div>

        <div class="vr-grid2" style="display:grid">
          <div class="vr-section">
            <div class="vr-label" style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--doc-ink-muted)">Allocation</div>
            <div class="vr-card" style="background:var(--doc-card);border:1px solid var(--doc-hairline);">${allocBars}</div>
          </div>
          <div class="vr-section">
            <div class="vr-label" style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--doc-ink-muted)">Monte Carlo - 1 Year</div>
            <div class="vr-card" style="background:var(--doc-card);border:1px solid var(--doc-hairline);">
              <div class="vr-mc-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:6px">${mcItems}</div>
              <div style="margin-top:10px;font-size:10px;color:var(--doc-ink-muted)">Based on ${mc.n_simulations?.toLocaleString() || '1,000'} simulations. Median: ${fmtD(mc.p50_final)} · Good: ${fmtD(mc.p95_final)} · Bad: ${fmtD(mc.p5_final)}</div>
            </div>
          </div>
        </div>

        <div class="vr-grid2" style="display:grid">
          <div class="vr-section">
            <div class="vr-label" style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--doc-ink-muted)">Downside Risk</div>
            <div class="vr-card" style="background:var(--doc-card);border:1px solid var(--doc-hairline);">${riskRows}</div>
          </div>
          <div class="vr-section">
            <div class="vr-label" style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--doc-ink-muted)">Correlation Matrix</div>
            <div class="vr-card" style="background:var(--doc-card);border:1px solid var(--doc-hairline);">
              <table class="vr-corr-table" style="width:100%;border-collapse:separate">
                <thead><tr><th style="width:50px"></th>${corrHeaders}</tr></thead>
                <tbody>${corrRows}</tbody>
              </table>
            </div>
          </div>
        </div>

        <div style="text-align:center">
          <div style="display:inline-block;border:1px solid var(--doc-hairline);padding:5px 16px;font-size:11px;color:var(--doc-ink-muted)">
            Generated by <strong style="color:var(--doc-ink)">Varense</strong> · Free tier · ${date}
          </div>
        </div>

      </div>

      <div class="vr-footer" style="background:var(--doc-bg);border-top:1px solid var(--doc-hairline);display:flex;justify-content:space-between;align-items:center">
        <div style="font-size:10px;color:var(--doc-ink-muted);line-height:1.5;max-width:460px">
          This report is for educational purposes only and does not constitute financial advice.
          Past performance does not guarantee future results. Always do your own research.
        </div>
        <div style="font-size:13px;font-weight:700;color:var(--doc-ink);font-family:var(--doc-font-primary)">varense.vercel.app</div>
      </div>
    `
  }

  // Checked before building the (large) report string at all — nothing to
  // do with it if the pop-up never opens. A blocked pop-up returns null
  // rather than throwing, so this is the one thing that used to fail
  // silently: win.document.write below would have thrown on a null win,
  // visible only in the console, with no on-screen sign anything went wrong.
  const win = window.open('', '_blank', 'width=1100,height=900')
  if (!win) {
    showExportMessage('Your browser blocked the report pop-up. Please allow pop-ups for this site, then try Export PDF again.')
    return
  }

  const reportHTML = buildReportHTML()
  const colorVars = tokensToCSSVars(tokens.color, ['color']).join('\n              ')
  win.document.write(`
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>Varense Portfolio Report</title>
          <link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>
          <link href="https://cdn.jsdelivr.net/npm/@fontsource/geist-sans@5.2.5/400.css" rel="stylesheet">
          <link href="https://cdn.jsdelivr.net/npm/@fontsource/geist-sans@5.2.5/500.css" rel="stylesheet">
          <link href="https://cdn.jsdelivr.net/npm/@fontsource/geist-sans@5.2.5/600.css" rel="stylesheet">
          <link href="https://cdn.jsdelivr.net/npm/@fontsource/geist-mono@5.2.5/400.css" rel="stylesheet">
          <link href="https://cdn.jsdelivr.net/npm/@fontsource/geist-mono@5.2.5/500.css" rel="stylesheet">
          <style>
            /* This popup is a separate document (window.open + document.write) that
               does NOT inherit the app's index.css, so it can't just reference the
               app's own CSS variables either. These --color-* lines are generated at
               runtime from design/tokens.json by tokensToCSSVars above (mirroring
               scripts/tokens-to-css.mjs's own build-time walk) — not hand-copied, so
               a token change in that file reaches this report automatically. */
            :root {
              ${colorVars}

              /* A light, print-friendly read of the same brand: no light mode exists
                 in design/tokens.json (colorMode: "dark-only"), so these two reuse
                 existing dark-theme token VALUES in an inverted role — the darkest
                 canvas tone (color.bg.base) as ink on paper, the lightest text tone
                 (color.text.primary) as the page itself — rather than inventing new,
                 undocumented hex. Everything else below is a direct var(--color-*)
                 reference, nothing new to keep in sync. */
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
              --doc-font-mono: 'Geist Mono', ui-monospace, 'SF Mono', Menlo, monospace;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { background: var(--doc-bg); font-family: var(--doc-font-primary); color: var(--doc-ink); }

            /* DESIGN.md Core Rules, applied to this report: zero border-radius
               everywhere (no exceptions), no box-shadows. */
            [class^="vr-"], .vr-metric-card, table, th, td, span, div, button { border-radius: 0 !important; }

            /* Every section below (a caption label plus its content card(s)) keeps
               itself whole across a page break, print or screen — the "Downside
               Risk"/"Correlation Matrix" headings stranded at the bottom of page 1
               with their content on page 2 is exactly what this prevents: the whole
               section now moves to the next page together instead of splitting. */
            .vr-section { break-inside: avoid; page-break-inside: avoid; }

            .vr-header { padding: 24px 32px; }
            .vr-body { padding: 22px 32px; gap: 18px; }
            .vr-footer { padding: 12px 32px; }
            .vr-label { margin-bottom: 8px; }
            .vr-card { padding: 16px; }
            .vr-chart-card { padding: 16px; }
            .vr-chart-meta { margin-bottom: 8px; }
            /* flex-wrap, not a fixed grid-template-columns — same reasoning as
               .dashboard-grid__row1/row2 in index.css: Beta/Alpha make this 7 or 9
               cards depending on the run, and a fixed column count leaves an odd
               remainder (Alpha) stranded alone on the last row instead of filling
               it. flex-grow on every card fills whatever's left on its own row,
               for any count, the same fix Dashboard's own metric rows already use. */
            .vr-metric-grid { gap: 7px; }
            .vr-metric-card { flex: 1 1 130px; min-width: 0; padding: 10px 12px; }
            .vr-metric-value { font-size: 18px; }
            .vr-grid2 { grid-template-columns: 1fr 1fr; gap: 16px; }
            .vr-alloc-row { margin-bottom: 10px; }
            .vr-mc-item { padding: 7px 10px; }
            .vr-risk-row { padding: 7px 0; }
            .vr-corr-table { border-spacing: 3px; }
            .vr-corr-head { padding: 0 4px 6px; }
            .vr-corr-cell { padding: 6px 4px; }

            /* Readable on a phone before printing — no pinch-zoom needed to read the
               report or reach the "Got it" button below. The viewport meta tag above
               makes these CSS px match device px; this collapses the desktop-width
               layout (two 2-column rows, a wide header) to something that fits a
               384-430px-wide phone screen on its own — the metric grid already
               reflows itself via flex-wrap above, no separate breakpoint needed. */
            @media screen and (max-width: 720px) {
              .vr-header { padding: 16px 18px; flex-wrap: wrap; gap: 12px; }
              .vr-header-right { text-align: left; }
              .vr-body { padding: 14px 16px; gap: 16px; }
              .vr-footer { padding: 12px 16px; flex-wrap: wrap; gap: 8px; }
              .vr-grid2 { grid-template-columns: 1fr; gap: 16px; }
            }

            /* Aims for one landscape A4 page: tighter padding/gaps/type than the
               on-screen version, print-only, so the on-screen phone layout above
               stays roomy. .vr-section's break-inside:avoid (above) is the fallback
               if a report with 5 tickers genuinely still doesn't fit — sections stay
               whole and move to page 2 together rather than splitting mid-card. */
            @media print {
              #modal-overlay { display: none !important; }
              @page { margin: 0; size: A4 landscape; }
              body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
              .vr-header { padding: 12px 24px; }
              .vr-body { padding: 10px 20px; gap: 8px; }
              .vr-footer { padding: 7px 20px; }
              .vr-label { margin-bottom: 4px; }
              .vr-card { padding: 8px 10px; }
              .vr-chart-card { padding: 6px 10px; }
              .vr-chart-meta { margin-bottom: 4px; }
              .vr-chart-svg { height: 72px; }
              .vr-metric-grid { gap: 5px; }
              .vr-metric-card { flex-basis: 100px; padding: 6px 8px; }
              .vr-metric-value { font-size: 14px; }
              .vr-grid2 { gap: 10px; }
              .vr-alloc-row { margin-bottom: 5px; }
              .vr-mc-item { padding: 4px 7px; }
              .vr-risk-row { padding: 4px 0; }
              .vr-corr-table { border-spacing: 2px; }
              .vr-corr-head { padding: 0 3px 3px; }
              .vr-corr-cell { padding: 3px 2px; }
            }
          </style>
        </head>
        <body>
          <!-- Instruction modal — platform-neutral: iPhone/Android print UIs have no
               "More settings" panel, so the old 4-step desktop-Chrome walkthrough
               didn't apply to either. "Background graphics" and "Landscape" are
               dropped entirely — both are already forced by this page's own CSS
               above (print-color-adjust:exact; @page size:A4 landscape) regardless
               of what's checked in the dialog, confirmed against real print output. -->
          <div id="modal-overlay" style="position:fixed;inset:0;background:rgba(0,0,0,0.4);z-index:9999;display:flex;align-items:center;justify-content:center;padding:16px;">
            <div style="background:var(--doc-bg);border:1px solid var(--doc-hairline);padding:24px 28px;max-width:420px;width:100%;font-family:var(--doc-font-primary);">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
                <svg viewBox="3 9 58 41" width="24" height="17" fill="none" role="img" aria-label="Varense">
                  <path d="M5.19,44.5 A27,30 0 1 1 58.81,44.5" fill="none" stroke="var(--doc-ink)" stroke-width="3.5" stroke-linejoin="miter" stroke-linecap="butt"></path>
                  <path d="M5,48 L16,48 L23,41.5 L26.5,44.5 L36,34 L48,48 L59,48" fill="none" stroke="var(--doc-ink)" stroke-width="3.5" stroke-linejoin="miter" stroke-linecap="butt"></path>
                </svg>
                <div style="font-size:16px;font-weight:700;color:var(--doc-ink);">Save your PDF report</div>
              </div>
              <div style="font-size:13px;color:var(--doc-ink);line-height:1.5;margin-bottom:20px;">
                In the print dialog, choose <strong>Save as PDF</strong>.
              </div>
              <button onclick="document.getElementById('modal-overlay').style.display='none';window.print();"
                style="width:100%;min-height:46px;padding:12px;background:var(--doc-ink);color:var(--doc-bg);border:none;font-size:14px;font-weight:700;cursor:pointer;letter-spacing:0.03em;font-family:var(--doc-font-primary);">
                Got it - open print dialog
              </button>
            </div>
          </div>

          <div>${reportHTML}</div>
          <script>
            // No auto-print — user clicks the button in the modal
          </script>
        </body>
      </html>
    `)
    win.document.close()
}
