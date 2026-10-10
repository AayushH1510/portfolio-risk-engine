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
// fmt/fmtN/fmtD, tokensToCSSVars, docTokenCSS, FONT_LINKS, MARK_PATHS and
// openReportPopup all moved to lib/pdfReportShared.js once a second report
// (lib/exportFullReportPdf.js) needed the identical token-walking and
// pop-up-boilerplate logic — a hand-copied second version here would have
// been exactly the kind of drift this file's own tokensToCSSVars was
// originally written to stop, just with two call sites instead of one. See
// that module's own header comment for the full reasoning.
import { fmt, fmtN, fmtD, docTokenCSS, FONT_LINKS, MARK_PATHS, openReportPopup } from './pdfReportShared'
import { buildAreaSparkline } from './reportCharts'

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

    // Growth chart as SVG sparkline — buildAreaSparkline (lib/reportCharts.js)
    // is this exact chart, generalised once the full report needed the same
    // scaling math for its own Dashboard/drawdown sections.
    const cumVals = cum.values
    const growthSparkline = buildAreaSparkline(cumVals, { gradientId: 'sg1', className: 'vr-chart-svg' })

    return `
      <div class="vr-header" style="background:var(--doc-bg);border-bottom:1px solid var(--doc-hairline);display:flex;justify-content:space-between;align-items:flex-start">
        <div>
          <div class="vr-brand" style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
            <svg viewBox="3 9 58 41" width="26" height="18.4" fill="none" role="img" aria-label="Varense">
              ${MARK_PATHS}
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
            ${growthSparkline}
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

  // openReportPopup (lib/pdfReportShared.js) does the window.open + null
  // check + pop-up-blocked message — shared with the full report so both
  // fail the same way, checked before building either's (large) HTML string
  // at all, since there's nothing to do with it if the pop-up never opens.
  const win = openReportPopup()
  if (!win) return

  const reportHTML = buildReportHTML()
  win.document.write(`
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>Varense Portfolio Report</title>
          ${FONT_LINKS}
          <style>
            /* This popup is a separate document (window.open + document.write) that
               does NOT inherit the app's index.css, so it can't just reference the
               app's own CSS variables either. docTokenCSS() (lib/pdfReportShared.js)
               generates the color and doc token lines at runtime from
               design/tokens.json (mirroring scripts/tokens-to-css.mjs's own
               build-time walk) — not hand-copied, so a token change in that file
               reaches this report automatically, and the full report (which needs
               the identical block) can't drift from it either. */
            :root {
              ${docTokenCSS()}
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
                  ${MARK_PATHS}
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
