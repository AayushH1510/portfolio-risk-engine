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

const fmt  = v => v != null ? `${(v * 100).toFixed(1)}%` : 'N/A'
const fmtN = v => v != null ? v.toFixed(2) : 'N/A'
const fmtD = v => v != null ? `$${Math.abs(v).toLocaleString('en-US', { maximumFractionDigits:0 })}` : 'N/A'

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

    const allocColors = ['var(--accent-dark)','var(--accent)','var(--signal-positive-soft)','var(--chart-mint)','var(--accent-light)']

    const metricCards = metricRows.map(m => `
      <div class="vr-metric-card" style="background:var(--white);border-radius:var(--radius-7);border-top:3px solid ${m.good ? 'var(--accent)':'var(--negative)'}">
        <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:var(--text-secondary);margin-bottom:4px">${m.label}</div>
        <div class="vr-metric-value" style="font-weight:700;font-family:var(--font-mono);color:${m.good ? 'var(--accent-dark)':'var(--report-red)'}">${m.val}</div>
      </div>
    `).join('')

    const tickerPills = tickers.map((t, i) => `
      <span style="display:inline-block;background:rgba(var(--signal-positive-rgb),0.15);border:1px solid rgba(var(--signal-positive-rgb),0.3);border-radius:var(--radius-4);padding:2px 9px;font-size:11px;font-weight:700;color:var(--accent);font-family:var(--font-mono);margin-right:5px">${t} ${Math.round(weights[i]*100)}%</span>
    `).join('')

    const allocBars = tickers.map((t, i) => `
      <div class="vr-alloc-row">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
          <span style="font-weight:700;font-family:var(--font-mono);font-size:13px">${t}</span>
          <span style="font-weight:700;color:var(--accent-dark);font-size:13px">${Math.round(weights[i]*100)}%</span>
        </div>
        <div style="height:6px;background:var(--report-track);border-radius:var(--radius-3)">
          <div style="height:6px;width:${weights[i]*100}%;background:${allocColors[i % allocColors.length]};border-radius:var(--radius-3)"></div>
        </div>
      </div>
    `).join('')

    const mcItems = [
      { label:'Median',        val: fmtD(mc.p50_final),                       color:'var(--accent-dark)' },
      { label:'Good year',     val: fmtD(mc.p95_final),                       color:'var(--accent)' },
      { label:'Bad year',      val: fmtD(mc.p5_final),                        color:'var(--report-red)' },
      { label:'Profit chance', val: `${Math.round(mc.prob_profit*100)}%`,      color: mc.prob_profit > 0.6 ? 'var(--accent-dark)':'var(--report-amber)' },
    ].map(m => `
      <div class="vr-mc-item" style="background:var(--report-card-alt);border-radius:var(--radius-5)">
        <div style="font-size:9px;color:var(--text-secondary);font-weight:700;text-transform:uppercase;margin-bottom:2px">${m.label}</div>
        <div style="font-size:14px;font-weight:700;color:${m.color};font-family:var(--font-mono)">${m.val}</div>
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
      <div class="vr-risk-row" style="display:flex;justify-content:space-between;border-bottom:1px solid var(--report-bg)">
        <span style="color:var(--text-muted);font-size:12px">${r.label}</span>
        <div style="text-align:right">
          <div style="font-weight:700;color:var(--report-red);font-family:var(--font-mono);font-size:12px">${r.a}</div>
          <div style="font-size:10px;color:var(--text-secondary)">${r.b}</div>
        </div>
      </div>
    `).join('')

    const corrHeaders = corr.tickers.map(t => `<th class="vr-corr-head" style="font-size:10px;color:var(--text-secondary);font-family:var(--font-mono);text-align:center">${t}</th>`).join('')
    const corrRows = corr.tickers.map((row, ri) => {
      const cells = corr.values[ri].map((val, ci) => {
        const bg = val >= 0.7 ? 'var(--report-heat-high)' : val >= 0.4 ? 'var(--report-heat-mid)' : 'var(--report-heat-low)'
        return `<td class="vr-corr-cell" style="text-align:center;font-size:11px;font-family:var(--font-mono);font-weight:600;border-radius:var(--radius-3);background:${bg}">${val.toFixed(2)}</td>`
      }).join('')
      return `<tr><td style="font-size:10px;color:var(--text-secondary);font-family:var(--font-mono);text-align:right;padding-right:8px">${row}</td>${cells}</tr>`
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
      <div class="vr-header" style="background:var(--card);display:flex;justify-content:space-between;align-items:flex-start">
        <div>
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
            <div style="width:32px;height:32px;background:var(--accent);border-radius:var(--radius-7);display:flex;align-items:center;justify-content:center;font-weight:900;color:var(--card);font-size:16px">V</div>
            <div>
              <div style="font-size:20px;font-weight:700;color:var(--white);letter-spacing:-0.02em">varense</div>
              <div style="font-size:10px;color:var(--accent);letter-spacing:0.05em">variance, made sense of</div>
            </div>
          </div>
          <div>${tickerPills}</div>
        </div>
        <div class="vr-header-right" style="text-align:right">
          <div style="font-size:18px;font-weight:700;color:var(--white)">Varense Portfolio Report</div>
          <div style="font-size:12px;color:var(--text-secondary);margin-top:4px">${date}</div>
          <div style="font-size:11px;color:var(--text-muted);margin-top:5px;font-family:var(--font-mono)">${d.period.start} - ${d.period.end} · ${d.period.n_days} days</div>
        </div>
      </div>

      <div class="vr-body" style="display:flex;flex-direction:column;background:var(--report-bg)">

        <div class="vr-section">
          <div class="vr-label" style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted)">Key Metrics</div>
          <div class="vr-metric-grid" style="display:grid">${metricCards}</div>
        </div>

        <div class="vr-section">
          <div class="vr-label" style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted)">Portfolio Growth</div>
          <div class="vr-card vr-chart-card" style="background:var(--white);border-radius:var(--radius-sm)">
            <div class="vr-chart-meta" style="display:flex;justify-content:space-between;font-size:12px">
              <span style="color:var(--text-secondary)">Cumulative return from ${d.period.start}</span>
              <span style="font-weight:700;color:${d.annualised_return > 0 ? 'var(--accent-dark)':'var(--report-red)'};font-family:var(--font-mono)">${fmt(cumVals[cumVals.length-1])} total</span>
            </div>
            <svg class="vr-chart-svg" width="100%" height="${svgH}" viewBox="0 0 ${svgW} ${svgH}" preserveAspectRatio="none">
              <defs>
                <linearGradient id="sg1" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="var(--accent)" stop-opacity="0.2"/>
                  <stop offset="100%" stop-color="var(--accent)" stop-opacity="0"/>
                </linearGradient>
              </defs>
              ${zeroY > 0 && zeroY < svgH ? `<line x1="0" y1="${zeroY}" x2="${svgW}" y2="${zeroY}" stroke="var(--report-gridline)" stroke-dasharray="6,4" stroke-width="1"/>` : ''}
              <polygon points="${points} ${svgW},${svgH} 0,${svgH}" fill="url(#sg1)"/>
              <polyline points="${points}" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
            </svg>
          </div>
        </div>

        <div class="vr-grid2" style="display:grid">
          <div class="vr-section">
            <div class="vr-label" style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted)">Allocation</div>
            <div class="vr-card" style="background:var(--white);border-radius:var(--radius-sm)">${allocBars}</div>
          </div>
          <div class="vr-section">
            <div class="vr-label" style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted)">Monte Carlo - 1 Year</div>
            <div class="vr-card" style="background:var(--white);border-radius:var(--radius-sm)">
              <div class="vr-mc-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:6px">${mcItems}</div>
              <div style="margin-top:10px;font-size:10px;color:var(--text-secondary)">Based on ${mc.n_simulations?.toLocaleString() || '1,000'} simulations. Median: ${fmtD(mc.p50_final)} · Good: ${fmtD(mc.p95_final)} · Bad: ${fmtD(mc.p5_final)}</div>
            </div>
          </div>
        </div>

        <div class="vr-grid2" style="display:grid">
          <div class="vr-section">
            <div class="vr-label" style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted)">Downside Risk</div>
            <div class="vr-card" style="background:var(--white);border-radius:var(--radius-sm)">${riskRows}</div>
          </div>
          <div class="vr-section">
            <div class="vr-label" style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted)">Correlation Matrix</div>
            <div class="vr-card" style="background:var(--white);border-radius:var(--radius-sm)">
              <table class="vr-corr-table" style="width:100%;border-collapse:separate">
                <thead><tr><th style="width:50px"></th>${corrHeaders}</tr></thead>
                <tbody>${corrRows}</tbody>
              </table>
            </div>
          </div>
        </div>

        <div style="text-align:center">
          <div style="display:inline-block;border:1px solid var(--report-border);border-radius:var(--radius-5);padding:5px 16px;font-size:11px;color:var(--text-secondary)">
            Generated by <strong style="color:var(--accent-dark)">varense</strong> · Free tier · ${date}
          </div>
        </div>

      </div>

      <div class="vr-footer" style="background:var(--card);display:flex;justify-content:space-between;align-items:center">
        <div style="font-size:10px;color:var(--text-muted);line-height:1.5;max-width:460px">
          This report is for educational purposes only and does not constitute financial advice.
          Past performance does not guarantee future results. Always do your own research.
        </div>
        <div style="font-size:13px;font-weight:700;color:var(--accent)">varense.com</div>
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
  win.document.write(`
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>Varense Portfolio Report</title>
          <link rel="preconnect" href="https://fonts.googleapis.com">
          <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
          <style>
            /* This popup is a separate document (window.open + document.write) that
               does NOT inherit the app's index.css, so the design tokens it uses below
               are duplicated here — values must stay in sync with index.css by hand. */
            :root {
              --accent-dark: #2d6a4f; --accent: #52b788; --signal-positive-soft: #74c69d;
              --chart-mint: #95d5b2; --accent-light: #b7e4c7; --report-text-dark: #1a2a1a;
              --card: #1e2420; --white: #ffffff; --text-muted: #5a7a5a; --text-secondary: #8aaa8a;
              --negative: #e05c5c; --report-red: #c0392b; --report-bg: #f0f4f0;
              --report-track: #e8ede8; --report-gridline: #e0e8e0; --report-heat-high: #fdd;
              --report-heat-mid: #fef3d0; --report-heat-low: #e8f5ee; --report-card-alt: #f8faf8;
              --report-card-soft: #f5faf5; --report-amber: #b7791f; --report-border: #c8d8c8;
              --black-rgb: 0,0,0; --signal-positive-rgb: 82,183,136;
              --font-mono: monospace; --font-report: Inter, sans-serif;
              --radius-14: 14px; --radius-3: 3px; --radius-4: 4px; --radius-5: 5px;
              --radius-7: 7px; --radius-sm: 8px; --radius-full: 50%;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { background: var(--report-bg); font-family: var(--font-report); }

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
            .vr-metric-grid { grid-template-columns: repeat(5, 1fr); gap: 7px; }
            .vr-metric-card { padding: 10px 12px; }
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
               layout (5-column metric grid, two 2-column rows, a wide header) to
               something that fits a 384-430px-wide phone screen on its own. */
            @media screen and (max-width: 720px) {
              .vr-header { padding: 16px 18px; flex-wrap: wrap; gap: 12px; }
              .vr-header-right { text-align: left; }
              .vr-body { padding: 14px 16px; gap: 16px; }
              .vr-footer { padding: 12px 16px; flex-wrap: wrap; gap: 8px; }
              .vr-metric-grid { grid-template-columns: repeat(2, 1fr); gap: 8px; }
              .vr-grid2 { grid-template-columns: 1fr; gap: 16px; }
            }
            @media screen and (min-width: 721px) and (max-width: 980px) {
              .vr-metric-grid { grid-template-columns: repeat(3, 1fr); }
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
              .vr-metric-card { padding: 6px 8px; }
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
          <div id="modal-overlay" style="position:fixed;inset:0;background:rgba(var(--black-rgb),0.55);z-index:9999;display:flex;align-items:center;justify-content:center;padding:16px;">
            <div style="background:var(--white);border-radius:var(--radius-14);padding:24px 28px;max-width:420px;width:100%;box-shadow:0 24px 64px rgba(var(--black-rgb),0.3);font-family:var(--font-report);">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
                <div style="width:34px;height:34px;background:var(--accent-dark);border-radius:var(--radius-sm);display:flex;align-items:center;justify-content:center;font-weight:900;color:var(--white);font-size:16px;flex-shrink:0;">V</div>
                <div style="font-size:16px;font-weight:700;color:var(--report-text-dark);">Save your PDF report</div>
              </div>
              <div style="font-size:13px;color:var(--report-text-dark);line-height:1.5;margin-bottom:20px;">
                In the print dialog, choose <strong>Save as PDF</strong>.
              </div>
              <button onclick="document.getElementById('modal-overlay').style.display='none';window.print();"
                style="width:100%;min-height:46px;padding:12px;background:var(--accent-dark);color:var(--white);border:none;border-radius:var(--radius-sm);font-size:14px;font-weight:700;cursor:pointer;letter-spacing:0.03em;">
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
