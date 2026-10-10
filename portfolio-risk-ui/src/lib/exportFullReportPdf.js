// The Full report PDF — a multi-page, portrait A4 companion to the one-page
// Summary PDF (lib/exportPdfReport.js), with a section per analysis tab in
// the app's own tab order (App.jsx's TABS array), each starting on a fresh
// page: cover -> Dashboard -> Risk Analysis -> Monte Carlo -> Efficient
// Frontier -> Valuation -> Compare (only if run) -> Backtest -> methodology/
// disclosure. Learn is skipped — it has no portfolio data of its own.
//
// Same split-into-its-own-lazy-module reasoning as exportPdfReport.js (see
// that file's own header comment): a large, click-only function that both
// ExportMenu.jsx and ExportPDF.jsx dynamically import rather than bundling
// into every /app load. Shares its design tokens, formatters and pop-up
// boilerplate with the Summary report via lib/pdfReportShared.js, and its
// chart-drawing with both the Summary report and each other via
// lib/reportCharts.js — see those modules' own header comments.
//
// Every section renders strictly from state already sitting in React when
// Export is clicked — nothing here calls the API. `stressTestData` and
// `valuationData` are each a lifted copy of a result that tab's own
// component (StressTest.jsx, Valuation.jsx) already fetched for its own
// purposes and reported upward via a callback (see App.jsx) — null until
// that tab has actually been opened and loaded since the current portfolio
// was run, in which case the section shows an actionable placeholder
// ("Open the ... tab before exporting...") rather than silently omitting
// itself. That's a different case from a section that genuinely needs the
// user to run something (Compare, the heavy-tier Monte Carlo/Frontier/
// Backtest), which keeps the plainer "Not run in this session" wording.
import { getBenchmarkLabel, getBenchmarkPhrase } from './benchmarks'
import { winner } from './compareMetrics'
import { fmt, fmtN, fmtD, docTokenCSS, FONT_LINKS, MARK_PATHS, openReportPopup } from './pdfReportShared'
import {
  buildAreaSparkline, buildMultiLineSVG, buildMonteCarloBandSVG,
  buildFrontierScatterSVG, buildHistogramSVG,
} from './reportCharts'

// ── Small shared builders, reused across every section below ──────────────

function pageHeader(title) {
  return `
    <div class="fr-page-header">
      <div class="fr-page-header-brand">
        <svg viewBox="3 9 58 41" width="15" height="10.6" fill="none" role="img" aria-label="Varense">${MARK_PATHS}</svg>
        Varense · Full Report
      </div>
      <div class="fr-page-header-title">${title}</div>
    </div>
  `
}

function sectionLabel(text) {
  return `<div class="fr-section-label">${text}</div>`
}

// Wraps a section label (and/or why-box) together with the content it
// introduces as one break-inside:avoid unit (.fr-group, defined in the
// document's own <style> below). Without this, each individual .fr-card
// already refuses to split internally, but the LABEL above it could still
// be left stranded at the bottom of one page with its card pushed to the
// next — exactly the "Downside Risk" heading separated from its own
// content that the Summary report's own break-inside fix was written to
// prevent, just one level up (a card's own heading, not a whole section's).
function group(...parts) {
  return `<div class="fr-group">${parts.join('')}</div>`
}

// The tab's own "why this matters" InsightBox copy, reused verbatim where
// one exists (RiskAnalysis.jsx, MonteCarlo.jsx, Frontier.jsx, Valuation.jsx,
// Backtest.jsx, CompareWrapper.jsx each have one) so this report explains
// itself using the same language the app already does, not a second,
// independently-drifting paraphrase.
function whyBox(text) {
  return `<div class="fr-why">${text}</div>`
}

// Any section whose figures aren't available to this export (never run
// this session, or genuinely not retained in shared state — see the
// module comment above) gets this instead of being silently dropped, per
// the brief.
function notRunBox(text) {
  return `<div class="fr-notrun">${text || 'Not run in this session.'}</div>`
}

// Flex-wrap, not a fixed grid — same "no stranded cards" reasoning as the
// Summary report's own .vr-metric-grid (itself ported from Dashboard's
// .dashboard-grid__row1/row2, see that CSS rule's own comment in
// index.css): a card count that varies with what data is available (Beta/
// Alpha only when beta_alpha is present) must never leave a lone remainder
// stranded narrow on its own row.
function metricGrid(items) {
  return `<div class="fr-metric-grid">${items.map(m => `
    <div class="fr-metric-card"${m.good === undefined ? '' : m.good ? '' : ' style="border-top:2px solid var(--color-signal-negative);"'}>
      <div class="fr-metric-card-label">${m.label}</div>
      <div class="fr-metric-card-value" style="color:${m.good === undefined ? 'var(--doc-ink)' : m.good ? 'var(--color-accent-mint)' : 'var(--color-signal-negative)'}">${m.val}</div>
    </div>
  `).join('')}</div>`
}

function allocRows(items, colorFn) {
  return items.map(({ label, w }) => `
    <div class="fr-alloc-row">
      <div style="display:flex;justify-content:space-between;margin-bottom:3px">
        <span style="font-size:11px;font-family:var(--doc-font-mono);color:var(--doc-ink)">${label}</span>
        <span style="font-size:11px;font-weight:700;color:${colorFn(w)}">${Math.round(w * 100)}%</span>
      </div>
      <div style="height:5px;background:var(--doc-track)"><div style="height:5px;width:${Math.max(w, 0) * 100}%;background:${colorFn(w)}"></div></div>
    </div>
  `).join('')
}

// ── Cover ───────────────────────────────────────────────────────────────

function buildCoverSection({ data, tickers, weights, portfolioValue, date }) {
  const tickerPills = tickers.map((t, i) => `<span class="fr-pill">${t} ${Math.round(weights[i] * 100)}%</span>`).join('')
  const benchmarkLabel = data.benchmark ? getBenchmarkLabel(data.benchmark) : null

  return `
    <section class="fr-page fr-cover">
      <div class="fr-cover-inner">
        <div class="fr-cover-mark">
          <svg viewBox="3 9 58 41" width="42" height="29.8" fill="none" role="img" aria-label="Varense">${MARK_PATHS}</svg>
          <span class="fr-cover-wordmark">Varense</span>
        </div>
        <div>
          <div class="fr-cover-title">Full Portfolio Report</div>
          <div class="fr-cover-date">${date}</div>
        </div>
        <div class="fr-cover-pills">${tickerPills}</div>
        <div class="fr-cover-facts">
          <div>
            <div class="fr-cover-fact-label">Period</div>
            <div class="fr-cover-fact-val">${data.period.start} – ${data.period.end}</div>
          </div>
          <div>
            <div class="fr-cover-fact-label">Portfolio value</div>
            <div class="fr-cover-fact-val">${fmtD(portfolioValue)}</div>
          </div>
          ${benchmarkLabel ? `
          <div>
            <div class="fr-cover-fact-label">Benchmark</div>
            <div class="fr-cover-fact-val">${benchmarkLabel}</div>
          </div>` : ''}
        </div>
      </div>
    </section>
  `
}

// ── Dashboard ───────────────────────────────────────────────────────────

function buildDashboardSection({ data, sectorData }) {
  const { annualised_return: ret, annualised_volatility: vol, sharpe_ratio: sharpe,
          sortino_ratio: sortino, max_drawdown: dd, var_cvar: vc, beta_alpha: ba,
          diversification_score: divScore, cumulative_returns: cum } = data

  const metrics = [
    { label: 'Annual Return', val: fmt(ret), good: ret > 0 },
    { label: 'Volatility', val: fmt(vol), good: vol < 0.25 },
    { label: 'Sharpe Ratio', val: fmtN(sharpe), good: sharpe > 1 },
    { label: 'Sortino Ratio', val: fmtN(sortino), good: sortino > 1 },
    { label: 'Max Drawdown', val: fmt(dd), good: false },
    { label: 'VaR 95%', val: fmt(vc.var_pct), good: false },
    { label: 'CVaR 95%', val: fmt(vc.cvar_pct), good: false },
    ...(ba ? [
      { label: 'Beta', val: fmtN(ba.beta), good: ba.beta < 1.2 },
      { label: 'Alpha', val: fmt(ba.alpha), good: ba.alpha > 0 },
    ] : []),
  ]

  const growthChart = buildAreaSparkline(cum.values, { gradientId: 'fg1' })

  const divBlock = divScore ? `
    <div class="fr-card">
      <div class="fr-metric-card-label">Diversification score</div>
      <div style="display:flex;align-items:baseline;gap:10px;margin-top:6px">
        <div style="font-size:26px;font-weight:700;font-family:var(--doc-font-mono);color:${divScore.score >= 70 ? 'var(--color-accent-mint)' : divScore.score >= 40 ? 'var(--color-signal-warning)' : 'var(--color-signal-negative)'}">${divScore.score}/100</div>
        <div style="font-size:11px;color:var(--doc-ink-secondary);line-height:1.5">${divScore.label}<br/>avg pairwise corr ${divScore.avg_pairwise_corr.toFixed(2)}</div>
      </div>
    </div>` : ''

  const sectorBlock = (sectorData && sectorData.length) ? `
    <div class="fr-card">
      <div class="fr-metric-card-label" style="margin-bottom:8px">Sector exposure</div>
      ${allocRows(sectorData.map(r => ({ label: r.sector, w: r.weight })), () => 'var(--doc-ink-secondary)')}
    </div>` : ''

  const extras = [divBlock, sectorBlock].filter(Boolean)
  const extrasHTML = extras.length === 2 ? `<div class="fr-grid2">${extras.join('')}</div>` : extras.join('')

  return `
    <section class="fr-page fr-pagebreak">
      ${pageHeader('Dashboard')}
      ${whyBox('A single-page view of how your portfolio has performed and how much risk it carried getting there.')}
      ${group(sectionLabel('Key Metrics'), metricGrid(metrics))}
      ${group(sectionLabel('Portfolio Growth'), `
      <div class="fr-card">
        <div class="fr-chart-meta">
          <span>Cumulative return from ${data.period.start}</span>
          <span style="font-weight:700;color:${ret > 0 ? 'var(--color-accent-mint)' : 'var(--color-signal-negative)'}">${fmt(cum.values[cum.values.length - 1])} total</span>
        </div>
        ${growthChart}
      </div>
      `)}
      ${extrasHTML}
    </section>
  `
}

// ── Risk Analysis ───────────────────────────────────────────────────────

// A single stress-test scenario card — mirrors StressTest.jsx's own
// ScenarioCard (same fields, same "no historical data"/excluded-tickers
// handling), simplified for print: no loading/retry state (this only ever
// renders once stressTestData is already in hand).
function buildStressCard({ name, period, portfolio_return, worst_day, recovery_days, excluded_tickers }, portfolioValue) {
  if (portfolio_return == null) {
    return `
      <div class="fr-card">
        <div class="fr-metric-card-label">${name}</div>
        <div style="font-size:10px;color:var(--doc-ink-muted);margin-top:2px;font-family:var(--doc-font-mono)">${period}</div>
        <div style="font-size:11px;color:var(--doc-ink-muted);margin-top:10px">No historical data available for this portfolio in this window.</div>
      </div>`
  }
  const lossColor = portfolio_return < -0.3 ? 'var(--color-signal-negative)' : portfolio_return < -0.1 ? 'var(--color-signal-warning)' : 'var(--color-accent-mint)'
  const dollarLoss = portfolioValue != null ? fmtD(portfolioValue * Math.abs(portfolio_return)) : null
  return `
    <div class="fr-card">
      <div class="fr-metric-card-label">${name}</div>
      <div style="font-size:10px;color:var(--doc-ink-muted);margin-top:2px;font-family:var(--doc-font-mono)">${period}</div>
      <div style="font-size:20px;font-weight:700;font-family:var(--doc-font-mono);color:${lossColor};margin-top:10px">${fmt(portfolio_return)}</div>
      ${dollarLoss ? `<div style="font-size:10px;color:var(--doc-ink-muted);font-family:var(--doc-font-mono)">-${dollarLoss}</div>` : ''}
      <div style="display:flex;justify-content:space-between;margin-top:10px;font-size:10px;color:var(--doc-ink-muted)">
        <span>Worst day <strong style="color:var(--color-signal-negative);font-family:var(--doc-font-mono)">${fmt(worst_day)}</strong></span>
        <span>Recovery <strong style="color:var(--doc-ink);font-family:var(--doc-font-mono)">${recovery_days == null ? 'Not yet' : `${recovery_days}d`}</strong></span>
      </div>
      ${excluded_tickers?.length ? `<div style="font-size:9px;color:var(--doc-ink-muted);margin-top:8px">Excludes ${excluded_tickers.join(', ')}</div>` : ''}
    </div>`
}

function buildRiskSection({ data, stressTestData, portfolioValue }) {
  const { rolling_volatility: rv, rolling_sharpe: rs, annualised_volatility: vol,
          sharpe_ratio: sharpe, var_cvar: vc, var_cvar_99: vc99,
          correlation_matrix: corr, treynor_ratio: treynor, information_ratio: infoRatio,
          portfolio_returns, drawdown_series: dd } = data

  const recentVol = rv.values[rv.values.length - 1]
  const recentSharpe = rs.values[rs.values.length - 1]
  const riskRising = recentVol > vol
  const sharpeImproving = recentSharpe > sharpe

  const rollingBlock = `
    <div class="fr-grid2">
      <div class="fr-card">
        <div class="fr-metric-card-label">Rolling volatility</div>
        <div style="font-size:20px;font-weight:700;font-family:var(--doc-font-mono);color:${riskRising ? 'var(--color-signal-warning)' : 'var(--color-accent-mint)'};margin:4px 0">${fmt(recentVol)}</div>
        <div style="font-size:11px;color:var(--doc-ink-muted)">${riskRising ? 'Rising' : 'Falling'} vs long-run average of ${fmt(vol)}</div>
      </div>
      <div class="fr-card">
        <div class="fr-metric-card-label">Rolling Sharpe</div>
        <div style="font-size:20px;font-weight:700;font-family:var(--doc-font-mono);color:${sharpeImproving ? 'var(--color-accent-mint)' : 'var(--color-signal-warning)'};margin:4px 0">${fmtN(recentSharpe)}</div>
        <div style="font-size:11px;color:var(--doc-ink-muted)">${sharpeImproving ? 'Improving' : 'Weakening'} vs long-run average of ${fmtN(sharpe)}</div>
      </div>
    </div>`

  const varRows = [
    { label: 'VaR 95%', a: fmt(vc.var_pct), b: `${fmtD(vc.var_dollar)}/day` },
    { label: 'CVaR 95%', a: fmt(vc.cvar_pct), b: `${fmtD(vc.cvar_dollar)} avg` },
    { label: 'VaR 99%', a: fmt(vc99?.var_pct), b: `${fmtD(vc99?.var_dollar)}/day` },
    { label: 'CVaR 99%', a: fmt(vc99?.cvar_pct), b: `${fmtD(vc99?.cvar_dollar)} avg` },
  ].map(r => `
    <div class="fr-risk-row">
      <span style="color:var(--doc-ink-muted);font-size:12px">${r.label}</span>
      <div style="text-align:right">
        <div style="font-weight:700;color:var(--color-signal-negative);font-family:var(--doc-font-mono);font-size:12px">${r.a}</div>
        <div style="font-size:10px;color:var(--doc-ink-muted)">${r.b}</div>
      </div>
    </div>`).join('')

  const corrHeaders = corr.tickers.map(t => `<th class="fr-corr-head">${t}</th>`).join('')
  const corrRows = corr.tickers.map((row, ri) => {
    const cells = corr.values[ri].map(val => {
      const bg = val >= 0.7 ? 'var(--doc-heat-high)' : val >= 0.4 ? 'var(--doc-heat-mid)' : 'var(--doc-heat-low)'
      return `<td class="fr-corr-cell" style="background:${bg}">${val.toFixed(2)}</td>`
    }).join('')
    return `<tr><td class="fr-corr-rowlabel">${row}</td>${cells}</tr>`
  }).join('')

  const ratioRow = (treynor != null || infoRatio != null) ? `
    <div class="fr-grid2">
      ${treynor != null ? `<div class="fr-card"><div class="fr-metric-card-label">Treynor ratio</div><div style="font-size:18px;font-weight:700;font-family:var(--doc-font-mono);margin-top:4px;color:var(--doc-ink)">${fmtN(treynor)}</div></div>` : '<div></div>'}
      ${infoRatio != null ? `<div class="fr-card"><div class="fr-metric-card-label">Information ratio</div><div style="font-size:18px;font-weight:700;font-family:var(--doc-font-mono);margin-top:4px;color:var(--doc-ink)">${fmtN(infoRatio)}</div></div>` : '<div></div>'}
    </div>` : ''

  const drawdownChart = buildAreaSparkline(dd.values, { color: 'var(--color-signal-negative)', gradientId: 'fdd1' })

  const histogramBlock = portfolio_returns ? group(
    sectionLabel('Daily Returns Distribution'),
    whyBox("Every daily return the portfolio produced over the period. A narrow cluster near the middle means consistent day-to-day performance; a wide spread with bars reaching far to either side means bigger swings and occasional sharp outlier days — a portfolio that can look calm most days while still carrying real downside risk."),
    `<div class="fr-card">${buildHistogramSVG({ returns: portfolio_returns, varPct: vc.var_pct })}</div>`,
  ) : ''

  return `
    <section class="fr-page fr-pagebreak">
      ${pageHeader('Risk Analysis')}
      ${whyBox("Two portfolios can have the same return but very different risk. These metrics show whether you're being compensated fairly for the risk you're taking, not just how much you made.")}
      ${group(sectionLabel('Rolling Risk'), rollingBlock)}
      ${group(sectionLabel('Downside Risk & Correlation'), `
      <div class="fr-grid2">
        <div class="fr-card">${varRows}</div>
        <div class="fr-card">
          <table class="fr-corr-table"><thead><tr><th></th>${corrHeaders}</tr></thead><tbody>${corrRows}</tbody></table>
        </div>
      </div>
      `)}
      ${ratioRow}
      ${group(sectionLabel('Drawdown'), `<div class="fr-card">${drawdownChart}</div>`)}
      ${histogramBlock}
      ${group(
        sectionLabel('Stress Test'),
        whyBox('Backtested returns show how your portfolio performs in normal conditions. Stress tests show what happens when markets panic, the scenario most investors are least prepared for.'),
        stressTestData
          ? `<div class="fr-strategy-row">${stressTestData.map(s => buildStressCard(s, portfolioValue)).join('')}</div>`
          : notRunBox('Open the Risk Analysis tab before exporting to include stress tests.'),
      )}
    </section>
  `
}

// ── Monte Carlo ─────────────────────────────────────────────────────────

function buildMonteCarloSection({ data }) {
  const mc = data.monte_carlo

  if (!mc) {
    return `
      <section class="fr-page fr-pagebreak">
        ${pageHeader('Monte Carlo')}
        ${whyBox('Most tools show you one future. This shows three: the realistic downside, the expected path, and the upside if things go right.')}
        ${notRunBox()}
      </section>`
  }

  const metrics = [
    { label: 'Starting value', val: fmtD(mc.portfolio_value) },
    { label: 'Median outcome', val: fmtD(mc.p50_final) },
    { label: 'Chance of profit', val: `${Math.round(mc.prob_profit * 100)}%`, good: mc.prob_profit > 0.6 },
    { label: 'Chance of 10% loss', val: `${Math.round(mc.prob_loss_10pct * 100)}%`, good: mc.prob_loss_10pct < 0.25 },
  ]

  const chart = buildMonteCarloBandSVG({ p5: mc.percentile_5, p50: mc.percentile_50, p95: mc.percentile_95 })

  return `
    <section class="fr-page fr-pagebreak">
      ${pageHeader('Monte Carlo')}
      ${whyBox(`Most tools show you one future. This shows three, the realistic downside, the expected path, and the upside if things go right, based on ${mc.n_simulations.toLocaleString()} simulated years for this exact portfolio.`)}
      ${group(sectionLabel('Key Outcomes'), metricGrid(metrics))}
      ${group(sectionLabel('Simulated Paths — next 12 months'), `
      <div class="fr-card">
        <div class="fr-chart-legend">
          <span style="color:var(--color-signal-negative)">Bad (5th) · ${fmtD(mc.p5_final)}</span>
          <span style="color:var(--color-accent-mint)">Median · ${fmtD(mc.p50_final)}</span>
          <span style="color:var(--doc-ink-secondary)">Good (95th) · ${fmtD(mc.p95_final)}</span>
        </div>
        ${chart}
      </div>
      `)}
    </section>
  `
}

// ── Efficient Frontier ──────────────────────────────────────────────────

function buildFrontierSection({ data, tickers, weights }) {
  const ef = data.efficient_frontier
  const hasFrontier = !!(ef && ef.vols?.length && ef.returns?.length)

  if (!hasFrontier) {
    return `
      <section class="fr-page fr-pagebreak">
        ${pageHeader('Efficient Frontier')}
        ${whyBox('You already picked the stocks. This shows if you picked the right mix.')}
        ${notRunBox()}
      </section>`
  }

  const yourVol = data.annualised_volatility, yourReturn = data.annualised_return, yourSharpe = data.sharpe_ratio
  const gap = (ef.max_sharpe_sharpe ?? 0) - yourSharpe
  const insightText = gap < 0.05
    ? `Your portfolio is operating near peak efficiency, with a Sharpe ratio of ${fmtN(yourSharpe)} close to the simulated optimum — your current allocation already reflects strong risk-adjusted decision-making.`
    : gap < 0.2
      ? `Your Sharpe ratio of ${fmtN(yourSharpe)} is close to the simulated optimum of ${fmtN(ef.max_sharpe_sharpe)}. Minor weight adjustments could close the remaining gap.`
      : `Your portfolio scores ${fmtN(yourSharpe)} against an achievable optimum of ${fmtN(ef.max_sharpe_sharpe)} — see the optimal weights below for where the biggest adjustment would help.`

  const scatter = buildFrontierScatterSVG({
    vols: ef.vols, returns: ef.returns, sharpes: ef.sharpes,
    yourVol, yourReturn, optVol: ef.max_sharpe_vol, optReturn: ef.max_sharpe_return,
  })

  const weightCard = (title, weightsMap, color) => {
    if (!weightsMap) return ''
    return `
      <div class="fr-card">
        <div class="fr-metric-card-label" style="margin-bottom:8px">${title}</div>
        ${allocRows(Object.entries(weightsMap).map(([t, w]) => ({ label: t, w: w ?? 0 })), () => color)}
      </div>`
  }
  const yourWeightsMap = Object.fromEntries(tickers.map((t, i) => [t, weights[i] ?? 0]))

  return `
    <section class="fr-page fr-pagebreak">
      ${pageHeader('Efficient Frontier')}
      ${whyBox('You already picked the stocks. This shows if you picked the right mix — somewhere on this curve is the version of your portfolio that gets more return for the same risk, or the same return for less.')}
      ${group(sectionLabel('5,000 Simulated Portfolios'), `
      <div class="fr-card">
        <div class="fr-chart-legend">
          <span style="color:var(--color-accent-mint)">● Your portfolio</span>
          <span style="color:var(--color-signal-warning)">● Optimal</span>
        </div>
        ${scatter}
      </div>
      `)}
      ${group(sectionLabel('Sharpe Ratio'), `
      <div class="fr-card">
        <div style="font-size:24px;font-weight:700;font-family:var(--doc-font-mono);color:var(--doc-ink)">${fmtN(yourSharpe)}</div>
        <div style="font-size:11px;color:var(--doc-ink-secondary);margin-top:6px;line-height:1.6">${insightText}</div>
      </div>
      `)}
      <div class="fr-grid2">
        ${weightCard('Your weights', yourWeightsMap, 'var(--color-accent-mint)')}
        ${weightCard(`Optimal — Sharpe ${fmtN(ef.max_sharpe_sharpe)}`, ef.max_sharpe_weights, 'var(--color-signal-warning)')}
      </div>
    </section>
  `
}

// ── Valuation ───────────────────────────────────────────────────────────
// Fundamentals (P/S, EV/EBITDA, risk flags, ...) are fetched by Valuation.jsx
// itself, on that tab's own mount — valuationData is the lifted copy
// Valuation.jsx reports up through onValuationLoaded (see App.jsx), null
// until that tab has actually been opened and loaded for the current
// portfolio.

const fmtX = v => v != null ? `${v.toFixed(1)}x` : 'N/A'
const fmtCap = v => {
  if (!v) return 'N/A'
  if (v >= 1e12) return `$${(v / 1e12).toFixed(1)}T`
  if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`
  return `$${(v / 1e6).toFixed(0)}M`
}

function buildValuationSection({ valuationData }) {
  if (!valuationData) {
    return `
      <section class="fr-page fr-pagebreak">
        ${pageHeader('Valuation')}
        ${whyBox('Price tells you what the market thinks. Fundamentals tell you why — what each holding earns, how fast it is growing, and how much debt it is carrying, with risk flags and strengths surfaced automatically.')}
        ${notRunBox('Open the Valuation tab before exporting to include fundamentals.')}
      </section>
    `
  }

  // Rows, not a fuller risk-flag/positives breakdown — api.py already
  // returns the list ranked by Value/Growth score (lowest = best), same
  // order Valuation.jsx itself renders with no client-side re-sort, so this
  // doesn't re-derive the ranking either. Error/fund rows mirror that
  // table's own handling (a fund has no per-company ratios, an error row
  // shows the message instead of six empty cells).
  const rows = valuationData.map(stock => {
    const hasError = !!stock.error
    const isFund = !hasError && stock.is_fund
    const flagCount = stock.flags?.length || 0
    const flagBadge = flagCount > 0 ? `<span class="fr-val-flag">${flagCount} flag${flagCount > 1 ? 's' : ''}</span>` : ''

    if (hasError) {
      return `<tr><td class="fr-val-ticker">${stock.ticker}</td><td colspan="6" style="text-align:left;color:var(--color-signal-negative);font-size:10px">${stock.error || 'Could not load data'}</td></tr>`
    }
    if (isFund) {
      return `<tr><td class="fr-val-ticker">${stock.ticker}</td><td colspan="6" style="text-align:left;color:var(--doc-ink-muted);font-size:10px">Fund/ETF — valuation ratios don't apply</td></tr>`
    }
    const revGrowthColor = stock.rev_growth > 0.15 ? 'var(--color-accent-mint)' : stock.rev_growth < 0 ? 'var(--color-signal-negative)' : 'var(--doc-ink)'
    const vgColor = stock.vg_score == null ? 'var(--doc-ink-muted)' : stock.vg_score < 0.15 ? 'var(--color-accent-mint)' : stock.vg_score < 0.4 ? 'var(--color-signal-warning)' : 'var(--color-signal-negative)'
    return `
      <tr>
        <td class="fr-val-ticker">${stock.ticker}${flagBadge}</td>
        <td>${fmtX(stock.ps_ratio)}</td>
        <td>${fmtX(stock.ev_ebitda)}</td>
        <td>${fmt(stock.gross_margin)}</td>
        <td style="color:${revGrowthColor}">${stock.rev_growth != null ? `${stock.rev_growth > 0 ? '+' : ''}${(stock.rev_growth * 100).toFixed(1)}%` : 'N/A'}</td>
        <td>${fmtCap(stock.market_cap)}</td>
        <td style="font-weight:700;color:${vgColor}">${stock.vg_score != null ? stock.vg_score.toFixed(2) : 'N/A'}</td>
      </tr>`
  }).join('')

  const table = `
    <table class="fr-val-table">
      <thead><tr><th>Ticker</th><th>P/S</th><th>EV/EBITDA</th><th>Gross Margin</th><th>Rev Growth</th><th>Mkt Cap</th><th>V/G Score</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`

  return `
    <section class="fr-page fr-pagebreak">
      ${pageHeader('Valuation')}
      ${whyBox('Price tells you what the market thinks. Fundamentals tell you why — what each holding earns, how fast it is growing, and how much debt it is carrying, with risk flags and strengths surfaced automatically.')}
      ${group(sectionLabel('Relative Valuation — ranked by Value/Growth score'), `<div class="fr-card" style="padding:0">${table}</div>`)}
    </section>
  `
}

// ── Compare — only rendered when a comparison has actually been run ───────

function buildCompareSection({ data, tickers, comparison }) {
  if (!comparison?.hasRun || !comparison?.data) return ''
  const dataB = comparison.data, tickersB = comparison.tickers || []

  const metrics = [
    { key: 'return', label: 'Ann. Return', aV: data.annualised_return, bV: dataB.annualised_return, fmtFn: fmt },
    { key: 'volatility', label: 'Volatility', aV: data.annualised_volatility, bV: dataB.annualised_volatility, fmtFn: fmt },
    { key: 'sharpe', label: 'Sharpe', aV: data.sharpe_ratio, bV: dataB.sharpe_ratio, fmtFn: fmtN },
    { key: 'sortino', label: 'Sortino', aV: data.sortino_ratio, bV: dataB.sortino_ratio, fmtFn: fmtN },
    { key: 'drawdown', label: 'Drawdown', aV: data.max_drawdown, bV: dataB.max_drawdown, fmtFn: fmt },
    { key: 'var95', label: 'VaR 95%', aV: data.var_cvar.var_pct, bV: dataB.var_cvar.var_pct, fmtFn: fmt },
    { key: 'cvar95', label: 'CVaR 95%', aV: data.var_cvar.cvar_pct, bV: dataB.var_cvar.cvar_pct, fmtFn: fmt },
    ...(data.beta_alpha && dataB.beta_alpha ? [
      { key: 'beta', label: 'Beta', aV: data.beta_alpha.beta, bV: dataB.beta_alpha.beta, fmtFn: fmtN },
      { key: 'alpha', label: 'Alpha', aV: data.beta_alpha.alpha, bV: dataB.beta_alpha.alpha, fmtFn: fmt },
    ] : []),
  ]

  // winner() (lib/compareMetrics.js) is the exact same "which side is
  // actually better" logic the live Compare tab's own table uses — reused,
  // not re-derived, so this report's WIN colouring can never disagree with
  // what the app itself would show for the same two portfolios.
  const rows = metrics.map(m => {
    const w = winner(typeof m.aV === 'number' ? m.aV : null, typeof m.bV === 'number' ? m.bV : null, m.key)
    return `
      <div class="fr-compare-row">
        <div style="text-align:right;color:${w === 'A' ? 'var(--color-accent-mint)' : 'var(--doc-ink-secondary)'};font-weight:${w === 'A' ? 700 : 400}">${m.fmtFn(m.aV)}</div>
        <div style="text-align:center;font-size:10px;color:var(--doc-ink-muted);text-transform:uppercase;letter-spacing:0.05em">${m.label}</div>
        <div style="text-align:left;color:${w === 'B' ? 'var(--color-signal-warning)' : 'var(--doc-ink-secondary)'};font-weight:${w === 'B' ? 700 : 400}">${m.fmtFn(m.bV)}</div>
      </div>`
  }).join('')

  // Same Math.min(...)-length truncation Comparison.jsx's own growthData
  // uses — portfolios A and B can genuinely have been run over different
  // date ranges, so the shorter series' own length is the only length both
  // can share on one chart.
  const minLen = Math.min(data.cumulative_returns.values.length, dataB.cumulative_returns.values.length)
  const growth = buildMultiLineSVG({
    series: [
      { values: data.cumulative_returns.values.slice(0, minLen), color: 'var(--color-accent-mint)', strokeWidth: 2 },
      { values: dataB.cumulative_returns.values.slice(0, minLen), color: 'var(--color-signal-warning)', strokeWidth: 2, dash: true },
    ],
  })

  return `
    <section class="fr-page fr-pagebreak">
      ${pageHeader('Compare')}
      ${whyBox('Two portfolios, one fair fight. The same Sharpe, drawdown, and VaR math runs on both — see which one actually earned its risk.')}
      ${group(`
      <div class="fr-compare-header">
        <div style="text-align:right"><strong style="color:var(--doc-ink)">Portfolio A</strong><br/><span style="font-family:var(--doc-font-mono);font-size:11px;color:var(--doc-ink-secondary)">${tickers.join(', ')}</span></div>
        <div style="color:var(--doc-ink-muted);font-size:11px">VS</div>
        <div style="text-align:left"><strong style="color:var(--doc-ink)">Portfolio B</strong><br/><span style="font-family:var(--doc-font-mono);font-size:11px;color:var(--doc-ink-secondary)">${tickersB.join(', ')}</span></div>
      </div>
      <div class="fr-card">${rows}</div>
      `)}
      ${group(sectionLabel('Cumulative Returns'), `<div class="fr-card">${growth}</div>`)}
    </section>
  `
}

// ── Backtest ────────────────────────────────────────────────────────────

function buildBacktestSection({ data }) {
  const backtest = data.backtest

  if (!backtest) {
    return `
      <section class="fr-page fr-pagebreak">
        ${pageHeader('Backtest')}
        ${whyBox('Stop guessing whether your weights make sense. This runs your exact allocation through the real historical period, side by side with a plain equal split and a benchmark.')}
        ${notRunBox(data.benchmark_cumulative ? undefined : 'Benchmark comparison was turned off for this run, so the backtest is not available.')}
      </section>`
  }

  const benchmarkLabel = getBenchmarkLabel(data.benchmark)
  const strategies = [
    { key: 'your_portfolio', label: 'Your Portfolio', color: 'var(--color-accent-mint)' },
    { key: 'equal_weight', label: 'Equal Weight', color: 'var(--color-signal-warning)' },
    { key: 'sp500', label: benchmarkLabel, color: 'var(--doc-ink-secondary)' },
  ]

  const cards = strategies.map(s => {
    const st = backtest[s.key]
    return `
      <div class="fr-card" style="border-top:2px solid ${s.color}">
        <div class="fr-metric-card-label">${s.label}</div>
        <div class="fr-backtest-row"><span>Annualised return</span><strong style="color:${st.annualised_return >= 0 ? 'var(--color-accent-mint)' : 'var(--color-signal-negative)'}">${fmt(st.annualised_return)}</strong></div>
        <div class="fr-backtest-row"><span>Sharpe ratio</span><strong style="color:var(--doc-ink)">${fmtN(st.sharpe_ratio)}</strong></div>
        <div class="fr-backtest-row"><span>Max drawdown</span><strong style="color:var(--color-signal-negative)">${fmt(st.max_drawdown)}</strong></div>
      </div>`
  }).join('')

  const chart = buildMultiLineSVG({
    series: strategies.map(s => ({
      values: backtest[s.key].cumulative_returns.values,
      color: s.color,
      dash: s.key !== 'your_portfolio',
      strokeWidth: s.key === 'your_portfolio' ? 2.2 : 1.6,
    })),
  })

  const years = Object.keys(backtest.your_portfolio.annual_returns).sort()
  const yearsTable = `
    <table class="fr-years-table">
      <thead><tr><th>Year</th>${strategies.map(s => `<th style="color:${s.color}">${s.label}</th>`).join('')}</tr></thead>
      <tbody>
        ${years.map(y => `<tr><td>${y}</td>${strategies.map(s => {
          const v = backtest[s.key].annual_returns[y]
          return `<td style="color:${v >= 0 ? 'var(--color-accent-mint)' : 'var(--color-signal-negative)'}">${fmt(v)}</td>`
        }).join('')}</tr>`).join('')}
      </tbody>
    </table>`

  return `
    <section class="fr-page fr-pagebreak">
      ${pageHeader('Backtest')}
      ${whyBox(`Stop guessing whether your weights make sense. This runs your exact allocation through the real historical period, side by side with a plain equal split and ${getBenchmarkPhrase(data.benchmark)}, same buy-and-hold, same dates for all three.`)}
      ${group(sectionLabel('Strategy Comparison'), `<div class="fr-strategy-row">${cards}</div>`)}
      ${group(sectionLabel(`Cumulative Return — ${backtest.period.start} to ${backtest.period.end}`), `<div class="fr-card">${chart}</div>`)}
      ${group(sectionLabel('Year-by-Year Returns'), `<div class="fr-card" style="padding:0">${yearsTable}</div>`)}
    </section>
  `
}

// ── Closing: methodology + disclosure ──────────────────────────────────

function buildClosingSection({ date }) {
  return `
    <section class="fr-page fr-pagebreak">
      ${pageHeader('Methodology & Disclosures')}
      <div class="fr-card">
        <div class="fr-metric-card-label" style="margin-bottom:10px">Methodology</div>
        <div style="font-size:12px;color:var(--doc-ink-secondary);line-height:1.8">
          Returns are annualised using CAGR (geometric compounding), not an arithmetic average — the same
          calculation drives the efficient frontier simulation, so "Your portfolio" sits correctly relative to
          the 5,000-portfolio cloud. Monte Carlo simulations model correlated asset returns via Cholesky
          decomposition. Value-at-Risk and Conditional Value-at-Risk are reported at both 95% and 99%
          confidence. All figures assume 252 trading days per year and a 4.5% annual risk-free rate.
        </div>
      </div>
      <div class="fr-card" style="margin-top:12px;border-top:2px solid var(--color-signal-negative)">
        <div style="font-size:11px;color:var(--doc-ink-secondary);line-height:1.8">
          This report is for educational purposes only and does not constitute financial advice. Past
          performance does not guarantee future results. Figures are generated from historical market data and
          statistical models that make simplifying assumptions about markets, and may not reflect actual future
          performance. Always do your own research and consult a qualified financial advisor before making
          investment decisions.
        </div>
      </div>
      <div style="text-align:center;margin-top:24px;font-size:12px;color:var(--doc-ink-muted)">
        Generated by <strong style="color:var(--doc-ink)">Varense</strong> · ${date} · varense.vercel.app
      </div>
    </section>
  `
}

// ── Entry point ─────────────────────────────────────────────────────────

export function runExportFullReportPDF({ data, tickers, weights, portfolioValue, sectorData, comparison, stressTestData, valuationData }) {
  // openReportPopup (lib/pdfReportShared.js) — same pop-up-blocked handling
  // as the Summary report, checked before building this much larger HTML
  // string at all.
  const win = openReportPopup()
  if (!win) return

  const date = new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' })

  // Each section is a self-contained <section class="fr-page ...">; Compare
  // can return '' (falsy) when no comparison has been run, filtered out
  // here rather than rendered empty — every other section always renders
  // (with a not-run note where its data isn't available), matching "skip
  // Compare entirely unless one has been run" from the brief, not "render
  // Compare with a placeholder."
  const sections = [
    buildCoverSection({ data, tickers, weights, portfolioValue, date }),
    buildDashboardSection({ data, sectorData }),
    buildRiskSection({ data, stressTestData, portfolioValue }),
    buildMonteCarloSection({ data }),
    buildFrontierSection({ data, tickers, weights }),
    buildValuationSection({ valuationData }),
    buildCompareSection({ data, tickers, comparison }),
    buildBacktestSection({ data }),
    buildClosingSection({ date }),
  ].filter(Boolean).join('')

  win.document.write(`
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>Varense Full Portfolio Report</title>
          ${FONT_LINKS}
          <style>
            /* docTokenCSS() (lib/pdfReportShared.js) generates the identical
               color/doc token block the Summary report's own style tag uses,
               so the two reports can never carry different brand colours. */
            :root {
              ${docTokenCSS()}
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { background: var(--doc-bg); color: var(--doc-ink); font-family: var(--doc-font-primary); }

            /* DESIGN.md Core Rules: zero border-radius everywhere, no shadows. */
            [class^="fr-"] { border-radius: 0 !important; }

            .fr-page { padding: 28px 36px 36px; }
            /* Each top-level section (every builder above except the cover)
               carries this so it always starts a fresh printed page — the
               brief's "each section starts on a new page" — independent of
               whether the section before it was short or tall. */
            .fr-pagebreak { break-before: page; page-break-before: always; }

            .fr-page-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--doc-hairline); padding-bottom: 10px; margin-bottom: 18px; }
            .fr-page-header-brand { display: flex; align-items: center; gap: 7px; font-size: 11px; font-weight: 600; color: var(--doc-ink-muted); font-family: var(--doc-font-mono); }
            .fr-page-header-title { font-size: 16px; font-weight: 700; color: var(--doc-ink); font-family: var(--doc-font-primary); }

            .fr-why { border-left: 3px solid var(--color-accent-mint); background: rgba(0,0,0,0.025); padding: 10px 14px; font-size: 12px; color: var(--doc-ink-secondary); line-height: 1.6; margin-bottom: 16px; break-inside: avoid; }
            .fr-notrun { border-left: 3px solid var(--doc-ink-muted); background: rgba(0,0,0,0.025); padding: 10px 14px; font-size: 12px; color: var(--doc-ink-muted); font-style: italic; margin-bottom: 12px; break-inside: avoid; }
            .fr-section-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--doc-ink-muted); margin: 18px 0 8px; }
            /* Keeps a label (and/or why-box) on the same page as the content
               it introduces — see the group() helper's own comment above. */
            .fr-group { break-inside: avoid; page-break-inside: avoid; }

            /* Every card is break-inside:avoid — "no stranded cards", the same
               rule the Summary report's own .vr-section uses, applied per-card
               here since a full-report section can legitimately span two
               printed pages (unlike the Summary's forced single page), and
               it's only ever an individual card that must not be split
               mid-content, not the whole section. */
            .fr-card { background: var(--doc-card); border: 1px solid var(--doc-hairline); padding: 14px 16px; break-inside: avoid; page-break-inside: avoid; margin-bottom: 12px; }
            .fr-grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
            .fr-metric-grid { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 12px; }
            .fr-metric-card { flex: 1 1 130px; min-width: 0; background: var(--doc-card); border: 1px solid var(--doc-hairline); padding: 10px 12px; break-inside: avoid; }
            .fr-metric-card-label { font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--doc-ink-muted); }
            .fr-metric-card-value { font-size: 17px; font-weight: 700; font-family: var(--doc-font-mono); color: var(--doc-ink); margin-top: 2px; }

            .fr-pill { display: inline-block; background: var(--doc-card); border: 1px solid var(--doc-hairline); padding: 3px 10px; font-size: 11px; font-weight: 700; color: var(--doc-ink); font-family: var(--doc-font-mono); margin: 0 5px 5px 0; }
            .fr-alloc-row { margin-bottom: 9px; }
            .fr-chart-meta { display: flex; justify-content: space-between; font-size: 12px; color: var(--doc-ink-secondary); margin-bottom: 8px; }
            .fr-chart-legend { display: flex; gap: 16px; font-size: 10px; font-weight: 600; margin-bottom: 8px; flex-wrap: wrap; }
            .fr-risk-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--doc-hairline); }

            .fr-corr-table { width: 100%; border-collapse: separate; border-spacing: 3px; }
            .fr-corr-head { font-size: 10px; color: var(--doc-ink-muted); font-family: var(--doc-font-mono); text-align: center; padding: 0 3px 5px; }
            .fr-corr-cell { text-align: center; font-size: 11px; font-family: var(--doc-font-mono); font-weight: 600; color: var(--doc-ink); padding: 5px 3px; }
            .fr-corr-rowlabel { font-size: 10px; color: var(--doc-ink-muted); font-family: var(--doc-font-mono); text-align: right; padding-right: 8px; }

            .fr-strategy-row { display: flex; flex-wrap: wrap; gap: 10px; }
            .fr-strategy-row .fr-card { flex: 1 1 180px; margin-bottom: 0; }
            .fr-backtest-row { display: flex; justify-content: space-between; font-size: 12px; padding: 4px 0; }
            .fr-backtest-row span { color: var(--doc-ink-muted); }
            .fr-backtest-row strong { font-family: var(--doc-font-mono); }

            .fr-years-table { width: 100%; border-collapse: collapse; font-family: var(--doc-font-mono); font-size: 11px; }
            .fr-years-table th { text-align: right; font-size: 9px; font-weight: 700; color: var(--doc-ink-muted); padding: 8px 10px; border-bottom: 1px solid var(--doc-hairline); text-transform: uppercase; letter-spacing: 0.04em; }
            .fr-years-table th:first-child { text-align: left; }
            .fr-years-table td { text-align: right; padding: 6px 10px; border-bottom: 1px solid rgba(0,0,0,0.04); }
            .fr-years-table td:first-child { text-align: left; color: var(--doc-ink-secondary); }

            .fr-val-table { width: 100%; border-collapse: collapse; font-size: 10px; }
            .fr-val-table th { text-align: right; font-size: 9px; font-weight: 700; color: var(--doc-ink-muted); padding: 8px 10px; border-bottom: 1px solid var(--doc-hairline); text-transform: uppercase; letter-spacing: 0.04em; }
            .fr-val-table th:first-child { text-align: left; }
            .fr-val-table td { text-align: right; padding: 7px 10px; border-bottom: 1px solid rgba(0,0,0,0.04); font-family: var(--doc-font-mono); color: var(--doc-ink); }
            .fr-val-ticker { text-align: left; font-weight: 700; }
            .fr-val-flag { font-size: 8px; font-weight: 700; padding: 1px 5px; background: rgba(224,85,90,0.15); color: var(--color-signal-negative); margin-left: 6px; font-family: var(--doc-font-primary); }

            .fr-compare-header { display: grid; grid-template-columns: 1fr auto 1fr; gap: 10px; align-items: center; margin-bottom: 14px; font-size: 13px; }
            .fr-compare-row { display: grid; grid-template-columns: 1fr auto 1fr; gap: 10px; align-items: center; padding: 7px 0; border-bottom: 1px solid rgba(0,0,0,0.04); font-family: var(--doc-font-mono); font-size: 13px; }

            .fr-cover { display: flex; align-items: center; justify-content: center; min-height: 92vh; }
            .fr-cover-inner { display: flex; flex-direction: column; align-items: center; text-align: center; gap: 22px; }
            .fr-cover-mark { display: flex; align-items: center; gap: 13px; }
            .fr-cover-wordmark { font-size: 32px; font-weight: 500; letter-spacing: -0.035em; color: var(--doc-ink); font-family: var(--doc-font-primary); }
            .fr-cover-title { font-size: 14px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--doc-ink-secondary); }
            .fr-cover-date { font-size: 12px; color: var(--doc-ink-muted); margin-top: 6px; }
            .fr-cover-pills { display: flex; flex-wrap: wrap; justify-content: center; max-width: 480px; }
            .fr-cover-facts { display: flex; gap: 36px; margin-top: 10px; }
            .fr-cover-fact-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--doc-ink-muted); margin-bottom: 4px; }
            .fr-cover-fact-val { font-size: 14px; font-weight: 700; font-family: var(--doc-font-mono); color: var(--doc-ink); }

            /* Readable on a phone before printing — same reasoning as the
               Summary report's identical breakpoint: the viewport meta tag
               above makes CSS px match device px, and every two-column grid
               here collapses to one column so nothing needs horizontal
               scrolling or pinch-zoom to read before tapping Export. */
            @media screen and (max-width: 720px) {
              .fr-page { padding: 18px 16px 24px; }
              .fr-grid2, .fr-compare-header, .fr-compare-row { grid-template-columns: 1fr; }
              .fr-compare-header > div:last-child, .fr-compare-row > div:last-child { text-align: left; }
              .fr-cover-facts { flex-direction: column; gap: 14px; align-items: center; }
            }

            @media print {
              #modal-overlay { display: none !important; }
              @page {
                margin: 14mm 16mm;
                size: A4 portrait;
                /* CSS Paged Media page-number support — Chrome's print engine
                   does not currently render @page margin-box content at all,
                   so this has no visible effect there today. Left in as a
                   good-faith "page numbers if supported" per the brief;
                   harmless in an engine that ignores it. */
                @bottom-right { content: "Page " counter(page) " of " counter(pages); font-family: var(--doc-font-mono); font-size: 9px; color: var(--doc-ink-muted); }
              }
              body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
              .fr-cover { min-height: 90vh; }
            }
          </style>
        </head>
        <body>
          <!-- Same instruction modal as the Summary report — platform-neutral
               wording, no steps the page's own print CSS already forces. -->
          <div id="modal-overlay" style="position:fixed;inset:0;background:rgba(0,0,0,0.4);z-index:9999;display:flex;align-items:center;justify-content:center;padding:16px;">
            <div style="background:var(--doc-bg);border:1px solid var(--doc-hairline);padding:24px 28px;max-width:420px;width:100%;font-family:var(--doc-font-primary);">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
                <svg viewBox="3 9 58 41" width="24" height="17" fill="none" role="img" aria-label="Varense">
                  ${MARK_PATHS}
                </svg>
                <div style="font-size:16px;font-weight:700;color:var(--doc-ink);">Save your full report PDF</div>
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

          <div>${sections}</div>
          <script>
            // No auto-print — user clicks the button in the modal
          </script>
        </body>
      </html>
    `)
  win.document.close()
}
