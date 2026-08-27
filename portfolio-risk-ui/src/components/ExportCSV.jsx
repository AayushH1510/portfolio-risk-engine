const fmt  = v => v != null ? `${(v * 100).toFixed(1)}%` : 'N/A'
const fmtN = v => v != null ? v.toFixed(2) : 'N/A'

function csvEscape(val) {
  const s = String(val ?? '')
  if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`
  return s
}

function csvRow(values) {
  return values.map(csvEscape).join(',')
}

function buildCSV(data) {
  const lines = []

  // ── Section 1: Portfolio Summary ──────────────────────────────────────
  const vc   = data.var_cvar
  const vc99 = data.var_cvar_99
  const ba   = data.beta_alpha
  const ds   = data.diversification_score

  lines.push(csvRow(['Metric', 'Value']))
  lines.push(csvRow(['Annual Return',        fmt(data.annualised_return)]))
  lines.push(csvRow(['Volatility',           fmt(data.annualised_volatility)]))
  lines.push(csvRow(['Sharpe Ratio',         fmtN(data.sharpe_ratio)]))
  lines.push(csvRow(['Sortino Ratio',        fmtN(data.sortino_ratio)]))
  lines.push(csvRow(['Treynor Ratio',        fmtN(data.treynor_ratio)]))
  lines.push(csvRow(['Information Ratio',    fmtN(data.information_ratio)]))
  lines.push(csvRow(['Max Drawdown',         fmt(data.max_drawdown)]))
  lines.push(csvRow(['VaR 95%',              fmt(vc?.var_pct)]))
  lines.push(csvRow(['CVaR 95%',             fmt(vc?.cvar_pct)]))
  lines.push(csvRow(['VaR 99%',              fmt(vc99?.var_pct)]))
  lines.push(csvRow(['CVaR 99%',             fmt(vc99?.cvar_pct)]))
  lines.push(csvRow(['Beta',                 fmtN(ba?.beta)]))
  lines.push(csvRow(['Alpha',                fmt(ba?.alpha)]))
  lines.push(csvRow(['Diversification Score', ds ? `${ds.score}/100` : 'N/A']))

  // ── Section 2: Daily Returns ───────────────────────────────────────────
  lines.push('')
  lines.push(csvRow(['Date', 'Portfolio Return', 'Cumulative Return']))
  const cum = data.cumulative_returns
  cum.dates.forEach((date, i) => {
    lines.push(csvRow([date, data.portfolio_returns?.[i] ?? '', cum.values[i]]))
  })

  // ── Section 3: Annual Backtest Returns ─────────────────────────────────
  if (data.backtest) {
    const { your_portfolio, equal_weight, sp500 } = data.backtest
    lines.push('')
    lines.push(csvRow(['Year', 'Your Portfolio', 'Equal Weight', 'S&P 500']))
    Object.keys(your_portfolio.annual_returns).sort().forEach(year => {
      lines.push(csvRow([
        year,
        fmt(your_portfolio.annual_returns[year]),
        fmt(equal_weight.annual_returns[year]),
        fmt(sp500.annual_returns[year]),
      ]))
    })
  }

  return lines.join('\n')
}

export default function ExportCSV({ data, tickers, weights }) {
  if (!data) return null

  const handleExport = () => {
    const csv  = buildCSV(data, tickers, weights)
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url  = URL.createObjectURL(blob)
    const date = new Date().toISOString().slice(0, 10)

    const a = document.createElement('a')
    a.href = url
    a.download = `varense-portfolio-${date}.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <button
      onClick={handleExport}
      style={{
        display: 'flex', alignItems: 'center', gap: 6,
        fontSize: 11, fontWeight: 600, padding: '6px 14px',
        borderRadius: 'var(--radius-6)', border: '1px solid var(--border-light)',
        background: 'rgba(var(--white-rgb),0.5)',
        color: 'var(--accent-dark)', cursor: 'pointer',
        transition: 'all 0.15s', letterSpacing: '0.03em',
      }}
      onMouseEnter={e => { e.currentTarget.style.background = 'rgba(var(--accent-dark-rgb),0.1)'; e.currentTarget.style.borderColor = 'var(--accent-dark)' }}
      onMouseLeave={e => { e.currentTarget.style.background = 'rgba(var(--white-rgb),0.5)'; e.currentTarget.style.borderColor = 'var(--border-light)' }}
    >
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="7 10 12 15 17 10"/>
        <line x1="12" y1="15" x2="12" y2="3"/>
      </svg>
      Export CSV
    </button>
  )
}
