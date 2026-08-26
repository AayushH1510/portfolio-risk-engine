import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'

const fmtPct = v => `${(v * 100).toFixed(1)}%`

const CHART_STYLE = { background: 'transparent', fontSize: 11, fontFamily: 'monospace' }
const AXIS_STYLE  = { fill: 'var(--text-muted)', fontSize: 10 }

const STRATEGIES = [
  { key: 'your_portfolio', label: 'Your Portfolio', color: '#52b788', dash: false },
  { key: 'equal_weight',   label: 'Equal Weight',   color: '#e09a30', dash: true  },
  { key: 'sp500',          label: 'S&P 500',        color: '#5a7a5a', dash: true  },
]

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--card)', border: '1px solid var(--border)',
      borderRadius: 8, padding: '8px 12px', fontSize: 11,
    }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, fontWeight: 600 }}>
          {p.name}: {fmtPct(p.value)}
        </div>
      ))}
    </div>
  )
}

function Row({ label, value, tone }) {
  const color = { good: 'var(--positive)', warning: 'var(--warning)', bad: 'var(--negative)' }[tone] || 'var(--text-primary)'
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ fontSize: 14, fontWeight: 700, fontFamily: 'monospace', color }}>{value}</span>
    </div>
  )
}

function StrategyCard({ label, color, sub, stats, borderTone }) {
  const { annualised_return, sharpe_ratio, max_drawdown } = stats
  const borderColor = borderTone === 'good' ? 'var(--positive)' : borderTone === 'bad' ? 'var(--negative)' : color

  return (
    <div className="card" style={{ padding: '14px 16px', borderTop: `2px solid ${borderColor}`, display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div>
        <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-secondary)' }}>
          {label}
        </div>
        {sub && (
          <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: 2 }}>{sub}</div>
        )}
      </div>
      <Row label="Annualised return" value={fmtPct(annualised_return)} tone={annualised_return >= 0 ? 'good' : 'bad'} />
      <Row label="Sharpe ratio"      value={sharpe_ratio.toFixed(2)}   tone={sharpe_ratio > 1 ? 'good' : sharpe_ratio > 0 ? 'warning' : 'bad'} />
      <Row label="Max drawdown"      value={fmtPct(max_drawdown)}      tone={Math.abs(max_drawdown) < 0.2 ? 'good' : Math.abs(max_drawdown) < 0.35 ? 'warning' : 'bad'} />
    </div>
  )
}

function ReturnsTable({ backtest }) {
  const years = Object.keys(backtest.your_portfolio.annual_returns).sort()

  return (
    <div className="card" style={{ padding: '14px 16px' }}>
      <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-muted)', marginBottom: 12 }}>
        Year-by-year returns
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: 0, fontFamily: 'monospace' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', fontSize: 10, fontWeight: 700, color: 'var(--text-muted)', padding: '4px 10px 8px 4px', borderBottom: '1px solid var(--border)' }}>
                Year
              </th>
              {STRATEGIES.map(s => (
                <th key={s.key} style={{ textAlign: 'right', fontSize: 10, fontWeight: 700, color: s.color, padding: '4px 10px 8px', borderBottom: '1px solid var(--border)' }}>
                  {s.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {years.map(y => (
              <tr key={y}>
                <td style={{ fontSize: 12, color: 'var(--text-secondary)', padding: '7px 10px 7px 4px', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                  {y}
                </td>
                {STRATEGIES.map(s => {
                  const v = backtest[s.key].annual_returns[y]
                  return (
                    <td key={s.key} style={{
                      textAlign: 'right', fontSize: 12, fontWeight: 600, padding: '7px 10px',
                      borderBottom: '1px solid rgba(255,255,255,0.04)',
                      color: v >= 0 ? 'var(--positive)' : 'var(--negative)',
                    }}>
                      {fmtPct(v)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function Backtest({ data, tickers }) {
  if (!data) return null

  const backtest = data.backtest
  if (!backtest) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)', fontSize: 13 }}>
        Enable "Compare vs S&P 500" in the sidebar and rerun the analysis to see the backtest.
      </div>
    )
  }

  const sharpes = {
    your_portfolio: backtest.your_portfolio.sharpe_ratio,
    equal_weight:   backtest.equal_weight.sharpe_ratio,
    sp500:          backtest.sp500.sharpe_ratio,
  }
  const bestKey  = Object.entries(sharpes).reduce((a, b) => (b[1] > a[1] ? b : a))[0]
  const worstKey = Object.entries(sharpes).reduce((a, b) => (b[1] < a[1] ? b : a))[0]
  const yourBorderTone = bestKey === 'your_portfolio' ? 'good' : worstKey === 'your_portfolio' ? 'bad' : null

  const dates = backtest.your_portfolio.cumulative_returns.dates
  const chartData = dates.map((d, i) => ({
    date: d,
    your_portfolio: backtest.your_portfolio.cumulative_returns.values[i],
    equal_weight:   backtest.equal_weight.cumulative_returns.values[i],
    sp500:          backtest.sp500.cumulative_returns.values[i],
  }))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, height: '100%', overflowY: 'auto' }}>

      {/* Summary cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, flexShrink: 0 }}>
        <StrategyCard
          label="Your Portfolio"
          sub={tickers?.join(', ')}
          color="#52b788"
          stats={backtest.your_portfolio}
          borderTone={yourBorderTone}
        />
        <StrategyCard label="Equal Weight" color="#e09a30" stats={backtest.equal_weight} />
        <StrategyCard label="S&P 500"      color="#5a7a5a" stats={backtest.sp500} />
      </div>

      {/* Cumulative return chart */}
      <div className="card" style={{ padding: '14px 16px', flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexShrink: 0 }}>
          <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-secondary)' }}>
            Cumulative return — {backtest.period.start} to {backtest.period.end}
          </div>
          <div style={{ display: 'flex', gap: 14, fontSize: 10 }}>
            {STRATEGIES.map(s => (
              <span key={s.key} style={{ display: 'flex', alignItems: 'center', gap: 5, color: 'var(--text-muted)' }}>
                <svg width="20" height="2" style={{ flexShrink: 0 }}>
                  <line x1="0" y1="1" x2="20" y2="1" stroke={s.color} strokeWidth="2" strokeDasharray={s.dash ? '4 3' : 'none'} />
                </svg>
                {s.label}
              </span>
            ))}
          </div>
        </div>
        <div style={{ flex: 1, minHeight: 0 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} style={CHART_STYLE} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
              <XAxis dataKey="date" tick={AXIS_STYLE} tickLine={false} axisLine={false} interval="preserveStartEnd" />
              <YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={AXIS_STYLE} tickLine={false} axisLine={false} width={48} />
              <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" strokeDasharray="4 4" />
              <Tooltip content={<CustomTooltip />} />
              {STRATEGIES.map(s => (
                <Line
                  key={s.key}
                  type="monotone"
                  dataKey={s.key}
                  name={s.label}
                  stroke={s.color}
                  strokeWidth={s.key === 'your_portfolio' ? 2 : 1.5}
                  strokeDasharray={s.dash ? '5 4' : undefined}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Year-by-year table */}
      <ReturnsTable backtest={backtest} />

    </div>
  )
}
