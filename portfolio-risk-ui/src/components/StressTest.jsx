import { useState, useEffect } from 'react'
import axios from 'axios'

const fmtPct    = v => v == null ? '—' : `${(v * 100).toFixed(1)}%`
const fmtDollar = v => v == null ? '—' : `$${Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: 0 })}`

function ScenarioCard({ scenario, portfolioValue }) {
  const { name, period, portfolio_return, worst_day, recovery_days, excluded_tickers } = scenario
  const noData = portfolio_return == null

  const lossColor = portfolio_return < -0.3 ? '#e05c5c' : portfolio_return < -0.1 ? '#e09a30' : '#52b788'

  return (
    <div className="card" style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div>
        <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-secondary)' }}>
          {name}
        </div>
        <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: 2 }}>
          {period}
        </div>
      </div>

      {noData ? (
        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          No historical data available for this portfolio in this window.
        </div>
      ) : (
        <>
          <div>
            <div style={{ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: 2 }}>
              Portfolio loss
            </div>
            <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'monospace', color: lossColor, lineHeight: 1.1 }}>
              {fmtPct(portfolio_return)}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'monospace', marginTop: 2 }}>
              −{fmtDollar(portfolioValue * Math.abs(portfolio_return))}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div>
              <div style={{ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: 2 }}>
                Worst day
              </div>
              <div style={{ fontSize: 14, fontWeight: 700, fontFamily: 'monospace', color: '#e05c5c' }}>
                {fmtPct(worst_day)}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: 2 }}>
                Recovery
              </div>
              <div style={{ fontSize: 14, fontWeight: 700, fontFamily: 'monospace', color: 'var(--text-primary)' }}>
                {recovery_days == null ? 'Not yet' : `${recovery_days}d`}
              </div>
            </div>
          </div>

          {excluded_tickers?.length > 0 && (
            <div style={{ fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.4 }}>
              Excludes {excluded_tickers.join(', ')} — no data for this period
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default function StressTest({ tickers, weights, portfolioValue }) {
  const [scenarios, setScenarios] = useState(null)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)

  useEffect(() => {
    if (!tickers?.length) return
    setLoading(true)
    setError(null)
    axios.post('/api/stress-test', { tickers, weights, portfolio_value: portfolioValue })
      .then(res => setScenarios(res.data.scenarios))
      .catch(e => setError(e.response?.data?.detail || 'Failed to load stress test scenarios'))
      .finally(() => setLoading(false))
  }, [tickers.join(','), weights.join(',')])

  return (
    <div className="card" style={{ padding: '14px 16px' }}>
      <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-muted)', marginBottom: 12 }}>
        Stress test — historical scenarios
      </div>

      {loading && (
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Running historical scenarios...</div>
      )}

      {error && (
        <div style={{ fontSize: 12, color: 'var(--negative)' }}>{error}</div>
      )}

      {scenarios && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
          {scenarios.map(s => (
            <ScenarioCard key={s.name} scenario={s} portfolioValue={portfolioValue} />
          ))}
        </div>
      )}
    </div>
  )
}
