import { useState, useEffect } from 'react'
import axios from 'axios'
import { errorMessage } from '../lib/errorMessage'
import MetricTooltip from './MetricTooltip'

// The backend's catch-all message for an unexpected server error — see
// api.py's GENERIC_ERROR_DETAIL. Swapped for a more actionable instruction
// here specifically, since "try again" with no guidance on how is a dead
// end for a component whose only action is an automatic re-fetch on mount.
const GENERIC_BACKEND_ERROR = 'Something went wrong processing your request. Please try again.'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const fmtPct    = v => v == null ? '-' : `${(v * 100).toFixed(1)}%`
const fmtDollar = v => v == null ? '-' : `$${Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: 0 })}`

function ScenarioCard({ scenario, portfolioValue }) {
  const { name, period, portfolio_return, worst_day, recovery_days, excluded_tickers } = scenario
  const noData = portfolio_return == null

  const lossColor = portfolio_return < -0.3 ? 'var(--signal-negative)' : portfolio_return < -0.1 ? 'var(--signal-caution)' : 'var(--signal-positive)'
  const recoveryText = recovery_days == null
    ? 'and hasn’t recovered yet'
    : `taking ${recovery_days} day${recovery_days === 1 ? '' : 's'} to recover`

  return (
    <div className="card" style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div>
        <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)' }}>
          {name}
        </div>
        <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: 2 }}>
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
            <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)', marginBottom: 2 }}>
              Portfolio loss
            </div>
            <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'var(--font-mono)', color: lossColor, lineHeight: 1.1 }}>
              {fmtPct(portfolio_return)}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: 2 }}>
              −{fmtDollar(portfolioValue * Math.abs(portfolio_return))}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div>
              <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)', marginBottom: 2 }}>
                Worst day
              </div>
              <div style={{ fontSize: 14, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--signal-negative)' }}>
                {fmtPct(worst_day)}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)', marginBottom: 2 }}>
                Recovery
              </div>
              <div style={{ fontSize: 14, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                {recovery_days == null ? 'Not yet' : `${recovery_days}d`}
              </div>
            </div>
          </div>

          <div style={{ fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.5 }}>
            In {name}, your exact portfolio would have lost{' '}
            <strong style={{ color: 'rgba(var(--text-primary-rgb),0.7)' }}>{fmtPct(portfolio_return)}</strong>, with its worst single day at{' '}
            <strong style={{ color: 'rgba(var(--text-primary-rgb),0.7)' }}>{fmtPct(worst_day)}</strong>, {recoveryText}.
          </div>

          {excluded_tickers?.length > 0 && (
            <div style={{ fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.4 }}>
              Excludes {excluded_tickers.join(', ')} - no data for this period
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
    axios.post(`${API}/api/stress-test`, { tickers, weights, portfolio_value: portfolioValue })
      .then(res => setScenarios(res.data.scenarios))
      .catch(e => {
        const msg = errorMessage(e, 'Failed to load stress test scenarios')
        setError(msg === GENERIC_BACKEND_ERROR ? 'Please refresh the page and try again.' : msg)
      })
      .finally(() => setLoading(false))
  }, [tickers.join(','), weights.join(',')])

  return (
    <div className="card" style={{ padding: '14px 16px' }}>
      <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)', marginBottom: 12 }}>
        <MetricTooltip metricKey="stress_test">Stress test - historical scenarios</MetricTooltip>
      </div>

      {loading && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <svg className="spin" width="18" height="18" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="var(--signal-positive)" strokeWidth="3" strokeDasharray="40 20" />
          </svg>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Running historical scenarios...</div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', opacity: 0.7 }}>
              Running the full simulation, this can take up to a minute.
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', opacity: 0.7 }}>
              Taking longer than usual? A quick refresh usually does the trick.
            </div>
          </div>
        </div>
      )}

      {error && (
        <div style={{ fontSize: 12, color: 'var(--signal-negative)' }}>{error}</div>
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
