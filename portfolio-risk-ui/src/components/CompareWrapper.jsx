import { useState } from 'react'
import Comparison from '../pages/Comparison'

const PERIODS = ['1M', '3M', '6M', '1Y', '3Y', '5Y', 'Max']

export default function CompareWrapper({ dataA, tickersA, nameA, compB, portfolios = [], currentPortfolioId = null }) {
  const {
    tickers, weights, period, portfolioValue,
    data: dataB, loading, error, hasRun,
    setTickers, setWeightsAll, setPeriod, setPortfolioValue,
    runComparison,
  } = compB

  const [tickerInput, setTickerInput] = useState(tickers.join(', '))
  const [nameB] = useState('Portfolio B')
  const [showSavedPicker, setShowSavedPicker] = useState(false)

  // Detect which saved portfolio matches the current main portfolio (A)
  const isCurrentA = (p) => {
    if (!tickersA || !p.tickers) return false
    const a = [...tickersA].sort().join(',')
    const b = [...p.tickers].sort().join(',')
    return a === b
  }

  const handleTickerBlur = () => {
    const parsed = tickerInput.split(',').map(t => t.trim().toUpperCase()).filter(Boolean)
    if (parsed.length > 0) setTickers(parsed)
  }

  const weightPct = (() => {
    const raw   = weights.map(w => w * 100)
    const floor = raw.map(v => Math.floor(v))
    const rem   = 100 - floor.reduce((a, b) => a + b, 0)
    const order = raw.map((v, i) => ({ i, frac: v - Math.floor(v) })).sort((a, b) => b.frac - a.frac)
    for (let k = 0; k < rem; k++) floor[order[k].i]++
    return floor
  })()

  const handleWeightChange = (idx, newPct) => {
    const n       = tickers.length
    const clamped = Math.max(0, Math.min(newPct, 100))
    const rem     = 100 - clamped
    const others  = tickers.map((_, i) => i).filter(i => i !== idx)
    const base    = others.length > 0 ? Math.floor(rem / others.length) : 0
    const extra   = rem - base * others.length
    const newW    = new Array(n).fill(0)
    newW[idx]     = clamped
    others.forEach((i, j) => { newW[i] = base + (j < extra ? 1 : 0) })
    setWeightsAll(newW.map(v => v / 100))
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: 12 }}>

      {/* Portfolio B config panel */}
      {!hasRun && (
        <div className="card" style={{ padding: '16px', flexShrink: 0 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
            <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)' }}>
              Configure Portfolio B
            </div>
            {portfolios.length > 0 && (
              <div style={{ position: 'relative' }}>
                <button
                  onClick={() => setShowSavedPicker(v => !v)}
                  style={{
                    fontSize: 11, fontWeight: 600, padding: '5px 12px',
                    border: '1px solid rgba(var(--signal-caution-rgb),0.4)', background: 'rgba(var(--signal-caution-rgb),0.08)',
                    color: 'var(--signal-caution)', cursor: 'pointer',
                  }}
                >
                  Load saved portfolio ▾
                </button>
                {showSavedPicker && (
                  <div style={{
                    position: 'absolute', top: '100%', right: 0, marginTop: 4,
                    background: 'var(--surface-elevated)', border: 'var(--border-default)',
                    zIndex: 100, minWidth: 220,
                    boxShadow: '0 8px 24px rgba(var(--black-rgb),0.4)',
                    overflow: 'hidden',
                  }}>
                    {portfolios.map(p => {
                      const isA = isCurrentA(p)
                      return (
                        <div
                          key={p.id}
                          onClick={() => {
                            if (isA) return
                            setTickers(p.tickers)
                            setTickerInput(p.tickers.join(', '))
                            const total = p.weights.reduce((a, b) => a + b, 0)
                            const normalised = p.weights.map(w => w / total)
                            setWeightsAll(normalised)
                            setPeriod(p.period)
                            setPortfolioValue(p.portfolio_value)
                            setShowSavedPicker(false)
                          }}
                          style={{
                            padding: '10px 14px',
                            cursor: isA ? 'not-allowed' : 'pointer',
                            borderBottom: '1px solid rgba(var(--text-primary-rgb),0.04)',
                            transition: 'background 0.1s',
                            opacity: isA ? 0.5 : 1,
                            background: isA ? 'rgba(var(--signal-positive-rgb),0.05)' : 'transparent',
                          }}
                          onMouseEnter={e => { if (!isA) e.currentTarget.style.background = 'rgba(var(--signal-caution-rgb),0.08)' }}
                          onMouseLeave={e => { if (!isA) e.currentTarget.style.background = 'transparent' }}
                        >
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
                            <div style={{ fontSize: 12, fontWeight: 600, color: isA ? 'var(--signal-positive)' : 'var(--text-primary)' }}>
                              {p.name}
                            </div>
                            {isA && (
                              <div style={{
                                fontSize: 9, fontWeight: 700, padding: '1px 6px',
                                background: 'rgba(var(--signal-positive-rgb),0.15)',
                                color: 'var(--signal-positive)', letterSpacing: '0.05em',
                              }}>
                                ✓ Portfolio A
                              </div>
                            )}
                          </div>
                          <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                            {p.tickers.join(', ')} · {p.period}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            )}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr auto', gap: 12, alignItems: 'end' }}>

            {/* Tickers */}
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 5 }}>Tickers</div>
              <input
                value={tickerInput}
                onChange={e => setTickerInput(e.target.value)}
                onBlur={handleTickerBlur}
                onKeyDown={e => e.key === 'Enter' && handleTickerBlur()}
                placeholder="SPY, QQQ, GLD"
                style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}
              />
            </div>

            {/* Period */}
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 5 }}>Period</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 3 }}>
                {PERIODS.map(p => (
                  <button key={p} onClick={() => setPeriod(p)} style={{
                    padding: '4px 0', fontSize: 10, fontWeight: 600,
                    background: period === p ? 'var(--signal-positive)' : 'var(--surface-elevated)',
                    color: period === p ? 'var(--surface-elevated)' : 'var(--text-muted)',
                    border: 'var(--border-default)', cursor: 'pointer',
                  }}>{p}</button>
                ))}
              </div>
            </div>

            {/* Portfolio value */}
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 5 }}>Portfolio value</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>$</span>
                <input type="number" min={1} value={portfolioValue} onChange={e => setPortfolioValue(parseFloat(e.target.value) || 1)} style={{ fontFamily: 'var(--font-mono)' }} />
              </div>
            </div>

            {/* Run button */}
            <button
              className="btn-primary"
              onClick={runComparison}
              disabled={loading}
              style={{ whiteSpace: 'nowrap', padding: '10px 20px' }}
            >
              {loading ? 'Running...' : 'Run comparison'}
            </button>
          </div>

          {/* Weights */}
          <div style={{ marginTop: 14 }}>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>Weights</div>
            <div style={{ display: 'grid', gridTemplateColumns: `repeat(${tickers.length}, 1fr)`, gap: 8 }}>
              {tickers.map((ticker, i) => (
                <div key={ticker}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{ticker}</span>
                    <span style={{ fontSize: 11, color: 'var(--signal-caution)', fontWeight: 600 }}>{weightPct[i]}%</span>
                  </div>
                  <input
                    type="range" min={0} max={100} value={weightPct[i]}
                    onChange={e => handleWeightChange(i, parseInt(e.target.value))}
                    style={{ width: '100%' }}
                  />
                </div>
              ))}
            </div>
          </div>

          {error && (
            <div style={{ marginTop: 10, fontSize: 12, color: 'var(--signal-negative)', padding: '6px 10px', background: 'rgba(var(--signal-negative-rgb),0.1)' }}>
              {error}
            </div>
          )}
        </div>
      )}

      {/* Re-configure button when results are showing */}
      {hasRun && (
        <div style={{ display: 'flex', justifyContent: 'flex-end', flexShrink: 0 }}>
          <button
            onClick={() => compB.setTickers(tickers)}
            style={{
              fontSize: 11, fontWeight: 600, padding: '5px 14px',
              border: 'var(--border-default)', background: 'transparent',
              color: 'var(--text-muted)', cursor: 'pointer',
            }}
            onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--signal-caution)'}
            onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--line-hairline)'}
          >
            ← Reconfigure Portfolio B
          </button>
          <button
            className="btn-primary"
            onClick={runComparison}
            disabled={loading}
            style={{ marginLeft: 8, padding: '5px 14px', fontSize: 11 }}
          >
            {loading ? 'Running...' : 'Re-run'}
          </button>
        </div>
      )}

      {/* No main portfolio yet */}
      {!dataA && (
        <div style={{ textAlign: 'center', padding: '32px 0', color: 'var(--text-muted)', fontSize: 13 }}>
          Run your main portfolio analysis first (Dashboard tab), then come back here to compare.
        </div>
      )}

      {/* Results */}
      {dataA && dataB && (
        <div style={{ flex: 1, minHeight: 0 }}>
          <Comparison
            dataA={dataA} dataB={dataB}
            nameA={nameA} nameB={nameB}
            tickersA={tickersA} tickersB={tickers}
          />
        </div>
      )}

    </div>
  )
}