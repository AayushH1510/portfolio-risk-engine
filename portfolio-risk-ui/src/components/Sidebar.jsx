import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import SavedPortfolios from './SavedPortfolios'
import Logo from './Logo'

const PERIODS = ['1M', '3M', '6M', '1Y', '3Y', '5Y', 'Max']
const WINDOWS = [
  { val: 20, label: '20d - reactive' },
  { val: 30, label: '30d - standard' },
  { val: 60, label: '60d - smooth' },
  { val: 90, label: '90d - long term' },
]

const SIDEBAR_MIN = 220
const SIDEBAR_MAX = 400
const SIDEBAR_STORAGE_KEY = 'varense-sidebar-width'

// index.css's --sidebar-width is the source of truth for the default (see
// DESIGN.md) — only read a hardcoded fallback here for the rare case this
// runs before the stylesheet has applied (or in a non-browser context).
function getInitialSidebarWidth() {
  if (typeof window === 'undefined') return 260
  const stored = parseInt(localStorage.getItem(SIDEBAR_STORAGE_KEY), 10)
  if (!Number.isNaN(stored)) return Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, stored))
  const cssValue = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--sidebar-width'), 10)
  return Number.isNaN(cssValue) ? 260 : cssValue
}

export default function Sidebar({
  tickers, weights, period, portfolioValue,
  showBenchmark, rollingWindow,
  setTickers, setWeightsAll, setPeriod,
  portfolios, onSavePortfolio, onLoadPortfolio, onDeletePortfolio,
  setPortfolioValue, setShowBenchmark, setRollingWindow,
  onRun, loading, onTickerClick,
}) {
  const [logoHovered, setLogoHovered]     = useState(false)
  const [sidebarWidth, setSidebarWidth]   = useState(getInitialSidebarWidth)
  const [tickerInput, setTickerInput]     = useState(tickers.join(', '))
  const [inputMode, setInputMode]         = useState('pct')
  const [useCustomDate, setUseCustomDate] = useState(false)
  const [customStart, setCustomStart]     = useState({ day: '01', month: '01', year: '2022' })
  const [customEnd, setCustomEnd]         = useState({
    day:   new Date().getDate().toString().padStart(2, '0'),
    month: (new Date().getMonth() + 1).toString().padStart(2, '0'),
    year:  new Date().getFullYear().toString(),
  })
  const [dollarAmts, setDollarAmts] = useState(() =>
    tickers.map(() => portfolioValue / tickers.length)
  )

  useEffect(() => {
    setDollarAmts(tickers.map(() => portfolioValue / tickers.length))
  }, [tickers.length, portfolioValue])

  useEffect(() => {
    setTickerInput(tickers.join(', '))
  }, [tickers.join(',')])

  // --sidebar-width (index.css) is the source of truth other CSS could key
  // off of — keep it live so a drag actually resizes the aside below rather
  // than a hardcoded pixel value doing the work.
  useEffect(() => {
    document.documentElement.style.setProperty('--sidebar-width', `${sidebarWidth}px`)
  }, [sidebarWidth])

  const handleResizeStart = (e) => {
    e.preventDefault()
    const startX     = e.clientX
    const startWidth = sidebarWidth

    const onMove = (moveEvent) => {
      const next = Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, startWidth + (moveEvent.clientX - startX)))
      setSidebarWidth(next)
    }
    const onUp = () => {
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      // Persist only on drag end, not every intermediate mousemove — a
      // functional update so this reads whatever the latest width actually
      // settled on rather than a value closed over at drag-start.
      setSidebarWidth(w => {
        localStorage.setItem(SIDEBAR_STORAGE_KEY, String(w))
        return w
      })
    }

    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  const handleTickerBlur = () => {
    const parsed = tickerInput.split(',').map(t => t.trim().toUpperCase()).filter(Boolean)
    if (parsed.length > 0) setTickers(parsed.slice(0, 5))
    if (parsed.length > 5) setTickerInput(parsed.slice(0, 5).join(', '))
  }

  const handlePctChange = (idx, newPct) => {
    const isLast = idx === tickers.length - 1
    if (isLast) return
    const newW     = [...weightPct]
    const otherSum = newW.reduce((s, v, i) => i !== idx && i !== tickers.length - 1 ? s + v : s, 0)
    const maxVal   = 100 - otherSum
    newW[idx] = Math.max(0, Math.min(newPct, maxVal))
    const usedSum = newW.reduce((s, v, i) => i !== tickers.length - 1 ? s + v : s, 0)
    newW[tickers.length - 1] = Math.max(0, 100 - usedSum)
    setWeightsAll(newW.map(v => v / 100))
  }

  const handleDollarChange = (idx, val) => {
    const clamped   = Math.max(0, Math.min(val, portfolioValue))
    const remaining = portfolioValue - clamped
    const newAmts   = [...dollarAmts]
    newAmts[idx]    = clamped
    const others      = newAmts.map((v, i) => ({ i, v })).filter(o => o.i !== idx)
    const totalOthers = others.reduce((s, o) => s + o.v, 0)
    others.forEach((o, j) => {
      if (j === others.length - 1) {
        const sumSoFar = others.slice(0, j).reduce((s, oo) => s + newAmts[oo.i], 0)
        newAmts[o.i] = Math.max(0, remaining - sumSoFar)
      } else {
        const share  = totalOthers > 0 ? o.v / totalOthers : 1 / others.length
        newAmts[o.i] = Math.max(0, Math.round(remaining * share))
      }
    })
    const total = newAmts.reduce((s, v) => s + v, 0)
    const drift = portfolioValue - total
    if (drift !== 0) {
      const lastOther = others[others.length - 1]
      if (lastOther) newAmts[lastOther.i] = Math.max(0, newAmts[lastOther.i] + drift)
    }
    setDollarAmts(newAmts)
    setWeightsAll(newAmts.map(v => v / portfolioValue))
  }

  const weightPct   = weights.map(w => Math.round(w * 100))
  const totalDollar = dollarAmts.reduce((a, b) => a + b, 0)

  // Clickable ticker label
  const TickerLabel = ({ ticker, weight }) => {
    const [hovered, setHovered] = useState(false)
    return (
      <span
        onClick={() => onTickerClick?.(ticker, weight)}
        title="Click to view stock details"
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        style={{
          fontSize: 11, fontWeight: 700, fontFamily: 'var(--font-mono)',
          color: hovered ? 'var(--signal-positive)' : 'var(--text-primary)',
          cursor: 'pointer',
          display: 'inline-flex', alignItems: 'center', gap: 3,
          paddingBottom: 1,
          borderBottom: hovered
            ? '1.5px solid var(--signal-positive)'
            : '1.5px dashed rgba(var(--signal-positive-rgb),0.4)',
          transition: 'color 0.15s, border-color 0.15s',
          userSelect: 'none',
        }}
      >
        {ticker}
        <svg
          width="8" height="8" viewBox="0 0 8 8" fill="none"
          style={{ opacity: hovered ? 1 : 0.5, transition: 'opacity 0.15s' }}
        >
          <path d="M1.5 4H6.5M4 1.5L6.5 4L4 6.5" stroke="var(--signal-positive)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </span>
    )
  }

  return (
    <aside className="grain-surface" style={{
      width: 'var(--sidebar-width)', minWidth: 'var(--sidebar-width)', flexShrink: 0, height: '100%',
      background: 'var(--surface-sidebar)',
      borderRight: 'var(--border-default)',
      display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative',
    }}>

      {/* Logo */}
      <div style={{ padding: '18px 16px 14px', borderBottom: 'var(--border-default)' }}>
        <Link
          to="/"
          onMouseEnter={() => setLogoHovered(true)}
          onMouseLeave={() => setLogoHovered(false)}
          style={{ display: 'inline-block', textDecoration: 'none', cursor: 'pointer' }}
        >
          <Logo
            variant="horizontal"
            size={28}
            ink={logoHovered ? 'var(--signal-positive)' : 'var(--text-primary)'}
          />
          <div style={{ fontSize: 10, color: 'var(--text-muted)', letterSpacing: '0.04em', marginTop: 4 }}>v1.2</div>
        </Link>
      </div>

      {/* Settings */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '14px 14px 0' }}>

        <SavedPortfolios
          portfolios={portfolios}
          onLoad={onLoadPortfolio}
          onDelete={onDeletePortfolio}
          onSave={onSavePortfolio}
          currentTickers={tickers}
          currentWeights={weights}
          currentPeriod={period}
          currentPortfolioValue={portfolioValue}
        />

        {/* 1. Stocks */}
        <Section label="1. Stocks" tourId="stocks">
          <input
            value={tickerInput}
            onChange={e => setTickerInput(e.target.value)}
            onBlur={handleTickerBlur}
            onKeyDown={e => e.key === 'Enter' && handleTickerBlur()}
            placeholder="AAPL, MSFT, GOOGL"
            style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}
          />
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
            Comma separated Tickers · Max 5 (Works best with 3)
          </div>
        </Section>

        {/* 2. Portfolio value */}
        <Section label="2. Portfolio value">
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>$</span>
            <input
              type="number" min={1} value={portfolioValue}
              onChange={e => setPortfolioValue(parseFloat(e.target.value) || 1)}
              style={{ fontFamily: 'var(--font-mono)' }}
            />
          </div>
        </Section>

        {/* 3. Weights */}
        <Section label="3. Weights" tourId="weights">

          <div style={{ display: 'flex', gap: 6, marginBottom: 10 }}>
            {['pct', 'dollar'].map(m => (
              <button key={m} onClick={() => setInputMode(m)} style={{
                flex: 1, padding: '5px 0', fontSize: 10,
                fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase',
                background: inputMode === m ? 'var(--signal-positive)' : 'var(--surface-elevated)',
                color: inputMode === m ? 'var(--surface-canvas)' : 'var(--text-muted)',
                border: 'var(--border-default)', transition: 'all 0.15s',
              }}>
                {m === 'pct' ? '% Split' : '$ Amount'}
              </button>
            ))}
          </div>

          {/* % Split mode */}
          {inputMode === 'pct' && (
            <>
              {tickers.map((ticker, i) => {
                const isLast = i === tickers.length - 1
                return (
                  <div key={ticker} style={{ marginBottom: 10 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                      <TickerLabel ticker={ticker} weight={weights[i]} />
                      <span style={{ fontSize: 11, fontWeight: 600, fontFamily: 'var(--font-mono)', color: isLast ? 'var(--text-muted)' : 'var(--signal-positive)' }}>
                        {weightPct[i]}%
                        {isLast && <span style={{ fontSize: 9, marginLeft: 4, color: 'var(--text-muted)' }}>auto</span>}
                      </span>
                    </div>
                    {isLast ? (
                      <div style={{ height: 4, background: 'var(--surface-elevated)', overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${weightPct[i]}%`, background: 'var(--text-muted)', opacity: 0.5 }} />
                      </div>
                    ) : (
                      <input
                        type="range" min={0} max={100} value={weightPct[i]}
                        onChange={e => handlePctChange(i, parseInt(e.target.value))}
                        style={{ width: '100%' }}
                      />
                    )}
                  </div>
                )
              })}
              <div style={{ fontSize: 10, color: 'var(--signal-positive)', marginTop: 2 }}>100% allocated</div>
            </>
          )}

          {/* $ Amount mode */}
          {inputMode === 'dollar' && (
            <>
              <div style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                marginBottom: 10, padding: '6px 10px',
                background: 'var(--surface-elevated)', border: 'var(--border-default)',
              }}>
                <span style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Total</span>
                <span style={{
                  fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 13,
                  color: Math.abs(totalDollar - portfolioValue) < 1 ? 'var(--signal-positive)' : 'var(--signal-caution)',
                }}>
                  ${Math.round(totalDollar).toLocaleString()}
                  <span style={{ fontSize: 9, color: 'var(--text-muted)', marginLeft: 4 }}>/ ${portfolioValue.toLocaleString()}</span>
                </span>
              </div>

              {tickers.map((ticker, i) => {
                const isLast = i === tickers.length - 1
                return (
                  <div key={ticker} style={{ marginBottom: 10 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                      <TickerLabel ticker={ticker} weight={weights[i]} />
                      <span style={{ fontSize: 11, fontWeight: 600, fontFamily: 'var(--font-mono)', color: isLast ? 'var(--text-muted)' : 'var(--signal-positive)' }}>
                        ${Math.round(dollarAmts[i]).toLocaleString()}
                        {isLast && <span style={{ fontSize: 9, marginLeft: 4, color: 'var(--text-muted)' }}>auto</span>}
                      </span>
                    </div>
                    {isLast ? (
                      <div style={{ height: 4, background: 'var(--surface-elevated)', overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${(dollarAmts[i] / portfolioValue) * 100}%`, background: 'var(--text-muted)', opacity: 0.4 }} />
                      </div>
                    ) : (
                      <input type="range" min={0} max={portfolioValue}
                        step={Math.max(1, Math.round(portfolioValue / 200))}
                        value={Math.round(dollarAmts[i])}
                        onChange={e => handleDollarChange(i, parseFloat(e.target.value))}
                        style={{ width: '100%' }} />
                    )}
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-muted)', marginTop: 2 }}>
                      <span>$0</span>
                      <span>{Math.round((dollarAmts[i] / portfolioValue) * 100)}%</span>
                      <span>${portfolioValue.toLocaleString()}</span>
                    </div>
                  </div>
                )
              })}
            </>
          )}

        </Section>

        {/* 4. Period */}
        <Section label="4. Period" tourId="period">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 4, marginBottom: 8 }}>
            {PERIODS.map(p => (
              <button key={p} onClick={() => { setPeriod(p); setUseCustomDate(false) }} style={{
                padding: '5px 0', fontSize: 10, fontWeight: 600,
                background: period === p && !useCustomDate ? 'var(--signal-positive)' : 'var(--surface-elevated)',
                color: period === p && !useCustomDate ? 'var(--surface-canvas)' : 'var(--text-muted)',
                border: 'var(--border-default)', transition: 'all 0.15s',
              }}>{p}</button>
            ))}
          </div>

          <button
            onClick={() => setUseCustomDate(v => !v)}
            style={{
              width: '100%', padding: '5px 0', fontSize: 10, fontWeight: 600,
              background: useCustomDate ? 'var(--signal-positive)' : 'var(--surface-elevated)',
              color: useCustomDate ? 'var(--surface-canvas)' : 'var(--text-muted)',
              border: 'var(--border-default)', transition: 'all 0.15s', marginBottom: useCustomDate ? 8 : 0,
            }}
          >
            Custom dates
          </button>

          {useCustomDate && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {[
                { label: 'Start', state: customStart, set: setCustomStart },
                { label: 'End',   state: customEnd,   set: setCustomEnd   },
              ].map(({ label, state, set }) => (
                <div key={label}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, marginBottom: 4, letterSpacing: '0.05em', textTransform: 'uppercase' }}>{label}</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '2fr 2fr 3fr', gap: 4 }}>
                    <select value={state.day} onChange={e => set(s => ({ ...s, day: e.target.value }))} style={{ fontSize: 11, padding: '5px 4px' }}>
                      {Array.from({ length: 31 }, (_, i) => (i + 1).toString().padStart(2, '0')).map(d => (
                        <option key={d} value={d}>{d}</option>
                      ))}
                    </select>
                    <select value={state.month} onChange={e => set(s => ({ ...s, month: e.target.value }))} style={{ fontSize: 11, padding: '5px 4px' }}>
                      {['01','02','03','04','05','06','07','08','09','10','11','12'].map((m, i) => (
                        <option key={m} value={m}>{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][i]}</option>
                      ))}
                    </select>
                    <select value={state.year} onChange={e => set(s => ({ ...s, year: e.target.value }))} style={{ fontSize: 11, padding: '5px 4px' }}>
                      {Array.from({ length: 15 }, (_, i) => (new Date().getFullYear() - i).toString()).map(y => (
                        <option key={y} value={y}>{y}</option>
                      ))}
                    </select>
                  </div>
                </div>
              ))}
              <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                {customStart.year}-{customStart.month}-{customStart.day} → {customEnd.year}-{customEnd.month}-{customEnd.day}
              </div>
            </div>
          )}
        </Section>

        {/* 5. Risk window */}
        <Section label="5. Risk window">
          <select value={rollingWindow} onChange={e => setRollingWindow(parseInt(e.target.value))}>
            {WINDOWS.map(w => <option key={w.val} value={w.val}>{w.label}</option>)}
          </select>
        </Section>

        {/* 6. Benchmark */}
        <Section label="6. Benchmark">
          <div onClick={() => setShowBenchmark(!showBenchmark)} style={{
            display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer',
            padding: '7px 10px',
            background: showBenchmark ? 'rgba(var(--signal-positive-rgb),0.12)' : 'var(--surface-elevated)',
            border: `1px solid ${showBenchmark ? 'var(--signal-positive)' : 'var(--line-hairline)'}`,
            transition: 'all 0.15s',
          }}>
            <div style={{
              width: 16, height: 16,
              background: showBenchmark ? 'var(--signal-positive)' : 'var(--surface-elevated)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'all 0.15s',
            }}>
              {showBenchmark && (
                <svg width="10" height="8" viewBox="0 0 10 8" fill="none">
                  <path d="M1 4L3.5 6.5L9 1" stroke="var(--surface-canvas)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              )}
            </div>
            <span style={{ fontSize: 11, color: showBenchmark ? 'var(--text-primary)' : 'var(--text-muted)', fontWeight: 500 }}>
              Compare vs S&P 500
            </span>
          </div>
        </Section>

      </div>

      {/* Run button */}
      <div style={{ padding: 14, borderTop: 'var(--border-default)' }}>
        <button
          data-tour="run-analysis"
          className="btn-primary"
          onClick={() => onRun(useCustomDate ? {
            start: `${customStart.year}-${customStart.month}-${customStart.day}`,
            end:   `${customEnd.year}-${customEnd.month}-${customEnd.day}`,
          } : null)}
          disabled={loading}
          style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}
        >
          {loading ? (
            <>
              <svg className="spin" width="14" height="14" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="40 20"/>
              </svg>
              Analysing...
            </>
          ) : 'Run Analysis'}
        </button>
      </div>

      {/* Resize handle — invisible until hover, drags --sidebar-width live */}
      <div
        onMouseDown={handleResizeStart}
        title="Drag to resize"
        style={{
          position: 'absolute', top: 0, right: -3, width: 6, height: '100%',
          cursor: 'col-resize', zIndex: 10, background: 'transparent',
          transition: 'background 0.15s',
        }}
        onMouseEnter={e => { e.currentTarget.style.background = 'rgba(var(--signal-positive-rgb),0.4)' }}
        onMouseLeave={e => { e.currentTarget.style.background = 'transparent' }}
      />
    </aside>
  )
}

function Section({ label, children, tourId }) {
  return (
    <div style={{ marginBottom: 18 }} data-tour={tourId}>
      <div style={{
        fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', letterSpacing: 'var(--tracking-caption)',
        textTransform: 'uppercase', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)', marginBottom: 8,
      }}>
        {label}
      </div>
      {children}
    </div>
  )
}