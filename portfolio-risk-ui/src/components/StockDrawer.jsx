import { useEffect, useState, useRef } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const fmt  = v => v != null ? `${(v * 100).toFixed(1)}%` : '—'
const fmtM = v => {
  if (v == null) return '—'
  if (v >= 1e12) return `$${(v / 1e12).toFixed(2)}T`
  if (v >= 1e9)  return `$${(v / 1e9).toFixed(1)}B`
  if (v >= 1e6)  return `$${(v / 1e6).toFixed(1)}M`
  return `$${v.toLocaleString()}`
}
const fmtP = v => v != null ? `$${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '—'
const fmtN = v => v != null ? v.toFixed(2) : '—'

function Row({ label, value, accent }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.05)',
    }}>
      <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.4)', fontFamily: 'var(--font-mono, monospace)' }}>
        {label}
      </span>
      <span style={{
        fontSize: 11, fontWeight: 600, fontFamily: 'monospace',
        color: accent || 'rgba(255,255,255,0.85)',
      }}>
        {value}
      </span>
    </div>
  )
}

function Sparkline({ data }) {
  if (!data?.length) return null
  const chartData = data.map((v, i) => ({ i, v }))
  const isUp = data[data.length - 1] >= data[0]
  const color = isUp ? '#52b788' : '#e05c5c'
  return (
    <ResponsiveContainer width="100%" height={64}>
      <AreaChart data={chartData} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id="sdGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={color} stopOpacity={0.3} />
            <stop offset="95%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <XAxis dataKey="i" hide />
        <YAxis domain={['auto', 'auto']} hide />
        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null
            return (
              <div style={{
                background: '#1e2e1e', border: '1px solid rgba(82,183,136,0.3)',
                borderRadius: 6, padding: '4px 8px', fontSize: 10,
                color: '#52b788', fontFamily: 'monospace',
              }}>
                {fmtP(payload[0].value)}
              </div>
            )
          }}
        />
        <Area
          type="monotone" dataKey="v"
          stroke={color} strokeWidth={1.5}
          fill="url(#sdGrad)" dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}

export default function StockDrawer({ ticker, weight, onClose }) {
  const [stock, setStock]     = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [visible, setVisible] = useState(false)
  const drawerRef = useRef(null)

  // Animate in
  useEffect(() => {
    if (ticker) {
      setVisible(false)
      requestAnimationFrame(() => requestAnimationFrame(() => setVisible(true)))
    }
  }, [ticker])

  // Fetch data
  useEffect(() => {
    if (!ticker) return
    setLoading(true)
    setError(null)
    setStock(null)
    fetch(`${API}/stock-detail/${ticker}`)
      .then(async r => {
        const d = await r.json()
        if (!r.ok) {
          const message = r.status === 503
            ? 'Yahoo Finance is rate-limiting right now, try again in a moment'
            : (d.detail || 'Could not load data')
          throw new Error(message)
        }
        setStock(d)
        setLoading(false)
      })
      .catch(e => { setError(e.message || 'Could not load data'); setLoading(false) })
  }, [ticker])

  // Close on Escape
  useEffect(() => {
    const handler = e => { if (e.key === 'Escape') handleClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const handleClose = () => {
    setVisible(false)
    setTimeout(onClose, 220)
  }

  if (!ticker) return null

  const isUp = stock?.change_pct >= 0
  const changeColor = isUp ? '#52b788' : '#e05c5c'

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={handleClose}
        style={{
          position: 'fixed', inset: 0, zIndex: 99,
          background: 'rgba(0,0,0,0.25)',
          opacity: visible ? 1 : 0,
          transition: 'opacity 0.2s ease',
        }}
      />

      {/* Drawer */}
      <div
        ref={drawerRef}
        style={{
          position: 'fixed', top: 48, right: 0, bottom: 0, zIndex: 100,
          width: 300,
          background: '#141f14',
          borderLeft: '1px solid rgba(82,183,136,0.2)',
          display: 'flex', flexDirection: 'column',
          transform: visible ? 'translateX(0)' : 'translateX(100%)',
          transition: 'transform 0.22s cubic-bezier(0.4, 0, 0.2, 1)',
          overflowY: 'auto',
        }}
      >
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between',
          padding: '16px 18px 12px',
          borderBottom: '1px solid rgba(255,255,255,0.07)',
          flexShrink: 0,
        }}>
          <div>
            <div style={{ fontSize: 22, fontWeight: 700, color: '#fff', fontFamily: 'monospace', letterSpacing: '0.02em' }}>
              {ticker}
            </div>
            {stock?.company_name && (
              <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.4)', marginTop: 2, maxWidth: 200, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {stock.company_name}
              </div>
            )}
          </div>
          <button
            onClick={handleClose}
            style={{
              background: 'rgba(255,255,255,0.07)', border: 'none',
              color: 'rgba(255,255,255,0.5)', cursor: 'pointer',
              width: 28, height: 28, borderRadius: 6,
              fontSize: 14, display: 'flex', alignItems: 'center', justifyContent: 'center',
              flexShrink: 0, marginTop: 2,
            }}
          >
            ✕
          </button>
        </div>

        {/* Body */}
        <div style={{ padding: '14px 18px', flex: 1, display: 'flex', flexDirection: 'column', gap: 16 }}>

          {loading && (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 120 }}>
              <svg className="spin" width="24" height="24" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="#52b788" strokeWidth="3" strokeDasharray="40 20"/>
              </svg>
            </div>
          )}

          {error && (
            <div style={{ fontSize: 12, color: '#e05c5c', textAlign: 'center', padding: '20px 0' }}>
              {error}
            </div>
          )}

          {stock && (
            <>
              {/* Price + change */}
              <div>
                <div style={{ fontSize: 28, fontWeight: 700, color: '#fff', fontFamily: 'monospace' }}>
                  {fmtP(stock.price)}
                </div>
                <div style={{ display: 'flex', gap: 8, marginTop: 4, alignItems: 'center' }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: changeColor, fontFamily: 'monospace' }}>
                    {isUp ? '▲' : '▼'} {fmtP(Math.abs(stock.change))}
                  </span>
                  <span style={{
                    fontSize: 11, fontWeight: 600, color: changeColor,
                    background: isUp ? 'rgba(82,183,136,0.12)' : 'rgba(224,92,92,0.12)',
                    padding: '2px 6px', borderRadius: 4, fontFamily: 'monospace',
                  }}>
                    {isUp ? '+' : ''}{(stock.change_pct * 100).toFixed(2)}%
                  </span>
                  <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)' }}>today</span>
                </div>
              </div>

              {/* Sparkline */}
              <div style={{
                background: 'rgba(255,255,255,0.03)', borderRadius: 8,
                padding: '8px 4px 4px',
                border: '1px solid rgba(255,255,255,0.06)',
              }}>
                <div style={{ fontSize: 9, color: 'rgba(255,255,255,0.3)', marginLeft: 8, marginBottom: 2, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                  30-day price
                </div>
                <Sparkline data={stock.sparkline} />
              </div>

              {/* Portfolio contribution */}
              {weight != null && (
                <div style={{
                  background: 'rgba(82,183,136,0.06)',
                  border: '1px solid rgba(82,183,136,0.18)',
                  borderRadius: 8, padding: '10px 14px',
                }}>
                  <div style={{ fontSize: 9, color: 'rgba(82,183,136,0.6)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
                    Portfolio position
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                    <div>
                      <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)' }}>Weight</div>
                      <div style={{ fontSize: 16, fontWeight: 700, color: '#52b788', fontFamily: 'monospace' }}>
                        {Math.round(weight * 100)}%
                      </div>
                    </div>
                    {stock.sector && (
                      <div>
                        <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)' }}>Sector</div>
                        <div style={{ fontSize: 11, fontWeight: 600, color: 'rgba(255,255,255,0.7)', marginTop: 2, lineHeight: 1.3 }}>
                          {stock.sector}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Fundamentals */}
              <div>
                <div style={{ fontSize: 9, color: 'rgba(255,255,255,0.3)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
                  Fundamentals
                </div>
                <Row label="Market Cap"     value={fmtM(stock.market_cap)} />
                <Row label="P/E Ratio"      value={fmtN(stock.pe_ratio)} />
                <Row label="EPS (TTM)"      value={stock.eps != null ? `$${stock.eps.toFixed(2)}` : '—'} />
                <Row label="52w High"       value={fmtP(stock.week_52_high)} accent="#52b788" />
                <Row label="52w Low"        value={fmtP(stock.week_52_low)}  accent="#e05c5c" />
                <Row label="Beta"           value={fmtN(stock.beta)} />
                <Row label="Dividend Yield" value={stock.dividend_yield ? fmt(stock.dividend_yield) : 'None'} />
                <Row label="Avg Volume"     value={stock.avg_volume != null ? `${(stock.avg_volume / 1e6).toFixed(1)}M` : '—'} />
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div style={{
          padding: '10px 18px', borderTop: '1px solid rgba(255,255,255,0.06)',
          fontSize: 9, color: 'rgba(255,255,255,0.2)', flexShrink: 0,
        }}>
          Data via Yahoo Finance · 15min delay
        </div>
      </div>
    </>
  )
}