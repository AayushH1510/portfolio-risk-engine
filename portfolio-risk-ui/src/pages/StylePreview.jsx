import { useEffect, useRef, useState } from 'react'
import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip, ResponsiveContainer,
} from 'recharts'

// ─────────────────────────────────────────────────────────────────────────
// Throwaway visual-approval preview for DESIGN.md. Hand-built markup only —
// nothing here reuses or touches an existing component. Not linked from any
// nav; reachable only by navigating directly to /style-preview.
// ─────────────────────────────────────────────────────────────────────────

const TOKENS = `
.sp-root {
  --surface-canvas:    #1d1a18;
  --surface-elevated:  #262321;
  --surface-card:      #3d3a39;
  --surface-raised:    #4d4947;

  --line-hairline:     #4d4947;
  --line-emphasis:     #5f5a57;
  --line-faint:        #2e2b29;

  --text-primary:      #f2efec;
  --text-secondary:    #a39d98;
  --text-muted:        #7a736e;
  --text-faint:        #5a544f;

  --signal-positive:       #52b788;
  --signal-negative:       #e0574f;
  --signal-caution:        #d99a3c;
  --signal-positive-wash:  #52b7881a;
  --signal-negative-wash:  #e0574f1a;
  --signal-caution-wash:   #d99a3c1a;

  --font-primary: 'Geist Sans', ui-sans-serif, system-ui, -apple-system, sans-serif;
  --font-mono:    'Geist Mono', ui-monospace, 'SF Mono', Menlo, monospace;

  --border-default:  1px solid var(--line-hairline);
  --border-emphasis: 1px solid var(--line-emphasis);
  --border-faint:    1px solid var(--line-faint);

  background: var(--surface-canvas);
  color: var(--text-primary);
  font-family: var(--font-primary);
  min-height: 100vh;
  font-feature-settings: "tnum" on, "zero" on;
}
.sp-root * { box-sizing: border-box; }
.sp-mono { font-family: var(--font-mono); font-feature-settings: "tnum" on, "zero" on; }

.sp-caption {
  font-size: 10px; font-weight: 500; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--text-muted);
}
.sp-micro { font-size: 11px; font-weight: 400; letter-spacing: 0.04em; color: var(--text-muted); }
.sp-body-sm { font-size: 12px; font-weight: 400; letter-spacing: 0; color: var(--text-secondary); }

.sp-btn-primary {
  background: var(--signal-positive); color: var(--surface-canvas);
  border: none; padding: 10px 20px; font-family: var(--font-primary);
  font-size: 12px; font-weight: 500; letter-spacing: 0.02em; text-transform: uppercase;
  cursor: pointer; transition: filter 120ms ease-out;
}
.sp-btn-primary:hover { filter: brightness(1.08); }

.sp-btn-ghost {
  background: transparent; color: var(--text-primary);
  border: var(--border-default); padding: 10px 20px; font-family: var(--font-primary);
  font-size: 12px; font-weight: 500; letter-spacing: 0.02em; text-transform: uppercase;
  cursor: pointer; transition: border-color 120ms ease-out, background 120ms ease-out;
}
.sp-btn-ghost:hover { border-color: var(--line-emphasis); background: var(--surface-elevated); }

.sp-tab {
  background: none; border: none; cursor: pointer;
  font-family: var(--font-primary); font-size: 12px; font-weight: 500;
  letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--text-muted); padding: 12px 16px;
  border-bottom: 2px solid transparent; margin-bottom: -1px;
  transition: color 200ms cubic-bezier(0.4,0,0.2,1), border-color 200ms cubic-bezier(0.4,0,0.2,1);
}
.sp-tab.active { color: var(--text-primary); border-bottom-color: var(--signal-positive); }

.sp-table { width: 100%; border-collapse: collapse; }
.sp-table th {
  font-size: 10px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--text-muted); text-align: left; padding: 8px 12px;
  border-bottom: var(--border-default);
}
.sp-table th.num, .sp-table td.num { text-align: right; }
.sp-table td {
  font-size: 12px; color: var(--text-secondary); padding: 10px 12px;
  border-bottom: var(--border-faint);
}
.sp-table tbody tr { transition: background 120ms ease-out; }
.sp-table tbody tr:hover { background: var(--surface-elevated); }

@keyframes sp-fade-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.sp-card-enter { animation: sp-fade-in 300ms ease-out both; }
`

function Card({ children, style, className = '', label, delay = 0 }) {
  return (
    <div
      className={`sp-card-enter ${className}`}
      style={{
        background: 'var(--surface-card)', border: 'var(--border-default)',
        borderRadius: 0, padding: '16px 20px', animationDelay: `${delay}ms`, ...style,
      }}
    >
      {label && <div className="sp-caption" style={{ marginBottom: 12 }}>{label}</div>}
      {children}
    </div>
  )
}

function useCountUp(target, durationMs = 600) {
  const [value, setValue] = useState(0)
  const startRef = useRef(null)
  useEffect(() => {
    let raf
    const step = (t) => {
      if (startRef.current === null) startRef.current = t
      const elapsed = t - startRef.current
      const pct = Math.min(1, elapsed / durationMs)
      const eased = 1 - Math.pow(1 - pct, 3) // ease-out cubic
      setValue(target * eased)
      if (pct < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [target, durationMs])
  return value
}

function MetricCard({ label, value, suffix = '', signal, sub, decimals = 2, delay }) {
  const numeric = useCountUp(value, 600)
  const color = signal === 'positive' ? 'var(--signal-positive)'
    : signal === 'negative' ? 'var(--signal-negative)'
    : 'var(--text-primary)'
  const topBorder = signal && signal !== 'neutral' ? `2px solid ${color}` : 'var(--border-default)'

  return (
    <Card delay={delay} style={{ borderTop: topBorder, flex: 1 }}>
      <div className="sp-caption" style={{ marginBottom: 12 }}>{label}</div>
      <div className="sp-mono" style={{
        fontSize: 32, fontWeight: 600, letterSpacing: '-0.045em', lineHeight: 1.12, color,
      }}>
        {signal === 'negative' && numeric > 0 ? '−' : ''}{numeric.toFixed(decimals)}{suffix}
      </div>
      {sub && <div className="sp-micro" style={{ marginTop: 6 }}>{sub}</div>}
    </Card>
  )
}

function SquareDot(props) {
  const { cx, cy, stroke } = props
  if (cx == null || cy == null) return null
  return <rect x={cx - 1.5} y={cy - 1.5} width={3} height={3} fill={stroke} />
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--surface-elevated)', border: 'var(--border-emphasis)',
      borderRadius: 0, padding: '10px 12px', maxWidth: 280,
    }}>
      <div className="sp-caption" style={{ color: 'var(--text-primary)', marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} className="sp-mono sp-body-sm" style={{ color: p.color }}>
          {p.name}: {p.value.toFixed(1)}%
        </div>
      ))}
    </div>
  )
}

function HoverTooltip({ children, title, body }) {
  const [open, setOpen] = useState(false)
  return (
    <span
      style={{ position: 'relative', display: 'inline-block' }}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <span style={{
        borderBottom: '1px dashed var(--text-faint)', cursor: 'help',
        color: 'var(--text-secondary)', fontSize: 12,
      }}>
        {children}
      </span>
      {open && (
        <div style={{
          position: 'absolute', top: '100%', left: 0, marginTop: 8,
          background: 'var(--surface-elevated)', border: 'var(--border-emphasis)',
          borderRadius: 0, padding: '10px 12px', maxWidth: 280, width: 240,
          zIndex: 20, animation: 'sp-fade-in 120ms ease-out both',
        }}>
          <div className="sp-caption" style={{ color: 'var(--text-primary)', marginBottom: 4 }}>{title}</div>
          <div className="sp-body-sm">{body}</div>
        </div>
      )}
    </span>
  )
}

const CHART_DATA = [
  { date: 'Jan', portfolio: 0, benchmark: 0 },
  { date: 'Feb', portfolio: 3.2, benchmark: 1.8 },
  { date: 'Mar', portfolio: 2.1, benchmark: 2.4 },
  { date: 'Apr', portfolio: 6.8, benchmark: 3.1 },
  { date: 'May', portfolio: 5.4, benchmark: 3.9 },
  { date: 'Jun', portfolio: 9.7, benchmark: 5.2 },
  { date: 'Jul', portfolio: 8.9, benchmark: 5.8 },
  { date: 'Aug', portfolio: 13.2, benchmark: 6.9 },
  { date: 'Sep', portfolio: 11.8, benchmark: 7.1 },
  { date: 'Oct', portfolio: 16.4, benchmark: 8.0 },
  { date: 'Nov', portfolio: 15.1, benchmark: 8.6 },
  { date: 'Dec', portfolio: 19.6, benchmark: 9.4 },
]

const TABLE_ROWS = [
  { ticker: 'AAPL', weight: '34.0%', price: 309.90, change: -0.14 },
  { ticker: 'MSFT', weight: '33.0%', price: 491.71, change: 0.90 },
  { ticker: 'GOOGL', weight: '33.0%', price: 264.12, change: 1.42 },
]

const TABS = ['Dashboard', 'Risk Analysis', 'Monte Carlo', 'Valuation', 'Learn']

export default function StylePreview() {
  const [activeTab, setActiveTab] = useState('Dashboard')

  useEffect(() => {
    const weights = ['400.css', '500.css', '600.css']
    const links = weights.flatMap(f => [
      { href: `https://cdn.jsdelivr.net/npm/@fontsource/geist-sans@5.2.5/${f}`, id: `sp-font-sans-${f}` },
      { href: `https://cdn.jsdelivr.net/npm/@fontsource/geist-mono@5.2.5/${f}`, id: `sp-font-mono-${f}` },
    ])
    const created = []
    links.forEach(({ href, id }) => {
      if (document.getElementById(id)) return
      const link = document.createElement('link')
      link.id = id
      link.rel = 'stylesheet'
      link.href = href
      document.head.appendChild(link)
      created.push(link)
    })
    return () => created.forEach(l => l.remove())
  }, [])

  return (
    <div className="sp-root">
      <style>{TOKENS}</style>

      <div style={{ display: 'flex', minHeight: '100vh' }}>

        {/* Sidebar */}
        <div style={{
          width: 260, minWidth: 260, background: 'var(--surface-canvas)',
          borderRight: 'var(--border-default)', padding: '20px 16px',
        }}>
          <div className="sp-mono" style={{ fontSize: 18, fontWeight: 600, letterSpacing: '-0.02em', marginBottom: 4 }}>
            Varense
          </div>
          <div className="sp-micro">style preview</div>

          <div className="sp-caption" style={{ marginTop: 20, marginBottom: 8 }}>Portfolio</div>
          {['AAPL', 'MSFT', 'GOOGL'].map(t => (
            <div key={t} className="sp-mono sp-body-sm" style={{ padding: '6px 0', color: 'var(--text-secondary)' }}>
              {t}
            </div>
          ))}

          <div className="sp-caption" style={{ marginTop: 20, marginBottom: 8 }}>Settings</div>
          <div className="sp-body-sm" style={{ padding: '6px 0' }}>Period — 1Y</div>
          <div className="sp-body-sm" style={{ padding: '6px 0' }}>Benchmark — S&amp;P 500</div>
        </div>

        {/* Main */}
        <div style={{ flex: 1, padding: '0 32px 48px', maxWidth: 1440 }}>

          {/* Tab bar */}
          <div style={{ display: 'flex', gap: 4, borderBottom: 'var(--border-default)', marginTop: 4 }}>
            {TABS.map(t => (
              <button
                key={t}
                className={`sp-tab ${activeTab === t ? 'active' : ''}`}
                onClick={() => setActiveTab(t)}
              >
                {t}
              </button>
            ))}
          </div>

          {/* Display headline */}
          <div style={{ marginTop: 40, marginBottom: 40 }}>
            <div className="sp-caption" style={{ marginBottom: 8 }}>Portfolio value</div>
            <div className="sp-mono" style={{
              fontSize: 64, fontWeight: 600, letterSpacing: '-0.06em', lineHeight: 1.0, color: 'var(--text-primary)',
            }}>
              $142,384.20
            </div>
            <div className="sp-body-sm" style={{ marginTop: 8 }}>
              <span style={{ color: 'var(--signal-positive)' }}>+$12,940.55 (+9.7%)</span>
              <span style={{ color: 'var(--text-muted)', marginLeft: 8 }}>past 12 months</span>
            </div>
          </div>

          {/* Metric cards row */}
          <div style={{ display: 'flex', gap: 12, marginBottom: 32 }}>
            <MetricCard label="Sharpe Ratio" value={2.52} signal="neutral" sub="above 1.0 is good" delay={0} />
            <MetricCard label="Annual Return" value={34.8} suffix="%" signal="positive" sub="beat S&P by 15.9%" delay={40} />
            <MetricCard label="Max Drawdown" value={17.6} suffix="%" signal="negative" sub="worst peak-to-trough" delay={80} />
          </div>

          {/* Chart card */}
          <Card label="Portfolio Growth" delay={120} style={{ marginBottom: 32 }}>
            <div style={{ height: 260 }}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={CHART_DATA} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="var(--line-faint)" horizontal vertical={false} />
                  <XAxis
                    dataKey="date" tickLine={false} axisLine={false}
                    tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                  />
                  <YAxis
                    tickLine={false} axisLine={false}
                    tickFormatter={v => `${v}%`}
                    tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                    width={36}
                  />
                  <RTooltip content={<ChartTooltip />} />
                  <Area
                    type="monotone" dataKey="portfolio" name="Portfolio"
                    stroke="var(--signal-positive)" strokeWidth={1.5} strokeLinecap="butt"
                    fill="var(--signal-positive)" fillOpacity={0.08}
                    dot={false} activeDot={<SquareDot />}
                    isAnimationActive animationDuration={700} animationEasing="ease"
                  />
                  <Line
                    type="monotone" dataKey="benchmark" name="Benchmark"
                    stroke="var(--text-muted)" strokeWidth={1.5} strokeLinecap="butt"
                    dot={false} activeDot={<SquareDot />}
                    isAnimationActive animationDuration={700} animationEasing="ease"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </Card>

          {/* Buttons + tooltip demo */}
          <Card label="Controls" delay={160} style={{ marginBottom: 32 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
              <button className="sp-btn-primary">Run Analysis</button>
              <button className="sp-btn-ghost">Export CSV</button>
              <HoverTooltip title="Sharpe Ratio" body="Return earned per unit of overall risk taken. Above 1.0 is generally considered good.">
                What is Sharpe Ratio?
              </HoverTooltip>
            </div>
          </Card>

          {/* Table */}
          <Card label="Holdings" delay={200}>
            <table className="sp-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Weight</th>
                  <th className="num">Price</th>
                  <th className="num">Change</th>
                </tr>
              </thead>
              <tbody>
                {TABLE_ROWS.map(r => (
                  <tr key={r.ticker}>
                    <td className="sp-mono" style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{r.ticker}</td>
                    <td className="sp-mono">{r.weight}</td>
                    <td className="sp-mono num">${r.price.toFixed(2)}</td>
                    <td
                      className="sp-mono num"
                      style={{ color: r.change >= 0 ? 'var(--signal-positive)' : 'var(--signal-negative)' }}
                    >
                      {r.change >= 0 ? '+' : '−'}{Math.abs(r.change).toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

        </div>
      </div>
    </div>
  )
}
