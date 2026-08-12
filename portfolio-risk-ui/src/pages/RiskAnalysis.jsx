import {
  LineChart, Line, AreaChart, Area,
  XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import ReturnHistogram from '../components/ReturnHistogram'

const fmt  = v => `${(v * 100).toFixed(2)}%`
const fmtS = v => `${(v * 100).toFixed(1)}%`

const AXIS = { fill: 'var(--text-muted)', fontSize: 10 }
const TIP  = {
  background: 'var(--card)', border: '1px solid var(--border)',
  borderRadius: 8, fontSize: 11, padding: '6px 10px',
}

function MiniChart({ data, dataKey, color, filled, refVal }) {
  return (
    <ResponsiveContainer width="100%" height={72}>
      {filled ? (
        <AreaChart data={data} margin={{ top: 4, right: 2, bottom: 0, left: 2 }}>
          <defs>
            <linearGradient id={`g_${dataKey}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor={color} stopOpacity={0.2}/>
              <stop offset="95%" stopColor={color} stopOpacity={0}/>
            </linearGradient>
          </defs>
          <XAxis dataKey="date" hide />
          <YAxis hide />
          {refVal != null && <ReferenceLine y={refVal} stroke="rgba(255,255,255,0.1)" strokeDasharray="3 3" />}
          <Tooltip formatter={v => fmtS(v)} contentStyle={TIP} itemStyle={{ color }} />
          <Area type="monotone" dataKey={dataKey} stroke={color} strokeWidth={1.5} fill={`url(#g_${dataKey})`} dot={false} />
        </AreaChart>
      ) : (
        <LineChart data={data} margin={{ top: 4, right: 2, bottom: 0, left: 2 }}>
          <XAxis dataKey="date" hide />
          <YAxis hide />
          {refVal != null && <ReferenceLine y={refVal} stroke="rgba(255,255,255,0.1)" strokeDasharray="3 3" />}
          <Tooltip formatter={v => v?.toFixed(2)} contentStyle={TIP} itemStyle={{ color }} />
          <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={1.5} dot={false} />
        </LineChart>
      )}
    </ResponsiveContainer>
  )
}

function MetricPill({ label, value, color, sub }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', gap: 3,
      padding: '10px 14px',
      background: 'rgba(255,255,255,0.03)',
      borderRadius: 8,
      border: '1px solid rgba(255,255,255,0.06)',
    }}>
      <div style={{ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)' }}>
        {label}
      </div>
      <div style={{ fontSize: 15, fontWeight: 700, fontFamily: 'monospace', color: color || 'var(--text-primary)' }}>
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 9, color: 'var(--text-muted)', fontFamily: 'monospace' }}>{sub}</div>
      )}
    </div>
  )
}

function CorrMatrix({ corr }) {
  const n = corr.tickers.length
  const cellPad  = n >= 5 ? '5px 3px' : n >= 4 ? '7px 4px' : '9px 6px'
  const cellFont = n >= 5 ? 9         : n >= 4 ? 10        : 11
  const headFont = n >= 5 ? 9         : 10
  const rowWidth = n >= 5 ? 32        : n >= 4 ? 38        : 44

  const cellColor = v => {
    if (v >= 0.8) return { bg: 'rgba(224,92,92,0.25)',   text: '#e05c5c' }
    if (v >= 0.5) return { bg: 'rgba(224,154,48,0.2)',   text: '#e09a30' }
    if (v >= 0.2) return { bg: 'rgba(255,255,255,0.06)', text: 'rgba(255,255,255,0.7)' }
    return            { bg: 'rgba(82,183,136,0.15)',  text: '#52b788' }
  }

  return (
    <div>
      <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: n >= 5 ? 2 : 4 }}>
        <thead>
          <tr>
            <th style={{ width: rowWidth }}/>
            {corr.tickers.map(t => (
              <th key={t} style={{
                fontSize: headFont, fontWeight: 700, color: 'var(--text-muted)',
                fontFamily: 'monospace', padding: '0 2px 6px', textAlign: 'center',
              }}>{t}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {corr.tickers.map((row, ri) => (
            <tr key={row}>
              <td style={{
                fontSize: headFont, fontWeight: 700, color: 'var(--text-muted)',
                fontFamily: 'monospace', paddingRight: 6, textAlign: 'right',
              }}>{row}</td>
              {corr.values[ri].map((val, ci) => {
                const { bg, text } = cellColor(Math.abs(val))
                return (
                  <td key={ci} style={{
                    textAlign: 'center', fontSize: cellFont, fontFamily: 'monospace',
                    fontWeight: 700, padding: cellPad, borderRadius: 5,
                    background: bg, color: text,
                  }}>
                    {val.toFixed(2)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ display: 'flex', gap: 14, marginTop: 10, fontSize: 10, color: 'var(--text-muted)' }}>
        {[
          ['≥ 0.8 High',  'rgba(224,92,92,0.3)'],
          ['0.5–0.8 Med', 'rgba(224,154,48,0.3)'],
          ['< 0.2 Low',   'rgba(82,183,136,0.25)'],
        ].map(([l, c]) => (
          <span key={l} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{ width: 8, height: 8, borderRadius: 2, background: c }}/>
            {l}
          </span>
        ))}
      </div>
    </div>
  )
}

export default function RiskAnalysis({ data, tickers, onTickerClick }) {
  if (!data) return null

  const {
    rolling_volatility: rv, rolling_sharpe: rs,
    correlation_matrix: corr, var_cvar, var_cvar_99,
    annualised_volatility: vol, sharpe_ratio: sharpe,
    portfolio_returns,
  } = data

  const rvData = rv.dates.map((d, i) => ({ date: d.slice(5), vol: rv.values[i] }))
  const rsData = rs.dates.map((d, i) => ({ date: d.slice(5), sharpe: rs.values[i] }))

  const recentVol    = rv.values[rv.values.length - 1]
  const recentSharpe = rs.values[rs.values.length - 1]
  const riskRising   = recentVol > vol

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, height: '100%', overflowY: 'auto' }}>

      {/* Row 1: Rolling charts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>

        <div className="card" style={{ padding: '14px 16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
            <div>
              <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-muted)', marginBottom: 4 }}>
                Rolling volatility
              </div>
              <div style={{ fontSize: 20, fontWeight: 700, fontFamily: 'monospace', color: riskRising ? '#e09a30' : '#52b788' }}>
                {fmtS(recentVol)}
              </div>
            </div>
            <div style={{
              fontSize: 10, fontWeight: 600, padding: '3px 8px', borderRadius: 20,
              background: riskRising ? 'rgba(224,154,48,0.12)' : 'rgba(82,183,136,0.12)',
              color: riskRising ? '#e09a30' : '#52b788',
              border: `1px solid ${riskRising ? 'rgba(224,154,48,0.3)' : 'rgba(82,183,136,0.3)'}`,
              marginTop: 2,
            }}>
              {riskRising ? '↑ Rising' : '↓ Falling'}
            </div>
          </div>
          <MiniChart data={rvData} dataKey="vol" color="#e09a30" filled refVal={vol} />
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6 }}>
            Avg <span style={{ color: 'var(--text-secondary)', fontWeight: 600, fontFamily: 'monospace' }}>{fmtS(vol)}</span>
            <span style={{ marginLeft: 8, opacity: 0.5 }}>dashed line = long-run average</span>
          </div>
        </div>

        <div className="card" style={{ padding: '14px 16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
            <div>
              <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-muted)', marginBottom: 4 }}>
                Rolling Sharpe
              </div>
              <div style={{ fontSize: 20, fontWeight: 700, fontFamily: 'monospace', color: recentSharpe > 1 ? '#52b788' : recentSharpe > 0 ? '#e09a30' : '#e05c5c' }}>
                {recentSharpe.toFixed(2)}
              </div>
            </div>
            <div style={{
              fontSize: 10, fontWeight: 600, padding: '3px 8px', borderRadius: 20,
              background: recentSharpe > sharpe ? 'rgba(82,183,136,0.12)' : 'rgba(224,154,48,0.12)',
              color: recentSharpe > sharpe ? '#52b788' : '#e09a30',
              border: `1px solid ${recentSharpe > sharpe ? 'rgba(82,183,136,0.3)' : 'rgba(224,154,48,0.3)'}`,
              marginTop: 2,
            }}>
              {recentSharpe > sharpe ? '↑ Improving' : '↓ Weakening'}
            </div>
          </div>
          <MiniChart data={rsData} dataKey="sharpe" color="#52b788" filled={false} refVal={sharpe} />
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6 }}>
            Avg <span style={{ color: 'var(--text-secondary)', fontWeight: 600, fontFamily: 'monospace' }}>{sharpe.toFixed(2)}</span>
            <span style={{ marginLeft: 8, opacity: 0.5 }}>above 1.0 is strong</span>
          </div>
        </div>
      </div>

      {/* Row 2: VaR + Correlation */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>

        {/* Downside risk — 95% and 99% side by side */}
        <div className="card" style={{ padding: '14px 16px' }}>
          <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-muted)', marginBottom: 12 }}>
            Downside risk
          </div>

          {/* Confidence level headers */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 6 }}>
            {[
              { label: '95% confidence', color: '#e05c5c' },
              { label: '99% confidence', color: '#b03a3a' },
            ].map(({ label, color }) => (
              <div key={label} style={{
                fontSize: 9, fontWeight: 700, textTransform: 'uppercase',
                letterSpacing: '0.07em', color, textAlign: 'center',
                padding: '3px 0', borderBottom: `1px solid ${color}30`,
              }}>
                {label}
              </div>
            ))}
          </div>

          {/* VaR row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 6 }}>
            <MetricPill
              label="VaR"
              value={`−${fmt(var_cvar.var_pct)}`}
              color="#e05c5c"
              sub={`$${Math.abs(var_cvar.var_dollar).toFixed(0)} per day`}
            />
            <MetricPill
              label="VaR"
              value={`−${fmt(var_cvar_99?.var_pct ?? 0)}`}
              color="#b03a3a"
              sub={`$${Math.abs(var_cvar_99?.var_dollar ?? 0).toFixed(0)} per day`}
            />
          </div>

          {/* CVaR row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 12 }}>
            <MetricPill
              label="CVaR (avg tail)"
              value={`−${fmt(var_cvar.cvar_pct)}`}
              color="#e05c5c"
              sub={`$${Math.abs(var_cvar.cvar_dollar).toFixed(0)} avg`}
            />
            <MetricPill
              label="CVaR (avg tail)"
              value={`−${fmt(var_cvar_99?.cvar_pct ?? 0)}`}
              color="#b03a3a"
              sub={`$${Math.abs(var_cvar_99?.cvar_dollar ?? 0).toFixed(0)} avg`}
            />
          </div>

          <div style={{
            fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6,
            padding: '10px 12px',
            background: 'rgba(224,92,92,0.05)', borderRadius: 8,
            border: '1px solid rgba(224,92,92,0.12)',
          }}>
            On your worst <strong style={{ color: 'rgba(255,255,255,0.7)' }}>5% of days</strong>, expect to lose up to{' '}
            <strong style={{ color: '#e05c5c' }}>${Math.abs(var_cvar.var_dollar).toFixed(0)}</strong>.
            {' '}The 99% threshold rises to{' '}
            <strong style={{ color: '#b03a3a' }}>${Math.abs(var_cvar_99?.var_dollar ?? 0).toFixed(0)}</strong> — the worst 1 in 100 days.
          </div>
        </div>

        <div className="card" style={{ padding: '14px 16px' }}>
          <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-muted)', marginBottom: 14 }}>
            Correlation matrix
          </div>
          <CorrMatrix corr={corr} />
          <div style={{
            fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6,
            padding: '10px 12px', marginTop: 12,
            background: 'rgba(255,255,255,0.02)', borderRadius: 8,
            border: '1px solid rgba(255,255,255,0.05)',
          }}>
            High correlation means your stocks move together — <strong style={{ color: 'rgba(255,255,255,0.65)' }}>less diversification</strong> than you might think.
          </div>
        </div>
      </div>

      {/* Row 3: Returns distribution */}
      {portfolio_returns && (
        <div className="card" style={{ padding: '14px 16px' }}>
          <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-muted)', marginBottom: 14 }}>
            Daily returns distribution
          </div>
          <ReturnHistogram
            portfolioReturns={portfolio_returns}
            varPct={var_cvar.var_pct}
            cvarPct={var_cvar.cvar_pct}
            confidence={var_cvar.confidence}
          />
        </div>
      )}

    </div>
  )
}