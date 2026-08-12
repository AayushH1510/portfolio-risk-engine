import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import MetricCard from '../components/MetricCard'
import RiskGauge from '../components/RiskGauge'
import InsightBox from '../components/InsightBox'
import SectorChart from '../components/SectorChart'

const fmt  = v => `${(v * 100).toFixed(1)}%`
const fmtD = v => `$${v.toLocaleString('en-US', { maximumFractionDigits: 0 })}`
const fmtDec = v => `$${Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: 0 })}`

const CHART_STYLE = {
  background: 'transparent',
  fontSize: 11,
  fontFamily: 'monospace',
}

const AXIS_STYLE = { fill: 'var(--text-muted)', fontSize: 10 }

function CustomTooltip({ active, payload, label, prefix = '', pct = false }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--card)', border: '1px solid var(--border)',
      borderRadius: 8, padding: '8px 12px', fontSize: 11,
    }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || 'var(--text-primary)', fontWeight: 600 }}>
          {p.name}: {pct ? fmt(p.value) : `${prefix}${p.value?.toFixed(2)}`}
        </div>
      ))}
    </div>
  )
}

export default function Dashboard({ data, tickers, weights, portfolioValue, onTickerClick, sectorData }) {
  if (!data) return null

  const { annualised_return: ret, annualised_volatility: vol,
          sharpe_ratio: sharpe, sortino_ratio: sortino,
          max_drawdown: dd, var_cvar, beta_alpha: ba,
          cumulative_returns: cum, benchmark_cumulative: bench,
          drawdown_series: ddSeries, period,
          diversification_score: divScore } = data

  const medianFinal = data.monte_carlo?.p50_final
  const profitProb  = data.monte_carlo?.prob_profit

  const growthData = cum.dates.map((d, i) => {
    const row = { date: d.slice(5), portfolio: cum.values[i] }
    if (bench?.values[i] != null) row.benchmark = bench.values[i]
    return row
  })

  const ddData = ddSeries.dates.map((d, i) => ({
    date: d.slice(5), drawdown: ddSeries.values[i],
  }))

  const sharpeColor = sharpe > 2 ? 'good' : sharpe > 1 ? 'good' : sharpe > 0 ? 'warning' : 'bad'
  const ddColor     = Math.abs(dd) < 0.15 ? 'good' : Math.abs(dd) < 0.30 ? 'warning' : 'bad'
  const retColor    = ret > 0.15 ? 'good' : ret > 0 ? 'warning' : 'bad'

  // Diversification score tone
  const divTone  = !divScore ? 'neutral'
    : divScore.score >= 70 ? 'good'
    : divScore.score >= 40 ? 'warning'
    : 'bad'

  // Second row: VaR, CVaR, Diversification, Beta (if available), Alpha (if available)
  // Dynamic columns based on what's available
  const secondRowItems = [
    { label: 'VaR 95%',  value: `-${fmtDec(var_cvar.var_dollar)}`,  tone: 'bad',    small: true },
    { label: 'CVaR 95%', value: `-${fmtDec(var_cvar.cvar_dollar)}`, tone: 'bad',    small: true },
    divScore ? { label: 'Diversification', value: `${divScore.score}/100`, tone: divTone, small: true, sub: `${divScore.label} · avg corr ${divScore.avg_pairwise_corr.toFixed(2)} · ${period.n_years}yr window` } : null,
    ba ? { label: 'Beta',  value: ba.beta.toFixed(2),  tone: ba.beta > 1.5 ? 'warning' : 'neutral', small: true } : null,
    ba ? { label: 'Alpha', value: fmt(ba.alpha),        tone: ba.alpha > 0 ? 'good' : 'bad',         small: true } : null,
  ].filter(Boolean)

  const secondCols = `repeat(${secondRowItems.length}, 1fr)`

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, height: '100%' }}>

      {/* Period pills */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        {[period.start, period.end, `${period.n_days} days`, `${period.n_years}yr`].map((pill, i) => (
          <div key={i} style={{
            fontSize: 10, fontWeight: 600, letterSpacing: '0.05em',
            background: 'rgba(255,255,255,0.07)', border: '1px solid var(--border)',
            borderRadius: 4, padding: '3px 8px', color: 'var(--text-muted)',
            fontFamily: 'monospace',
          }}>
            {pill}
          </div>
        ))}
      </div>

      {/* Top metrics row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 10 }}>
        <MetricCard label="Annual Return"  value={fmt(ret)}           tone={retColor}   sub={ba ? `${fmt(ba.alpha)} alpha` : null} />
        <MetricCard label="Volatility"     value={fmt(vol)}           tone={vol < 0.2 ? 'good' : vol < 0.35 ? 'warning' : 'bad'} />
        <MetricCard label="Sharpe Ratio"   value={sharpe.toFixed(2)}  tone={sharpeColor} sub="above 1.0 is good" />
        <MetricCard label="Sortino Ratio"  value={sortino.toFixed(2)} tone={sharpeColor} />
        <MetricCard label="Max Drawdown"   value={fmt(dd)}            tone={ddColor} />
      </div>

      {/* Second metrics row — dynamic */}
      <div style={{ display: 'grid', gridTemplateColumns: secondCols, gap: 10 }}>
        {secondRowItems.map(item => (
          <MetricCard
            key={item.label}
            label={item.label}
            value={item.value}
            tone={item.tone}
            small={item.small}
            sub={item.sub}
          />
        ))}
      </div>

      {/* Sector exposure */}
      <SectorChart sectorData={sectorData} />

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 220px', gap: 10, flex: 1, minHeight: 0 }}>

        {/* Growth chart */}
        <div className="card" style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-secondary)' }}>
              Portfolio growth
            </div>
            <div style={{ display: 'flex', gap: 10, fontSize: 10 }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-muted)' }}>
                <span style={{ width: 20, height: 2, background: 'var(--accent)', display: 'inline-block', borderRadius: 1 }} />
                Your portfolio
              </span>
              {bench && (
                <span style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-muted)' }}>
                  <span style={{ width: 20, height: 2, display: 'inline-block', borderRadius: 1, opacity: 0.6, borderBottom: '2px dashed #5a7a5a' }} />
                  S&P 500
                </span>
              )}
            </div>
          </div>
          <div style={{ flex: 1, minHeight: 0 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={growthData} style={CHART_STYLE} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                <defs>
                  <linearGradient id="gPort" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#52b788" stopOpacity={0.25}/>
                    <stop offset="95%" stopColor="#52b788" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="date" tick={AXIS_STYLE} tickLine={false} axisLine={false} interval="preserveStartEnd" />
                <YAxis tickFormatter={v => `${(v*100).toFixed(0)}%`} tick={AXIS_STYLE} tickLine={false} axisLine={false} width={42} />
                <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" strokeDasharray="4 4" />
                <Tooltip content={<CustomTooltip pct />} />
                <Area type="monotone" dataKey="portfolio" stroke="#52b788" strokeWidth={2} fill="url(#gPort)" dot={false} name="Portfolio" />
                {bench && <Line type="monotone" dataKey="benchmark" stroke="#5a7a5a" strokeWidth={1.5} strokeDasharray="5 4" dot={false} name="S&P 500" />}
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <InsightBox
            label="Return"
            tone={retColor}
            text={`Portfolio grew at <strong>${fmt(ret)}/yr</strong>. ${ba ? `Beat the S&P 500 by <strong>${fmt(ret - ba.benchmark_return)}</strong>.` : ''} Median 1-year projection: <strong>${fmtD(medianFinal || 0)}</strong> (${Math.round((profitProb || 0) * 100)}% chance of profit).`}
          />
        </div>

        {/* Right column — Risk gauge + Drawdown */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, minHeight: 0 }}>

          <RiskGauge
            vol={vol}
            drawdown={dd}
            varPct={var_cvar.var_pct}
          />

          <div className="card" style={{ padding: '14px 16px', flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-muted)', marginBottom: 10 }}>
              Drawdown
            </div>
            <div style={{ flex: 1, minHeight: 0 }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={ddData} style={CHART_STYLE} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                  <defs>
                    <linearGradient id="gDD" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#e05c5c" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#e05c5c" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="date" tick={{ ...AXIS_STYLE, fontSize: 9 }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
                  <YAxis tickFormatter={v => `${(v*100).toFixed(0)}%`} tick={{ ...AXIS_STYLE, fontSize: 9 }} tickLine={false} axisLine={false} width={36} />
                  <Tooltip content={<CustomTooltip pct />} />
                  <Area type="monotone" dataKey="drawdown" stroke="#e05c5c" strokeWidth={1.5} fill="url(#gDD)" dot={false} name="Drawdown" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div style={{ fontSize: 11, marginTop: 8 }}>
              <span style={{ color: 'var(--text-muted)' }}>Worst drop: </span>
              <span style={{ color: 'var(--negative)', fontWeight: 600, fontFamily: 'monospace' }}>{fmt(dd)}</span>
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}