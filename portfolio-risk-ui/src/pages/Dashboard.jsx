import { useState } from 'react'
import {
  AreaChart, Area, LineChart, Line, ComposedChart,
  XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import MetricCard from '../components/MetricCard'
import RiskGauge from '../components/RiskGauge'
import InsightBox from '../components/InsightBox'
import SectorChart from '../components/SectorChart'
import FirstResultCallout from '../components/FirstResultCallout'
import MetricTooltip from '../components/MetricTooltip'

// Distinct hues for the Portfolio Growth chart's "By Holding" mode — one
// line per ticker, differentiated by hue rather than dash pattern (dashing
// more than 2-3 lines gets unreadable). Deliberately skips
// --signal-positive/--signal-negative/--signal-caution as ticker colors:
// those carry a "good/bad/warning" meaning everywhere else in the app, and
// a ticker landing on one arbitrarily would read as a value judgment it
// doesn't deserve. Cycles via modulo past the 5-ticker cap just as a safety
// net, not because more than 5 is expected.
const TICKER_COLORS = [
  'var(--chart-blue)',
  'var(--chart-teal)',
  'var(--chart-highlight)',
  'var(--signal-caution)',
  'var(--signal-negative)',
]
const colorForTicker = (i) => TICKER_COLORS[i % TICKER_COLORS.length]

const fmt  = v => `${(v * 100).toFixed(1)}%`
const fmtD = v => `$${v.toLocaleString('en-US', { maximumFractionDigits: 0 })}`
const fmtDec = v => `$${Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: 0 })}`

const CHART_STYLE = {
  background: 'transparent',
  fontSize: 11,
  fontFamily: 'var(--font-mono)',
}

const AXIS_STYLE = { fill: 'var(--text-muted)', fontSize: 10 }

function CustomTooltip({ active, payload, label, prefix = '', pct = false }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--surface-elevated)', border: 'var(--border-emphasis)',
      padding: '8px 12px', fontSize: 'var(--text-body-sm)', fontFamily: 'var(--font-primary)',
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

export default function Dashboard({ data, tickers, weights, portfolioValue, onTickerClick, sectorData, sectorLoading }) {
  const [growthMode, setGrowthMode] = useState('combined')

  if (!data) return null

  const { annualised_return: ret, annualised_volatility: vol,
          sharpe_ratio: sharpe, sortino_ratio: sortino,
          max_drawdown: dd, var_cvar, beta_alpha: ba,
          cumulative_returns: cum, benchmark_cumulative: bench,
          per_ticker_cumulative_returns: perTicker,
          drawdown_series: ddSeries, period,
          diversification_score: divScore } = data

  const medianFinal = data.monte_carlo?.p50_final
  const profitProb  = data.monte_carlo?.prob_profit

  const growthData = cum.dates.map((d, i) => {
    const row = { date: d.slice(5), portfolio: cum.values[i] }
    if (bench?.values[i] != null) row.benchmark = bench.values[i]
    return row
  })

  // "By Holding" mode — one cumulative-return series per ticker, from the
  // new per_ticker_cumulative_returns field. Falls back gracefully (toggle
  // just won't render) if a response is ever missing it.
  const canShowByHolding = !!perTicker && tickers.every(tk => perTicker[tk])
  const holdingData = canShowByHolding
    ? perTicker[tickers[0]].dates.map((d, i) => {
        const row = { date: d.slice(5) }
        tickers.forEach(tk => { row[tk] = perTicker[tk].values[i] })
        if (bench?.values[i] != null) row.benchmark = bench.values[i]
        return row
      })
    : []

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
    { label: 'VaR 95%',  metricKey: 'var_95',  value: `-${fmtDec(var_cvar.var_dollar)}`,  tone: 'bad',    small: true },
    { label: 'CVaR 95%', metricKey: 'cvar_95', value: `-${fmtDec(var_cvar.cvar_dollar)}`, tone: 'bad',    small: true },
    divScore ? { label: 'Diversification', metricKey: 'diversification_score', value: `${divScore.score}/100`, tone: divTone, small: true, sub: `${divScore.label} · avg corr ${divScore.avg_pairwise_corr.toFixed(2)} · ${period.n_years}yr window` } : null,
    ba ? { label: 'Beta',  metricKey: 'beta',  value: ba.beta.toFixed(2),  tone: ba.beta > 1.5 ? 'warning' : 'neutral', small: true } : null,
    ba ? { label: 'Alpha', metricKey: 'alpha', value: fmt(ba.alpha),        tone: ba.alpha > 0 ? 'good' : 'bad',         small: true } : null,
  ].filter(Boolean)

  const secondCols = `repeat(${secondRowItems.length}, 1fr)`

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, height: '100%' }}>

      <FirstResultCallout />

      {/* Period pills */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        {[period.start, period.end, `${period.n_days} days`, `${period.n_years}yr`].map((pill, i) => (
          <div key={i} style={{
            fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', letterSpacing: 'var(--tracking-caption)',
            background: 'var(--surface-elevated)', border: 'var(--border-default)',
            padding: '3px 8px', color: 'var(--text-muted)',
            fontFamily: 'var(--font-mono)',
          }}>
            {pill}
          </div>
        ))}
      </div>

      {/* Top metrics row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 10 }}>
        <MetricCard label={<MetricTooltip metricKey="annual_return">Annual Return</MetricTooltip>}  value={fmt(ret)}           tone={retColor}   sub={ba ? `${fmt(ba.alpha)} alpha` : null} />
        <MetricCard label={<MetricTooltip metricKey="volatility">Volatility</MetricTooltip>}     value={fmt(vol)}           tone={vol < 0.2 ? 'good' : vol < 0.35 ? 'warning' : 'bad'} />
        <div data-tour="sharpe-card">
          <MetricCard label={<MetricTooltip metricKey="sharpe_ratio">Sharpe Ratio</MetricTooltip>} value={sharpe.toFixed(2)} tone={sharpeColor} sub="above 1.0 is good" />
        </div>
        <MetricCard label={<MetricTooltip metricKey="sortino_ratio">Sortino Ratio</MetricTooltip>}  value={sortino.toFixed(2)} tone={sharpeColor} />
        <MetricCard label={<MetricTooltip metricKey="max_drawdown">Max Drawdown</MetricTooltip>}   value={fmt(dd)}            tone={ddColor} />
      </div>

      {/* Second metrics row — dynamic */}
      <div style={{ display: 'grid', gridTemplateColumns: secondCols, gap: 10 }}>
        {secondRowItems.map(item => (
          <MetricCard
            key={item.label}
            label={item.metricKey
              ? <MetricTooltip metricKey={item.metricKey}>{item.label}</MetricTooltip>
              : item.label}
            value={item.value}
            tone={item.tone}
            small={item.small}
            sub={item.sub}
          />
        ))}
      </div>

      {/* Sector exposure */}
      <SectorChart sectorData={sectorData} loading={sectorLoading} />

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 220px', gap: 10, flex: 1, minHeight: 0 }}>

        {/* Growth chart */}
        <div className="card" style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8, flexWrap: 'wrap', gap: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)' }}>
                Portfolio growth
              </div>
              {canShowByHolding && (
                <div style={{ display: 'flex', gap: 6 }}>
                  {['combined', 'byHolding'].map(m => (
                    <button key={m} onClick={() => setGrowthMode(m)} style={{
                      padding: '3px 8px', fontSize: 10,
                      fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase',
                      background: growthMode === m ? 'var(--signal-positive)' : 'var(--surface-elevated)',
                      color: growthMode === m ? 'var(--surface-canvas)' : 'var(--text-muted)',
                      border: 'var(--border-default)', transition: 'all 0.15s',
                    }}>
                      {m === 'combined' ? 'Combined' : 'By Holding'}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <div style={{ display: 'flex', gap: 10, fontSize: 10, flexWrap: 'wrap' }}>
              {growthMode === 'combined' ? (
                <span style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-muted)' }}>
                  <span style={{ width: 20, height: 2, background: 'var(--signal-positive)', display: 'inline-block' }} />
                  Your portfolio
                </span>
              ) : (
                tickers.map((tk, i) => (
                  <span key={tk} style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-muted)' }}>
                    <span style={{ width: 20, height: 2, background: colorForTicker(i), display: 'inline-block' }} />
                    {tk}
                  </span>
                ))
              )}
              {bench && (
                <span style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-muted)' }}>
                  <span style={{ width: 20, height: 2, display: 'inline-block', opacity: 0.6, borderBottom: '2px dashed var(--text-muted)' }} />
                  S&P 500
                </span>
              )}
            </div>
          </div>
          <div style={{ flex: 1, minHeight: 0 }}>
            {sectorLoading ? (
              <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                <svg className="spin" width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="var(--signal-positive)" strokeWidth="3" strokeDasharray="40 20" />
                </svg>
                <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Loading chart…</div>
              </div>
            ) : growthMode === 'combined' ? (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={growthData} style={CHART_STYLE} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                  <XAxis dataKey="date" tick={AXIS_STYLE} tickLine={false} axisLine={false} interval="preserveStartEnd" />
                  <YAxis tickFormatter={v => `${(v*100).toFixed(0)}%`} tick={AXIS_STYLE} tickLine={false} axisLine={false} width={42} />
                  <ReferenceLine y={0} stroke="rgba(var(--text-primary-rgb),0.1)" strokeDasharray="4 4" />
                  <Tooltip content={<CustomTooltip pct />} />
                  <Area type="monotone" dataKey="portfolio" stroke="var(--signal-positive)" strokeWidth={1.5} fill="var(--signal-positive)" fillOpacity={0.08} dot={false} name="Portfolio" />
                  {bench && <Line type="monotone" dataKey="benchmark" stroke="var(--text-muted)" strokeWidth={1.5} strokeDasharray="5 4" dot={false} name="S&P 500" />}
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={holdingData} style={CHART_STYLE} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                  <XAxis dataKey="date" tick={AXIS_STYLE} tickLine={false} axisLine={false} interval="preserveStartEnd" />
                  <YAxis tickFormatter={v => `${(v*100).toFixed(0)}%`} tick={AXIS_STYLE} tickLine={false} axisLine={false} width={42} />
                  <ReferenceLine y={0} stroke="rgba(var(--text-primary-rgb),0.1)" strokeDasharray="4 4" />
                  <Tooltip content={<CustomTooltip pct />} />
                  {tickers.map((tk, i) => (
                    <Line key={tk} type="monotone" dataKey={tk} stroke={colorForTicker(i)} strokeWidth={1.5} dot={false} name={tk} />
                  ))}
                  {bench && <Line type="monotone" dataKey="benchmark" stroke="var(--text-muted)" strokeWidth={1.5} strokeDasharray="5 4" dot={false} name="S&P 500" />}
                </ComposedChart>
              </ResponsiveContainer>
            )}
          </div>

          <InsightBox
            label="Return"
            tone={retColor}
            compact
            text={`Portfolio grew at <strong>${fmt(ret)}/yr</strong>. ${ba ? `Beat the S&P 500 by <strong>${fmt(ret - ba.benchmark_return)}</strong>.` : ''}${
              // Median 1-year projection comes from Monte Carlo, the heavy
              // tier — it isn't in the fast summary this tab otherwise
              // renders from, so say so plainly instead of a misleading
              // "$0 (0% chance of profit)" for the moment before it arrives.
              medianFinal != null
                ? ` Median 1-year projection: <strong>${fmtD(medianFinal)}</strong> (${Math.round((profitProb || 0) * 100)}% chance of profit).`
                : ' Median 1-year projection: running the full simulation...'
            }`}
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
            <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)', marginBottom: 10 }}>
              Drawdown
            </div>
            <div style={{ flex: 1, minHeight: 0 }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={ddData} style={CHART_STYLE} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                  <XAxis dataKey="date" tick={{ ...AXIS_STYLE, fontSize: 9 }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
                  <YAxis tickFormatter={v => `${(v*100).toFixed(0)}%`} tick={{ ...AXIS_STYLE, fontSize: 9 }} tickLine={false} axisLine={false} width={36} />
                  <Tooltip content={<CustomTooltip pct />} />
                  <Area type="monotone" dataKey="drawdown" stroke="var(--signal-negative)" strokeWidth={1.5} fill="var(--signal-negative)" fillOpacity={0.08} dot={false} name="Drawdown" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div style={{ fontSize: 11, marginTop: 8 }}>
              <span style={{ color: 'var(--text-muted)' }}>Worst drop: </span>
              <span style={{ color: 'var(--signal-negative)', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>{fmt(dd)}</span>
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}