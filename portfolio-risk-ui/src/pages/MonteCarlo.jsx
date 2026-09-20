import { useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine, Customized,
} from 'recharts'
import MetricCard from '../components/MetricCard'
import InsightBox from '../components/InsightBox'
import HeavyTierPending from '../components/HeavyTierPending'
import MetricTooltip from '../components/MetricTooltip'
import useOverlay from '../hooks/useOverlay'
import useCompactViewport from '../hooks/useCompactViewport'

const fmtD = v => `$${v.toLocaleString('en-US', { maximumFractionDigits: 0 })}`

const SCENARIOS = [
  { key: 'bear', label: 'Bear', color: 'var(--signal-negative)', bg: 'rgba(var(--signal-negative-rgb),0.13)', sub: '−10%/yr headwind' },
  { key: 'base', label: 'Base', color: 'var(--signal-positive)', bg: 'rgba(var(--signal-positive-rgb),0.13)', sub: 'historical drift' },
  { key: 'bull', label: 'Bull', color: 'var(--accent-light)', bg: 'rgba(var(--accent-light-rgb),0.13)', sub: '+10%/yr tailwind' },
]

function ScenarioToggle({ scenario, onChange }) {
  return (
    <div className="card" style={{ display: 'flex', gap: 8, padding: 8, flexShrink: 0 }}>
      {SCENARIOS.map(s => {
        const active = scenario === s.key
        return (
          <button
            key={s.key}
            onClick={() => onChange(s.key)}
            style={{
              flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2,
              padding: '8px 0',
              border: `1px solid ${active ? s.color : 'var(--line-hairline)'}`,
              background: active ? s.bg : 'rgba(var(--text-primary-rgb),0.03)',
              cursor: 'pointer', transition: 'all 0.15s',
            }}
          >
            <span style={{
              fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)',
              fontFamily: 'var(--font-primary)', color: active ? s.color : 'var(--text-muted)',
            }}>
              {s.label}
            </span>
            <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>{s.sub}</span>
          </button>
        )
      })}
    </div>
  )
}

// Canvas layer that draws the 1,000 faint grey paths
// Custom Recharts layer — renders sampled paths using the chart's own coordinate system
function SimulatedPaths({ allPaths, xAxisMap, yAxisMap }) {
  if (!allPaths || !xAxisMap || !yAxisMap) return null

  const xAxis = Object.values(xAxisMap)[0]
  const yAxis = Object.values(yAxisMap)[0]
  if (!xAxis || !yAxis) return null

  const { scale: xScale } = xAxis
  const { scale: yScale } = yAxis
  if (!xScale || !yScale) return null

  const nDays = allPaths[0]?.length - 1 || 252
  const step  = Math.max(1, Math.floor(allPaths.length / 150))

  const paths = allPaths
    .filter((_, i) => i % step === 0)
    .map((path, pi) => {
      const d = path.map((v, i) => {
        const x = xScale(i)
        const y = yScale(v)
        return `${i === 0 ? 'M' : 'L'}${x},${y}`
      }).join(' ')
      return <path key={pi} d={d} stroke="rgba(var(--chart-sage-rgb),0.09)" strokeWidth="0.7" fill="none" />
    })

  return <g>{paths}</g>
}

export default function MonteCarlo({ data, heavyError }) {
  const [scenario, setScenario] = useState('base')
  // Same dual-axis signal, same hooks-before-early-return placement, as
  // Backtest.jsx's and RiskAnalysis.jsx's identical reasoning: `data` (and
  // `data.monte_carlo`, gated just below) genuinely toggles within one
  // mounted instance of this component, so the hooks that drive the
  // fullscreen expand control have to run every render regardless of which
  // return path fires.
  const isCompactViewport = useCompactViewport()
  const [expandRequested, setExpandRequested] = useState(false)
  const isChartExpanded = expandRequested && isCompactViewport
  const chartCardRef = useRef(null)
  useOverlay(isChartExpanded, chartCardRef, () => setExpandRequested(false))

  if (!data) return null

  // Monte Carlo lives in the heavy tier — the summary response Dashboard
  // renders from doesn't include it yet, /api/analyse-full fills it in in
  // the background right after. Show a scoped state here instead of
  // crashing on the missing fields; this re-renders on its own the moment
  // useAnalysis's background call resolves and replaces `data`.
  if (!data.monte_carlo) {
    return <HeavyTierPending label="Loading Monte Carlo simulation..." error={heavyError} />
  }

  const mc = data[`monte_carlo_${scenario}`] || data.monte_carlo
  const {
    percentile_5: p5, percentile_50: p50, percentile_95: p95,
    p5_final, p50_final, p95_final,
    prob_profit, prob_loss_10pct, portfolio_value, n_simulations,
    all_paths,
  } = mc

  const chartData = p50.map((val, i) => ({
    day: i, p5: p5[i], p50: val, p95: p95[i],
  }))

  const gain    = p50_final - portfolio_value
  const gainPct = ((p50_final / portfolio_value) - 1) * 100
  const tone    = prob_profit > 0.75 ? 'good' : prob_profit > 0.5 ? 'warning' : 'bad'

  // Built once here, referenced twice below (normal flex slot vs.
  // portalled fullscreen) — same reasoning as Backtest.jsx's chartCardEl
  // (that const's own comment): App.jsx's .fade-up mount-animation wrapper
  // is a containing block for position:fixed descendants, so a portal to
  // document.body is required to actually cover the viewport, not a style
  // choice.
  const chartCardEl = (
    <div
      ref={chartCardRef}
      className={`montecarlo-chart-card card${isChartExpanded ? ' chart-fullscreen-expanded' : ''}`}
      style={{ padding: '16px', flex: 1, display: 'flex', flexDirection: 'column' }}
    >

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexShrink: 0, flexWrap: 'wrap', gap: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)' }}>
            {n_simulations.toLocaleString()} simulated futures - next 12 months
          </div>
          {/* Rendered only on a compact viewport — desktop and tablet never
              see it, so isChartExpanded can never become true there
              regardless of expandRequested. Same control, reused from
              Dashboard/Backtest (.chart-expand-btn, index.css), not
              reimplemented. */}
          {isCompactViewport && (
            <button
              onClick={() => setExpandRequested(v => !v)}
              className="chart-expand-btn"
              aria-label={isChartExpanded ? 'Collapse chart' : 'Expand chart to full screen'}
              title={isChartExpanded ? 'Collapse' : 'Expand'}
            >
              {isChartExpanded ? '✕' : '⤢'}
            </button>
          )}
        </div>
        <div style={{ display: 'flex', gap: 14, fontSize: 10, flexWrap: 'wrap' }}>
          {[
            { color: 'var(--signal-negative)', label: `Bad (5th) - ${fmtD(p5_final)}`, dash: true },
            { color: 'var(--signal-positive)', label: `Median - ${fmtD(p50_final)}`, dash: false },
            { color: 'var(--accent-light)', label: `Good (95th) - ${fmtD(p95_final)}`, dash: true },
          ].map(l => (
            <span key={l.label} style={{ display: 'flex', alignItems: 'center', gap: 5, color: 'var(--text-muted)' }}>
              <svg width="20" height="2" style={{ flexShrink: 0 }}>
                <line x1="0" y1="1" x2="20" y2="1" stroke={l.color} strokeWidth="2" strokeDasharray={l.dash ? '4 3' : 'none'} />
              </svg>
              {l.label}
            </span>
          ))}
          <span style={{ display: 'flex', alignItems: 'center', gap: 5, color: 'var(--text-muted)' }}>
            <div style={{ width: 16, height: 1.5, background: 'rgba(var(--chart-sage-rgb),0.2)' }} />
            Simulated paths
          </span>
        </div>
      </div>

      {/* Chart */}
      <div style={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 8, right: 8, bottom: 6, left: 8 }}>
            <XAxis
              dataKey="day"
              height={42}
              tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
              tickLine={false} axisLine={false}
              label={{ value: 'Trading days (252 = 1 year)', position: 'insideBottom', offset: 4, fill: 'var(--text-muted)', fontSize: 10 }}
            />
            <YAxis
              tickFormatter={v => `$${(v / 1000).toFixed(0)}k`}
              tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
              tickLine={false} axisLine={false}
              width={48}
            />
            <Tooltip
              formatter={(v, n) => [fmtD(v), n === 'p5' ? 'Bad scenario' : n === 'p50' ? 'Median' : 'Good scenario']}
              labelFormatter={l => `Day ${l}`}
              contentStyle={{ background: 'var(--surface-elevated)', border: 'var(--border-emphasis)', fontSize: 11 }}
            />
            <ReferenceLine y={portfolio_value} stroke="rgba(var(--text-primary-rgb),0.1)" strokeDasharray="4 4"
              label={{ value: 'Start', fill: 'var(--text-muted)', fontSize: 9, position: 'right' }} />
            <Line type="monotone" dataKey="p5" stroke="var(--signal-negative)" strokeWidth={1.5} dot={false} strokeDasharray="5 4" />
            <Line type="monotone" dataKey="p50" stroke="var(--signal-positive)" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="p95" stroke="var(--accent-light)" strokeWidth={1.5} dot={false} strokeDasharray="5 4" />
            {all_paths && (
              <Customized component={<SimulatedPaths allPaths={all_paths} />} />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Secondary: restates numbers already shown in the metric cards
          above, unlike "Why three scenarios" which explains a control the
          reader hasn't used yet. */}
      <InsightBox
        label="Monte Carlo insight"
        tone={tone}
        compact
        priority="secondary"
        text={`Based on ${n_simulations.toLocaleString()} simulations, <strong>${Math.round(prob_profit * 100)}%</strong> of futures end the year profitable. The median outcome is <strong>${fmtD(p50_final)}</strong> - a ${gainPct > 0 ? 'gain' : 'loss'} of <strong>${fmtD(Math.abs(gain))}</strong>. Good year: <strong>${fmtD(p95_final)}</strong>. Bad year: <strong>${fmtD(p5_final)}</strong>. ${prob_loss_10pct > 0.2 ? `Note: <strong>${Math.round(prob_loss_10pct * 100)}% chance of losing 10%+</strong> - ensure you can hold through that.` : 'Past performance does not guarantee future results.'}`}
      />
    </div>
  )

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:12, height:'100%', overflowY:'auto' }}>

      {/* Why three scenarios — primary: explains the Bear/Base/Bull control
          directly below it, so it's the one that should default open. */}
      <InsightBox
        label="Why three scenarios, not one"
        compact
        priority="primary"
        text="Most tools show you one future. This shows you three, the realistic downside, the expected path, and the upside if things go right. See how your exact portfolio holds up before the market decides for you."
      />

      {/* Scenario toggle */}
      <ScenarioToggle scenario={scenario} onChange={setScenario} />

      {/* Metrics row — flex-wrap, not grid repeat(4,1fr): see
          .montecarlo-metrics-row's own comment in index.css. */}
      <div className="montecarlo-metrics-row" style={{ display:'flex', flexWrap:'wrap', gap:10, flexShrink:0 }}>
        <MetricCard small label={<MetricTooltip metricKey="starting_value">Starting value</MetricTooltip>}      value={fmtD(portfolio_value)} tone="neutral" />
        <MetricCard small label={<MetricTooltip metricKey="median_outcome">Median outcome</MetricTooltip>}      value={fmtD(p50_final)} sub={`${gainPct > 0 ? '+' : ''}${gainPct.toFixed(1)}%`} tone={gain > 0 ? 'good' : 'bad'} />
        <MetricCard small label={<MetricTooltip metricKey="chance_of_profit">Chance of profit</MetricTooltip>}    value={`${Math.round(prob_profit * 100)}%`} tone={prob_profit > 0.6 ? 'good' : 'warning'} sub={`${n_simulations.toLocaleString()} simulations`} />
        <MetricCard small label={<MetricTooltip metricKey="chance_of_loss">Chance of 10% Loss</MetricTooltip>} value={`${Math.round(prob_loss_10pct * 100)}%`} tone={prob_loss_10pct < 0.1 ? 'good' : prob_loss_10pct < 0.25 ? 'warning' : 'bad'} />
      </div>

      {/* Chart — built once above (chartCardEl), rendered either in its
          normal flex slot or through a portal when expanded. See
          Backtest.jsx's identical render call for why a portal is required,
          not a style choice. */}
      {isChartExpanded ? createPortal(chartCardEl, document.body) : chartCardEl}
    </div>
  )
}