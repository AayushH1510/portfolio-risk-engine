import { useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import HeavyTierPending from '../components/HeavyTierPending'
import InsightBox from '../components/InsightBox'
import useOverlay from '../hooks/useOverlay'
import useCompactViewport from '../hooks/useCompactViewport'
import { getBenchmarkLabel, getBenchmarkPhrase } from '../lib/benchmarks'

const fmtPct = v => `${(v * 100).toFixed(1)}%`

const CHART_STYLE = { background: 'transparent', fontSize: 11, fontFamily: 'var(--font-mono)' }
const AXIS_STYLE  = { fill: 'var(--text-muted)', fontSize: 10 }

const getStrategies = benchmarkLabel => [
  { key: 'your_portfolio', label: 'Your Portfolio', color: 'var(--signal-positive)', dash: false },
  { key: 'equal_weight',   label: 'Equal Weight',   color: 'var(--signal-caution)', dash: true  },
  { key: 'sp500',          label: benchmarkLabel,   color: 'var(--text-muted)', dash: true  },
]

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--surface-elevated)', border: 'var(--border-default)',
      padding: '8px 12px', fontSize: 11,
    }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, fontWeight: 600 }}>
          {p.name}: {fmtPct(p.value)}
        </div>
      ))}
    </div>
  )
}

function Row({ label, value, tone }) {
  const color = { good: 'var(--signal-positive)', warning: 'var(--signal-caution)', bad: 'var(--signal-negative)' }[tone] || 'var(--text-primary)'
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ fontSize: 14, fontWeight: 700, fontFamily: 'var(--font-mono)', color }}>{value}</span>
    </div>
  )
}

function StrategyCard({ label, color, sub, stats, borderTone }) {
  const { annualised_return, sharpe_ratio, max_drawdown } = stats
  const borderColor = borderTone === 'good' ? 'var(--signal-positive)' : borderTone === 'bad' ? 'var(--signal-negative)' : color

  return (
    <div className="card" style={{ padding: '14px 16px', borderTop: `2px solid ${borderColor}`, display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div>
        <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)' }}>
          {label}
        </div>
        {sub && (
          <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: 2 }}>{sub}</div>
        )}
      </div>
      <Row label="Annualised return" value={fmtPct(annualised_return)} tone={annualised_return >= 0 ? 'good' : 'bad'} />
      <Row label="Sharpe ratio"      value={sharpe_ratio.toFixed(2)}   tone={sharpe_ratio > 1 ? 'good' : sharpe_ratio > 0 ? 'warning' : 'bad'} />
      <Row label="Max drawdown"      value={fmtPct(max_drawdown)}      tone={Math.abs(max_drawdown) < 0.2 ? 'good' : Math.abs(max_drawdown) < 0.35 ? 'warning' : 'bad'} />
    </div>
  )
}

function ReturnsTable({ backtest, strategies }) {
  const years = Object.keys(backtest.your_portfolio.annual_returns).sort()

  return (
    <div className="card" style={{ padding: '14px 16px' }}>
      <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)', marginBottom: 12 }}>
        Year-by-year returns
      </div>
      {/* overflowX:'auto' already made this reachable at any width — the
          complaint was distribution, not overflow: with no explicit column
          widths, the browser's default auto table layout sizes each column
          to its own content's max-content width, so "Year" (4 characters)
          gets a narrow column while the three longer strategy-label/percent
          columns bunch into the remaining space instead of sharing it. Below
          768px (.backtest-returns-table, index.css) table-layout:fixed
          divides the available width evenly across all four columns
          instead, regardless of what any one column's content needs — a
          phone-width fix, not a blanket relayout: at desktop width the
          table's content already has enough room that auto-layout's
          content-based columns and an even four-way split land close
          enough that forcing the change everywhere isn't worth trading away
          "desktop stays byte-for-byte as before," so table-layout stays
          the plain CSS default there (base rule, index.css). table-layout
          lives in a class, not inline — same "inline always wins over a
          stylesheet rule" reason as every other viewport-conditional style
          this pass touched. */}
      <div style={{ overflowX: 'auto' }}>
        <table className="backtest-returns-table" style={{ width: '100%', borderCollapse: 'separate', borderSpacing: 0, fontFamily: 'var(--font-mono)' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', fontSize: 10, fontWeight: 700, color: 'var(--text-muted)', padding: '4px 10px 8px 4px', borderBottom: 'var(--border-default)' }}>
                Year
              </th>
              {strategies.map(s => (
                <th key={s.key} style={{ textAlign: 'right', fontSize: 10, fontWeight: 700, color: s.color, padding: '4px 10px 8px', borderBottom: 'var(--border-default)' }}>
                  {s.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {years.map(y => (
              <tr key={y}>
                <td style={{ fontSize: 12, color: 'var(--text-secondary)', padding: '7px 10px 7px 4px', borderBottom: '1px solid rgba(var(--text-primary-rgb),0.04)' }}>
                  {y}
                </td>
                {strategies.map(s => {
                  const v = backtest[s.key].annual_returns[y]
                  return (
                    <td key={s.key} style={{
                      textAlign: 'right', fontSize: 12, fontWeight: 600, padding: '7px 10px',
                      borderBottom: '1px solid rgba(var(--text-primary-rgb),0.04)',
                      color: v >= 0 ? 'var(--signal-positive)' : 'var(--signal-negative)',
                    }}>
                      {fmtPct(v)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function Backtest({ data, tickers, heavyError }) {
  // Same dual-axis "phone, either orientation" signal Dashboard's growth-
  // chart expand and RiskAnalysis's confidence-header fold both use.
  // Hooks placed before either early return below (both `data` and
  // `backtest` genuinely toggle null -> non-null within one mounted
  // instance of this component, before/after Run Analysis and while the
  // heavy tier is still loading) so they always run the same number of
  // times regardless of which return path fires.
  const isCompactViewport = useCompactViewport()
  const [expandRequested, setExpandRequested] = useState(false)
  const isChartExpanded = expandRequested && isCompactViewport
  const chartCardRef = useRef(null)
  useOverlay(isChartExpanded, chartCardRef, () => setExpandRequested(false))

  if (!data) return null

  const backtest = data.backtest
  // backtest is null two different ways: benchmark comparison is off (a
  // real, permanent state — benchmark_cumulative is absent too in that
  // case), or it just hasn't arrived from the background /api/analyse-full
  // call yet (benchmark_cumulative IS present already, since that's fast-
  // tier — only backtest itself is heavy). Tell those apart instead of
  // showing the "enable benchmark" message while it's simply still loading.
  if (!backtest) {
    if (!data.benchmark_cumulative) {
      return (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)', fontSize: 13 }}>
          Select a benchmark in the sidebar and rerun the analysis to see the backtest.
        </div>
      )
    }
    return <HeavyTierPending label="Loading backtest results..." error={heavyError} />
  }

  const strategies = getStrategies(getBenchmarkLabel(data.benchmark))

  const sharpes = {
    your_portfolio: backtest.your_portfolio.sharpe_ratio,
    equal_weight:   backtest.equal_weight.sharpe_ratio,
    sp500:          backtest.sp500.sharpe_ratio,
  }
  const bestKey  = Object.entries(sharpes).reduce((a, b) => (b[1] > a[1] ? b : a))[0]
  const worstKey = Object.entries(sharpes).reduce((a, b) => (b[1] < a[1] ? b : a))[0]
  const yourBorderTone = bestKey === 'your_portfolio' ? 'good' : worstKey === 'your_portfolio' ? 'bad' : null

  const dates = backtest.your_portfolio.cumulative_returns.dates
  const chartData = dates.map((d, i) => ({
    date: d,
    your_portfolio: backtest.your_portfolio.cumulative_returns.values[i],
    equal_weight:   backtest.equal_weight.cumulative_returns.values[i],
    sp500:          backtest.sp500.cumulative_returns.values[i],
  }))

  // Built once here, referenced twice below (normal flex slot vs.
  // portalled fullscreen) — see the render call's own comment for why a
  // portal is required, not just a CSS class swap in place.
  const chartCardEl = (
    <div
      ref={chartCardRef}
      // min-height lives in CSS (.backtest-chart-card, index.css), not
      // inline — an inline value always wins over a stylesheet rule
      // regardless of media-query match, so a compact-viewport min-height
      // override couldn't reach an inline minHeight:0 here. Dashboard.jsx's
      // growth chart hit this exact bug first (see .dashboard-grid__growth's
      // own comment): min-height:260px inline-shadowed by minHeight:0 was
      // silently inert. Confirmed here too, not just inferred from that
      // precedent — the min-height:280px rule below measured 0 effect until
      // this moved out.
      className={`backtest-chart-card card${isChartExpanded ? ' chart-fullscreen-expanded' : ''}`}
      style={{ padding: '14px 16px', flex: 1, display: 'flex', flexDirection: 'column' }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexShrink: 0, flexWrap: 'wrap', gap: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)' }}>
            Cumulative return - {backtest.period.start} to {backtest.period.end}
          </div>
          {/* Rendered only on a compact viewport — desktop and tablet never
              see it, so isChartExpanded can never become true there
              regardless of expandRequested. Same control, same 44x44
              ::before tap target, as Dashboard's growth chart
              (.chart-expand-btn, index.css) — reused, not reimplemented. */}
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
        {/* flexWrap — same as Dashboard's identical growth-chart legend row
            (Dashboard.jsx); without it this was the one remaining clipped
            element found in the full re-sweep (RESPONSIVE_AUDIT.md), a
            small (~37px) contained-vertical-only overflow at 320px from the
            three-item legend not fitting one line and having no wrap
            behaviour to fall back on. */}
        <div style={{ display: 'flex', gap: 14, fontSize: 10, flexWrap: 'wrap' }}>
          {strategies.map(s => (
            <span key={s.key} style={{ display: 'flex', alignItems: 'center', gap: 5, color: 'var(--text-muted)' }}>
              <svg width="20" height="2" style={{ flexShrink: 0 }}>
                <line x1="0" y1="1" x2="20" y2="1" stroke={s.color} strokeWidth="2" strokeDasharray={s.dash ? '4 3' : 'none'} />
              </svg>
              {s.label}
            </span>
          ))}
        </div>
      </div>
      {/* overflow:'hidden' — same Recharts tooltip-position containment
          backstop as Dashboard/Comparison/RiskAnalysis's chart wrappers
          (see RESPONSIVE_AUDIT.md); this tab's chart didn't have it yet. */}
      <div style={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} style={CHART_STYLE} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <XAxis dataKey="date" tick={AXIS_STYLE} tickLine={false} axisLine={false} interval="preserveStartEnd" />
            <YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={AXIS_STYLE} tickLine={false} axisLine={false} width={48} />
            <ReferenceLine y={0} stroke="rgba(var(--text-primary-rgb),0.1)" strokeDasharray="4 4" />
            <Tooltip content={<CustomTooltip />} />
            {strategies.map(s => (
              <Line
                key={s.key}
                type="monotone"
                dataKey={s.key}
                name={s.label}
                stroke={s.color}
                strokeWidth={s.key === 'your_portfolio' ? 2 : 1.5}
                strokeDasharray={s.dash ? '5 4' : undefined}
                dot={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )

  return (
    <div className="tab-scroll-root" style={{ display: 'flex', flexDirection: 'column', gap: 12, height: '100%', overflowY: 'auto' }}>

      {/* Why this tab matters — same compact InsightBox pattern as the other tabs.
          Benchmark is named dynamically (getBenchmarkPhrase) so the copy stays
          accurate for whichever benchmark is selected, never hardcodes "S&P 500". */}
      <InsightBox
        label="Why run the backtest"
        compact
        priority="primary"
        text={`Stop guessing whether your weights make sense. This runs your exact allocation through the real historical period, side by side with a plain equal split and ${getBenchmarkPhrase(data.benchmark)}, same buy-and-hold, same dates for all three. See which one would actually have come out ahead.`}
      />

      {/* Summary cards — flex-wrap, not grid repeat(3,1fr): see
          .backtest-strategy-row's own comment in index.css. At a width
          where all three don't fit one line, grid's auto-fit/repeat shares
          one column count across every wrapped row, so a 3-into-2 wrap
          strands the last card at one-third width on its own line instead
          of filling it — the identical failure mode Dashboard's metric-card
          rows already fixed the same way (RESPONSIVE_AUDIT.md). flex sizes
          each wrapped line independently, so the lone third card grows to
          fill its own row. */}
      <div className="backtest-strategy-row" style={{ display: 'flex', flexWrap: 'wrap', gap: 10, flexShrink: 0 }}>
        <StrategyCard
          label="Your Portfolio"
          sub={tickers?.join(', ')}
          color="var(--signal-positive)"
          stats={backtest.your_portfolio}
          borderTone={yourBorderTone}
        />
        <StrategyCard label="Equal Weight" color="var(--signal-caution)" stats={backtest.equal_weight} />
        <StrategyCard label={strategies[2].label} color="var(--text-muted)" stats={backtest.sp500} />
      </div>

      {/* Cumulative return chart — built once, rendered either in its normal
          flex slot or through a portal when expanded, same reasoning as
          Dashboard's growthCardEl (see that const's own comment,
          Dashboard.jsx): App.jsx's .fade-up mount-animation wrapper leaves a
          resting transform that makes it a containing block for any
          position:fixed descendant, so a plain position:fixed here would
          resolve against .fade-up's box instead of the viewport — portalling
          to document.body (which has no such ancestor) is required, not a
          style preference. */}
      {isChartExpanded ? createPortal(chartCardEl, document.body) : chartCardEl}

      {/* Year-by-year table */}
      <ReturnsTable backtest={backtest} strategies={strategies} />

    </div>
  )
}
