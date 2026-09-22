import { useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import InsightBox from '../components/InsightBox'
import useOverlay from '../hooks/useOverlay'
import useCompactViewport from '../hooks/useCompactViewport'
import { computeYDomain } from '../lib/chartDomain'

const fmt    = v => v != null ? `${(v * 100).toFixed(1)}%` : 'N/A'
const fmtD   = v => v != null ? `$${Math.abs(v).toFixed(0)}` : 'N/A'
const fmtN   = v => v != null ? v.toFixed(2) : 'N/A'
const TIP    = { background:'var(--surface-elevated)', border:'var(--border-emphasis)', fontSize:11 }

// Colour for A and B
const A_COLOR = 'var(--signal-positive)'
const B_COLOR = 'var(--signal-caution)'

function winner(aVal, bVal, lowerBetter = false) {
  if (aVal == null || bVal == null) return null
  return lowerBetter ? (aVal < bVal ? 'A' : 'B') : (aVal > bVal ? 'A' : 'B')
}

function MetricRow({ label, aVal, bVal, fmt: fmtFn = fmtN, lowerBetter = false, isCompactViewport = false }) {
  const w = winner(
    typeof aVal === 'number' ? aVal : null,
    typeof bVal === 'number' ? bVal : null,
    lowerBetter
  )
  // Compact viewport only — see .compare-ab-row's compact override
  // (index.css) for why: below 768px the centre label track moves out from
  // between A and B onto its own line, leaving the two value columns
  // directly adjacent with no track between them. The desktop layout (right-
  // aligned A / centred label / left-aligned B) relied on that centre track
  // itself as the visual separator between the two values — remove the
  // track without changing the alignment and the two numbers render flush
  // against the shared column boundary with nothing between them ("10.8%
  // 18.4%" with no gap, the reported bug). Centring each value in its own
  // half-width column pulls it away from that boundary instead. Desktop is
  // untouched — same right/left alignment and win-only colour it always
  // had — because these are inline styles (an inline declaration always
  // wins over a stylesheet media query, the same reason gridTemplateColumns
  // itself lives in the CSS class and not here), so the only way to keep
  // desktop byte-for-byte is to branch in JS on the same isCompactViewport
  // signal the rest of this file already uses, not to try to express it in
  // CSS.
  const align = isCompactViewport ? 'center' : null
  const justifyA = isCompactViewport ? 'center' : 'flex-end'
  const justifyB = isCompactViewport ? 'center' : 'flex-start'
  // Attribution color — at compact viewport, A/B always render in the
  // header's own green/orange (not just the winner) so which value belongs
  // to which portfolio never depends solely on left/right position, per the
  // request. Desktop keeps its original win-only colouring (a muted value
  // reads as "not the winner" there, which was never reported as a problem
  // and isn't part of what broke on the device).
  const colorA = isCompactViewport ? A_COLOR : (w === 'A' ? A_COLOR : 'var(--text-secondary)')
  const colorB = isCompactViewport ? B_COLOR : (w === 'B' ? B_COLOR : 'var(--text-secondary)')
  return (
    // .compare-ab-row — shared with the nameA/VS/nameB header below, see
    // that class's own comment in index.css for why: A and B must stay in
    // two side-by-side columns identified purely by position (this row has
    // no per-cell name to fall back on), so the fix moves the label out of
    // the centre track instead of stacking the row. gridTemplateColumns
    // lives in that CSS class, not inline, for the usual reason (an inline
    // declaration always wins over a stylesheet rule regardless of media-
    // query match) — only the per-cell gridArea assignment stays inline,
    // since the area names ('a'/'label'/'b') don't change between the two
    // breakpoints, only where the class's own template puts them.
    <div className="compare-ab-row" style={{
      alignItems: 'center', padding: '8px 0',
      borderBottom: '1px solid rgba(var(--text-primary-rgb),0.04)',
    }}>
      {/* A value */}
      <div style={{
        gridArea: 'a',
        textAlign: align || 'right',
        fontSize: 14, fontWeight: 700, fontFamily: 'var(--font-mono)',
        color: colorA,
        display: 'flex', alignItems: 'center', justifyContent: justifyA, gap: 6,
      }}>
        {w === 'A' && <span style={{ fontSize: 10, background: 'rgba(var(--signal-positive-rgb),0.15)', color: A_COLOR, padding: '1px 5px', fontFamily: 'var(--font-sans-generic)' }}>WIN</span>}
        {fmtFn(aVal)}
      </div>

      {/* Label */}
      <div style={{ gridArea: 'label', textAlign: 'center', fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        {label}
      </div>

      {/* B value */}
      <div style={{
        gridArea: 'b',
        textAlign: align || 'left',
        fontSize: 14, fontWeight: 700, fontFamily: 'var(--font-mono)',
        color: colorB,
        display: 'flex', alignItems: 'center', justifyContent: justifyB, gap: 6,
      }}>
        {fmtFn(bVal)}
        {w === 'B' && <span style={{ fontSize: 10, background: 'rgba(var(--signal-caution-rgb),0.15)', color: B_COLOR, padding: '1px 5px', fontFamily: 'var(--font-sans-generic)' }}>WIN</span>}
      </div>
    </div>
  )
}

export default function Comparison({ dataA, dataB, nameA, nameB, tickersA, tickersB }) {
  // Same dual-axis signal, same hooks-before-early-return placement, as
  // MonteCarlo.jsx's/Backtest.jsx's identical reasoning: dataA/dataB
  // genuinely toggle within one mounted instance of this component (a
  // Portfolio B re-run), so the hooks driving the fullscreen expand
  // controls have to run every render regardless of which return path
  // fires. Two independent expand states — Cumulative Returns and Monte
  // Carlo are two separate charts, the fifth and sixth consumers of this
  // mechanism (after Dashboard/Backtest/MonteCarlo/Frontier), each needing
  // its own toggle rather than sharing one.
  const isCompactViewport = useCompactViewport()

  const [growthExpandRequested, setGrowthExpandRequested] = useState(false)
  const isGrowthExpanded = growthExpandRequested && isCompactViewport
  const growthCardRef = useRef(null)
  useOverlay(isGrowthExpanded, growthCardRef, () => setGrowthExpandRequested(false))

  const [mcExpandRequested, setMcExpandRequested] = useState(false)
  const isMcExpanded = mcExpandRequested && isCompactViewport
  const mcCardRef = useRef(null)
  useOverlay(isMcExpanded, mcCardRef, () => setMcExpandRequested(false))

  if (!dataA || !dataB) return null

  // Build combined cumulative returns chart
  const maxLen  = Math.min(dataA.cumulative_returns.dates.length, dataB.cumulative_returns.dates.length)
  const growthData = dataA.cumulative_returns.dates.slice(0, maxLen).map((d, i) => ({
    date: d.slice(5),
    a:    dataA.cumulative_returns.values[i],
    b:    dataB.cumulative_returns.values[i],
  }))

  // Build combined Monte Carlo median lines
  const mcLen  = Math.min(dataA.monte_carlo.percentile_50.length, dataB.monte_carlo.percentile_50.length)
  const mcData = dataA.monte_carlo.percentile_50.slice(0, mcLen).map((v, i) => ({
    day: i,
    a:   v,
    b:   dataB.monte_carlo.percentile_50[i],
  }))
  // Same computeYDomain Dashboard.jsx's growth/by-holding charts use (now
  // shared via lib/chartDomain.js rather than duplicated) — Recharts' own
  // default domain (no domain prop at all) can leave real data (e.g.
  // $10k-$13k) crushed into a thin band in the middle of a short compact-
  // viewport chart. Gated on isCompactViewport, and — unlike Dashboard's
  // own charts — the domain prop is only spread onto <YAxis> at all when
  // compact (see below), not passed as a literal ['auto', 'auto'] the rest
  // of the time. Checked directly, not assumed from Dashboard's pattern:
  // Recharts' true default (the prop omitted entirely) is NOT the same as
  // explicitly passing domain={['auto', 'auto']} for this chart's data —
  // the omitted-prop path renders $0k-$14k-ish (confirmed via a live
  // getBoundingClientRect/tick-label check), the explicit-['auto','auto']
  // path renders a visibly tighter $9k-$14k even at 1440px desktop width,
  // a real, measured desktop pixel-diff (~1.6k-5.3k px depending on width)
  // that a same-code control run confirmed was NOT just Recharts' usual
  // sub-pixel SVG noise. Conditionally omitting the prop rather than
  // passing a literal 'auto' array is what actually keeps desktop
  // byte-for-byte — see the spread below.
  const mcYDomain = isCompactViewport ? computeYDomain(mcData, ['a', 'b']) : null

  // Score — count wins per portfolio
  const metrics = [
    { aV: dataA.annualised_return,          bV: dataB.annualised_return,          lb: false },
    { aV: dataA.annualised_volatility,       bV: dataB.annualised_volatility,       lb: true  },
    { aV: dataA.sharpe_ratio,                bV: dataB.sharpe_ratio,                lb: false },
    { aV: dataA.sortino_ratio,               bV: dataB.sortino_ratio,               lb: false },
    { aV: Math.abs(dataA.max_drawdown),      bV: Math.abs(dataB.max_drawdown),      lb: true  },
    { aV: Math.abs(dataA.var_cvar.var_pct),  bV: Math.abs(dataB.var_cvar.var_pct),  lb: true  },
  ]
  const winsA = metrics.filter(m => winner(m.aV, m.bV, m.lb) === 'A').length
  const winsB = metrics.filter(m => winner(m.aV, m.bV, m.lb) === 'B').length
  const overallWinner = winsA > winsB ? 'A' : winsB > winsA ? 'B' : null
  const overallColor  = overallWinner === 'A' ? A_COLOR : B_COLOR

  // Built once here, referenced twice below (normal grid slot vs. portalled
  // fullscreen) — same reasoning as MonteCarlo.jsx's/Backtest.jsx's own
  // chartCardEl: App.jsx's .fade-up mount-animation wrapper is a containing
  // block for position:fixed descendants, so a portal to document.body is
  // required to actually cover the viewport, not a style choice. Unlike
  // those two cards, this one and mcChartCardEl below don't need a
  // component-specific CSS class for a compact-viewport min-height floor —
  // neither is a flex:1 child of a viewport-height-constrained column the
  // way Dashboard's/Backtest's/MonteCarlo's chart cards are (Comparison's
  // whole page is a plain, freely-scrolling flex column), so there's
  // nothing here that gets starved to 0 the way those needed a floor to
  // prevent. The inner chart wrapper's own minHeight (200, matching the
  // pre-existing value) already acts as the real floor in both the normal
  // and expanded case — flex:1 lets it grow to fill 100dvh once expanded,
  // exactly like the other four consumers.
  const growthChartCardEl = (
    <div
      ref={growthCardRef}
      className={`card${isGrowthExpanded ? ' chart-fullscreen-expanded' : ''}`}
      style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column' }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10, flexShrink: 0, flexWrap: 'wrap', gap: 8 }}>
        <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)' }}>
          Cumulative returns
        </div>
        {/* Rendered only on a compact viewport — desktop/tablet never see
            it, so isGrowthExpanded can never become true there regardless
            of growthExpandRequested. Same control, reused from
            Dashboard/Backtest/MonteCarlo/Frontier (.chart-expand-btn,
            index.css), not reimplemented. */}
        {isCompactViewport && (
          <button
            onClick={() => setGrowthExpandRequested(v => !v)}
            className="chart-expand-btn"
            aria-label={isGrowthExpanded ? 'Collapse chart' : 'Expand chart to full screen'}
            title={isGrowthExpanded ? 'Collapse' : 'Expand'}
          >
            {isGrowthExpanded ? '✕' : '⤢'}
          </button>
        )}
      </div>
      {/* overflow:'hidden' — containment backstop for Recharts'
          tooltip-position clamp, which isn't an absolute safety net
          (it floors against ResponsiveContainer's last-*committed*
          measured width, not a live DOM read); same fix, same
          reasoning as Dashboard.jsx's chart wrappers — see
          RESPONSIVE_AUDIT.md's touch-tooltip-overflow postmortem. */}
      <div style={{ flex: 1, minHeight: 200, overflow: 'hidden' }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={growthData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
            <XAxis dataKey="date" tick={{ fill:'var(--text-muted)', fontSize:10 }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
            <YAxis tickFormatter={v => `${(v*100).toFixed(0)}%`} tick={{ fill:'var(--text-muted)', fontSize:10 }} tickLine={false} axisLine={false} width={42} />
            <ReferenceLine y={0} stroke="rgba(var(--text-primary-rgb),0.08)" strokeDasharray="4 4" />
            <Tooltip contentStyle={TIP} formatter={v => fmt(v)} />
            <Area type="monotone" dataKey="a" stroke={A_COLOR} strokeWidth={1.5} fill={A_COLOR} fillOpacity={0.08} dot={false} name={nameA} />
            <Area type="monotone" dataKey="b" stroke={B_COLOR} strokeWidth={1.5} fill={B_COLOR} fillOpacity={0.08} dot={false} name={nameB} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )

  // Same reasoning as growthChartCardEl above. The chart wrapper moved from
  // a literal height:180 to flex:1/minHeight:180 — a fixed height doesn't
  // grow with the fullscreen-expanded card's own 100dvh, so without this
  // change the expanded state would show a 180px chart with dead space
  // below it, same defect Frontier's own expand pass had to avoid.
  const mcChartCardEl = (
    <div
      ref={mcCardRef}
      className={`card${isMcExpanded ? ' chart-fullscreen-expanded' : ''}`}
      style={{ padding: '14px 16px', flexShrink: 0, display: 'flex', flexDirection: 'column' }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10, flexShrink: 0, flexWrap: 'wrap', gap: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)' }}>
            Monte Carlo - median outcomes
          </div>
          {isCompactViewport && (
            <button
              onClick={() => setMcExpandRequested(v => !v)}
              className="chart-expand-btn"
              aria-label={isMcExpanded ? 'Collapse chart' : 'Expand chart to full screen'}
              title={isMcExpanded ? 'Collapse' : 'Expand'}
            >
              {isMcExpanded ? '✕' : '⤢'}
            </button>
          )}
        </div>
        <div style={{ display: 'flex', gap: 14, fontSize: 10, flexWrap: 'wrap' }}>
          {[[A_COLOR, nameA, dataA.monte_carlo.p50_final], [B_COLOR, nameB, dataB.monte_carlo.p50_final]].map(([c, n, v]) => (
            <span key={n} style={{ display: 'flex', alignItems: 'center', gap: 5, color: 'var(--text-muted)' }}>
              <div style={{ width: 16, height: 2, background: c }} />
              {n} - <span style={{ color: c, fontFamily: 'var(--font-mono)', fontWeight: 600 }}>${Math.round(v).toLocaleString()}</span>
            </span>
          ))}
        </div>
      </div>
      {/* overflow:'hidden' — same containment backstop as the growth
          chart's wrapper above. */}
      <div style={{ flex: 1, minHeight: 180, overflow: 'hidden' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={mcData} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
            <XAxis dataKey="day" tick={{ fill:'var(--text-muted)', fontSize:10 }} tickLine={false} axisLine={false} label={{ value:'Trading days', position:'insideBottom', offset:-2, fill:'var(--text-muted)', fontSize:10 }} />
            <YAxis {...(mcYDomain ? { domain: mcYDomain } : {})} tickFormatter={v => `$${(v/1000).toFixed(0)}k`} tick={{ fill:'var(--text-muted)', fontSize:10 }} tickLine={false} axisLine={false} width={48} />
            <ReferenceLine y={dataA.monte_carlo.portfolio_value} stroke="rgba(var(--text-primary-rgb),0.08)" strokeDasharray="4 4" />
            <Tooltip contentStyle={TIP} formatter={v => [`$${Math.round(v).toLocaleString()}`, '']} labelFormatter={l => `Day ${l}`} />
            <Line type="monotone" dataKey="a" stroke={A_COLOR} strokeWidth={2} dot={false} name={nameA} />
            <Line type="monotone" dataKey="b" stroke={B_COLOR} strokeWidth={2} dot={false} name={nameB} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )

  return (
    // No longer its own scroll root (was className="tab-scroll-root",
    // height:'100%', overflowY:'auto') — this component is only ever
    // mounted inside CompareWrapper.jsx, never directly as a tab (App.jsx
    // routes 'compare' to CompareWrapper, which imports this as an inner
    // component). Giving it an independent height:100%/overflowY:auto root
    // made it its own bounded, independently-scrolling box nested inside
    // CompareWrapper's — CompareWrapper's own "why this tab matters" box
    // and Reconfigure/Re-run bar sat outside that box as fixed siblings,
    // so only this component's own content (the metrics table, both
    // charts, the summary InsightBox) actually scrolled. CompareWrapper is
    // now the single scroll root for the whole tab (its own root comment
    // has the other half of this); this renders at its natural content
    // height like any other plain child there.
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

      {/* Header — portfolio labels. Reuses .compare-ab-row, the exact same
          grid class/area shape as every MetricRow below, so "the header's
          two names still sit directly above their columns once the centre
          track is gone" is guaranteed by construction — both resolve the
          same 'a'/'b' areas to the same two column tracks, not just
          measured to agree by coincidence. Chip alignment branches on
          isCompactViewport for the same reason MetricRow's values do (see
          that component's own comment): below 768px the chip needs to sit
          centred over its now-centred column, not pinned to the inner edge
          where the vanished label track used to be. */}
      <div className="compare-ab-row" style={{ alignItems: 'center', flexShrink: 0 }}>
        <div style={{ gridArea: 'a', textAlign: isCompactViewport ? 'center' : 'right' }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, background: 'rgba(var(--signal-positive-rgb),0.1)', border: '1px solid rgba(var(--signal-positive-rgb),0.3)', padding: '8px 14px' }}>
            <div style={{ width: 10, height: 10, background: A_COLOR }} />
            <div>
              <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)' }}>{nameA}</div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{tickersA.join(', ')}</div>
            </div>
          </div>
        </div>
        <div style={{ gridArea: 'label', textAlign: 'center', fontSize: 11, color: 'var(--text-muted)', fontWeight: 600 }}>VS</div>
        <div style={{ gridArea: 'b', textAlign: isCompactViewport ? 'center' : 'left' }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, background: 'rgba(var(--signal-caution-rgb),0.1)', border: '1px solid rgba(var(--signal-caution-rgb),0.3)', padding: '8px 14px' }}>
            <div style={{ width: 10, height: 10, background: B_COLOR }} />
            <div>
              <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)' }}>{nameB}</div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{tickersB.join(', ')}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Winner banner */}
      {overallWinner && (
        <div style={{
          textAlign: 'center', padding: '10px',
          background: `rgba(${overallWinner === 'A' ? 'var(--signal-positive-rgb)' : 'var(--signal-caution-rgb)'},0.08)`,
          border: `1px solid ${overallColor}`,
          fontSize: 12, color: overallColor, fontWeight: 600, flexShrink: 0,
        }}>
          {overallWinner === 'A' ? nameA : nameB} wins on {Math.max(winsA, winsB)} of {metrics.length} metrics
        </div>
      )}

      {/* Two-column layout. .compare-summary-row: gridTemplateColumns lives
          in that CSS class, not inline, so the compact-viewport override
          can reach it (the usual inline-always-wins reason). Table first,
          chart second in DOM order — already what "stack below 768px:
          table first, then chart" needs, no reorder required, just letting
          the column collapse. Stacking is also what fixes the Cumulative
          Returns tooltip that clipped earlier: the chart card was ~105px
          wide against ~120px of tooltip content when it had to share a
          1fr 1fr row at phone width; full column width resolves that
          without touching the tooltip itself. */}
      <div className="compare-summary-row" style={{ display: 'grid', gap: 12, flexShrink: 0 }}>

        {/* Metrics comparison */}
        <div className="card" style={{ padding: '14px 16px' }}>
          <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)', marginBottom: 10 }}>
            Head to head
          </div>
          <MetricRow label="Return"    aVal={dataA.annualised_return}         bVal={dataB.annualised_return}         fmt={fmt}  lowerBetter={false} isCompactViewport={isCompactViewport} />
          <MetricRow label="Volatility" aVal={dataA.annualised_volatility}    bVal={dataB.annualised_volatility}    fmt={fmt}  lowerBetter={true}  isCompactViewport={isCompactViewport} />
          <MetricRow label="Sharpe"    aVal={dataA.sharpe_ratio}              bVal={dataB.sharpe_ratio}              fmt={fmtN} lowerBetter={false} isCompactViewport={isCompactViewport} />
          <MetricRow label="Sortino"   aVal={dataA.sortino_ratio}             bVal={dataB.sortino_ratio}             fmt={fmtN} lowerBetter={false} isCompactViewport={isCompactViewport} />
          <MetricRow label="Drawdown"  aVal={dataA.max_drawdown}              bVal={dataB.max_drawdown}              fmt={v => fmt(v)} lowerBetter={true} isCompactViewport={isCompactViewport} />
          <MetricRow label="VaR 95%"   aVal={dataA.var_cvar.var_pct}         bVal={dataB.var_cvar.var_pct}         fmt={v => fmt(v)} lowerBetter={true} isCompactViewport={isCompactViewport} />
          <MetricRow label="CVaR 95%"  aVal={dataA.var_cvar.cvar_pct}        bVal={dataB.var_cvar.cvar_pct}        fmt={v => fmt(v)} lowerBetter={true} isCompactViewport={isCompactViewport} />
          {dataA.beta_alpha && dataB.beta_alpha && (
            <>
              <MetricRow label="Beta"  aVal={dataA.beta_alpha.beta}          bVal={dataB.beta_alpha.beta}          fmt={fmtN} lowerBetter={true} isCompactViewport={isCompactViewport} />
              <MetricRow label="Alpha" aVal={dataA.beta_alpha.alpha}         bVal={dataB.beta_alpha.alpha}         fmt={fmt}  lowerBetter={false} isCompactViewport={isCompactViewport} />
            </>
          )}
        </div>

        {/* Growth chart — built below (growthChartCardEl), rendered either
            in its normal grid slot or through a portal when expanded. */}
        {isGrowthExpanded ? createPortal(growthChartCardEl, document.body) : growthChartCardEl}
      </div>

      {/* Monte Carlo comparison — built below (mcChartCardEl), same
          normal-slot-or-portal rendering as the growth chart above. */}
      {isMcExpanded ? createPortal(mcChartCardEl, document.body) : mcChartCardEl}

      {/* Secondary: restates numbers the Head to head table (Return, Sharpe)
          and the Monte Carlo chart's own legend (median outcome) already
          show, unlike "Why the comparison matters" which explains what the
          tab is for before the reader has looked at either — same
          reasoning as Monte Carlo's own primary/secondary split. */}
      <InsightBox
        label="Comparison summary"
        priority="secondary"
        tone={overallWinner ? 'good' : 'neutral'}
        text={overallWinner
          ? `<strong>${overallWinner === 'A' ? nameA : nameB}</strong> leads on ${Math.max(winsA, winsB)} of ${metrics.length} risk-adjusted metrics. Return: <strong>${fmt(dataA.annualised_return)}</strong> vs <strong>${fmt(dataB.annualised_return)}</strong>. Sharpe: <strong>${fmtN(dataA.sharpe_ratio)}</strong> vs <strong>${fmtN(dataB.sharpe_ratio)}</strong>. Monte Carlo median: <strong>$${Math.round(dataA.monte_carlo.p50_final).toLocaleString()}</strong> vs <strong>$${Math.round(dataB.monte_carlo.p50_final).toLocaleString()}</strong>.`
          : `Both portfolios are evenly matched across ${metrics.length} metrics. Look at individual metrics above to decide which fits your risk tolerance.`
        }
      />

    </div>
  )
}