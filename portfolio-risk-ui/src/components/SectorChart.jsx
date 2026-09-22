import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LabelList } from 'recharts'
import InsightBox from './InsightBox'
import MetricTooltip from './MetricTooltip'

const truncate = (label, max = 24) =>
  label.length > max ? `${label.slice(0, max - 1)}…` : label

// Recharts' default category-tick renderer routes through its internal <Text>
// component, which does its own pixel-width auto-wrap/truncation whenever the
// axis has a `width` prop — independent of (and on top of) tickFormatter. A
// plain <text> tick bypasses that entirely so `truncate` is the only thing
// that ever shortens the label.
//
// anchor='end' (desktop, default) is untouched: Recharts' own `x` already
// sits at the axis's right edge, flush against the bars, and right-aligned
// text reads back from there exactly as it always has.
//
// anchor='start' (phone — see isPhone below) ignores Recharts' `x`
// entirely and pins text to a small fixed left offset instead. Recharts'
// `x` for a left-side category axis is always the axis's RIGHT edge
// (adjacent to the plot area) regardless of text-anchor — switching only
// textAnchor to 'start' while keeping that x would start each label right
// where the bars begin and run it further right, into the plot itself.
// Anchoring to the axis's own left edge (~margin.left) instead is what
// actually closes the gap the fixed-165px column leaves on a narrow
// phone-width card — see SectorChart's own comment on yAxisWidth.
function SectorTick({ x, y, payload, anchor = 'end', leftX = 2 }) {
  return (
    <text
      x={anchor === 'start' ? leftX : x}
      y={y}
      dy={3}
      textAnchor={anchor}
      fill="var(--text-muted)"
      fontSize={10}
      fontFamily="var(--font-mono)"
    >
      {truncate(payload.value)}
    </text>
  )
}

function SectorTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  // The percentage is already shown next to the bar itself (LabelList,
  // below) — repeating it here told the viewer nothing new. The tickers
  // that got classified into this sector do.
  const tickers = payload[0].payload?.tickers
  return (
    <div style={{
      background: 'var(--surface-card)', border: 'var(--border-default)',
      padding: '8px 12px', fontSize: 11,
    }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      <div style={{ color: 'var(--text-primary)', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
        {tickers?.length ? tickers.join(', ') : '-'}
      </div>
    </div>
  )
}

export default function SectorChart({ sectorData, loading, isPhone }) {
  if (loading) {
    return (
      <div className="card" style={{
        padding: '10px 16px', height: 96, display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center', gap: 8, flexShrink: 0,
      }}>
        <svg className="spin" width="18" height="18" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="10" stroke="var(--signal-positive)" strokeWidth="3" strokeDasharray="40 20" />
        </svg>
        <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Loading sector exposure…</div>
      </div>
    )
  }

  // null: haven't fetched yet, or the fetch itself failed — say nothing.
  // []: fetch succeeded but no ticker resolved to a usable sector, most
  // commonly an all-ETF/fund portfolio — that's a real, expected outcome,
  // not a loading gap, so it gets its own explanatory state below rather
  // than silently rendering nothing.
  if (sectorData == null) return null

  // A ticker set to exactly 0% allocation (a valid Sidebar state — the user
  // typed it in but zeroed its weight) still resolves to a sector in
  // App.jsx's aggregation (weightBySector accumulates from 0, not skipped),
  // so it can show up here as a real row with a real bar — just one that's
  // 0px wide and reads "Media 0%", which looks like a rendering fault
  // rather than an accurate zero. Filtered here, not in App.jsx's
  // aggregation: this is the one place that decides what actually renders,
  // and nothing else consumes sectorData that would need the zero entries
  // kept.
  const nonZeroSectorData = sectorData.filter(d => d.weight > 0)

  if (!nonZeroSectorData.length) {
    return (
      <div className="card" style={{ padding: '10px 16px', flexShrink: 0 }}>
        <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)', marginBottom: 5 }}>
          <MetricTooltip metricKey="sector_exposure">Sector exposure</MetricTooltip>
        </div>
        <InsightBox
          tone="neutral"
          compact
          label="Not applicable"
          text="Your portfolio is made up of funds and ETFs, which each hold many companies across different sectors, so a single sector breakdown doesn't apply. Sector Exposure works best with individual stock holdings."
        />
      </div>
    )
  }

  const totalWeight = nonZeroSectorData.reduce((sum, d) => sum + d.weight, 0)
  const hasExcluded = totalWeight < 0.995

  // Recharts' category YAxis divides the plot height evenly per row and
  // silently drops a tick label (though not its bar) once rows get packed
  // too tight to fit — invisible with the 1-2 sectors a 3-ticker tech
  // portfolio usually produces, but a 4-5 ticker portfolio spanning three
  // or more sectors (e.g. tech + financials + energy) routinely hits it.
  // Scale the card to the row count instead of a fixed height so every
  // sector always gets enough room to keep its label.
  const chartHeight = Math.max(96, 34 + nonZeroSectorData.length * 24)

  // Desktop's fixed 165px label column is a small fraction of a wide card
  // there, so it's never actually been a problem — but on a phone-width
  // card (single column, ~290-360px total) it reserves nearly half the
  // chart for text, right-aligned with a lot of empty space to its own
  // left, and correspondingly compresses the bars into whatever's left.
  // Phone only: shrink the column to fit the longest truncated label
  // actually present (not a guessed constant — same data-driven-by-content
  // approach RiskAnalysis's correlation matrix already uses for its own
  // cell sizing) and left-align within it, so bars get the width back.
  // ~6.2px/char is Geist Mono's advance width at this 10px size.
  const longestLabel = Math.max(...nonZeroSectorData.map(d => truncate(d.sector).length))
  const yAxisWidth = isPhone ? Math.max(60, Math.min(140, Math.round(longestLabel * 6.2) + 12)) : 165

  return (
    <div className="card" style={{ padding: '10px 16px', height: chartHeight, display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 5 }}>
        <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)' }}>
          <MetricTooltip metricKey="sector_exposure">Sector exposure</MetricTooltip>
        </div>
        {hasExcluded && (
          <div style={{ fontSize: 'var(--text-micro)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)' }}>
            ETFs & unrecognized tickers excluded
          </div>
        )}
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={nonZeroSectorData}
            layout="vertical"
            style={{ background: 'transparent', fontSize: 11, fontFamily: 'var(--font-mono)' }}
            margin={{ top: 0, right: 36, bottom: 0, left: 4 }}
          >
            <XAxis type="number" hide domain={[0, 1]} />
            <YAxis
              type="category"
              dataKey="sector"
              tick={<SectorTick anchor={isPhone ? 'start' : 'end'} leftX={4} />}
              tickLine={false}
              axisLine={false}
              width={yAxisWidth}
            />
            <Tooltip content={<SectorTooltip />} cursor={{ fill: 'rgba(var(--text-primary-rgb),0.03)' }} />
            <Bar dataKey="weight" fill="var(--signal-positive)" radius={[0, 0, 0, 0]} barSize={8}>
              <LabelList
                dataKey="weight"
                position="right"
                formatter={v => `${(v * 100).toFixed(0)}%`}
                style={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
