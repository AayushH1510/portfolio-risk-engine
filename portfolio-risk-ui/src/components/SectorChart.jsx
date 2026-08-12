import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LabelList } from 'recharts'

const truncate = (label, max = 24) =>
  label.length > max ? `${label.slice(0, max - 1)}…` : label

// Recharts' default category-tick renderer routes through its internal <Text>
// component, which does its own pixel-width auto-wrap/truncation whenever the
// axis has a `width` prop — independent of (and on top of) tickFormatter. A
// plain <text> tick bypasses that entirely so `truncate` is the only thing
// that ever shortens the label.
function SectorTick({ x, y, payload }) {
  return (
    <text x={x} y={y} dy={3} textAnchor="end" fill="var(--text-muted)" fontSize={10} fontFamily="monospace">
      {truncate(payload.value)}
    </text>
  )
}

function SectorTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--card)', border: '1px solid var(--border)',
      borderRadius: 8, padding: '8px 12px', fontSize: 11,
    }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      <div style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
        {(payload[0].value * 100).toFixed(1)}%
      </div>
    </div>
  )
}

export default function SectorChart({ sectorData }) {
  if (!sectorData?.length) return null

  const totalWeight = sectorData.reduce((sum, d) => sum + d.weight, 0)
  const hasExcluded = totalWeight < 0.995

  return (
    <div className="card" style={{ padding: '14px 16px', height: 120, display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
        <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-secondary)' }}>
          Sector exposure
        </div>
        {hasExcluded && (
          <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            ETFs & unrecognized tickers excluded
          </div>
        )}
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={sectorData}
            layout="vertical"
            style={{ background: 'transparent', fontSize: 11, fontFamily: 'monospace' }}
            margin={{ top: 0, right: 36, bottom: 0, left: 4 }}
          >
            <XAxis type="number" hide domain={[0, 1]} />
            <YAxis
              type="category"
              dataKey="sector"
              tick={<SectorTick />}
              tickLine={false}
              axisLine={false}
              width={165}
            />
            <Tooltip content={<SectorTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
            <Bar dataKey="weight" fill="#52b788" radius={[0, 4, 4, 0]} barSize={8}>
              <LabelList
                dataKey="weight"
                position="right"
                formatter={v => `${(v * 100).toFixed(0)}%`}
                style={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'monospace' }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
