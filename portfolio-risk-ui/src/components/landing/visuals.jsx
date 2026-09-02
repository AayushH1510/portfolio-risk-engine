// The four feature deep-dive visuals. All colour comes from CSS custom
// properties; path/bar geometry is fixed content (matching the reference
// exactly), not a design token, the same way chart data in the real app
// isn't a token either.

const DIST_HEIGHTS = [6, 11, 19, 31, 48, 66, 84, 100, 92, 74, 55, 38, 24, 14, 7]

function distBarColor(i) {
  if (i < 3) return 'var(--color-signal-negative)'
  if (i === 3) return 'var(--color-signal-warning)'
  if (i === 7) return 'var(--color-chart-neutral100)' // peak
  if (i >= 12) return 'var(--color-chart-neutral300)'
  return 'var(--color-chart-neutral200)'
}

export function DistributionVisual() {
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: '4px', height: '190px' }}>
        {DIST_HEIGHTS.map((h, i) => (
          <div key={i} style={{ flex: 1, height: `${h}%`, background: distBarColor(i), borderRadius: 'var(--radius-xs) var(--radius-xs) 0 0' }} />
        ))}
      </div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '1px',
          background: 'var(--color-line-default)',
          border: '1px solid var(--color-line-default)',
          borderRadius: 'var(--radius-sm)',
          overflow: 'hidden',
          marginTop: 'var(--space-7)',
        }}
      >
        <div style={{ background: 'var(--color-bg-inset)', padding: '16px 18px' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', letterSpacing: '0.14em', color: 'var(--color-text-faint)', marginBottom: 'var(--space-2)' }}>VaR 95%</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '24px', color: 'var(--color-signal-negative)' }}>−$180</div>
        </div>
        <div style={{ background: 'var(--color-bg-inset)', padding: '16px 18px' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', letterSpacing: '0.14em', color: 'var(--color-text-faint)', marginBottom: 'var(--space-2)' }}>CVaR 95%</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '24px', color: 'var(--color-signal-negative)' }}>−$245</div>
        </div>
      </div>
    </div>
  )
}

// Deterministic LCG so the walk paths never change between renders — no
// hydration-mismatch risk (this app has no SSR) but keeping it seeded also
// means the visual is reproducible for screenshots/tests. See README > "Generate
// the walks from a fixed seed."
function seededPaths() {
  let seed = 20260828
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    return seed / 0x7fffffff
  }
  const paths = []
  for (let i = 0; i < 24; i++) {
    let y = 150
    let d = 'M0,150'
    for (let x = 26; x <= 520; x += 26) {
      y += (rnd() - 0.53) * 26
      y = Math.max(14, Math.min(236, y))
      d += ` L${x},${y.toFixed(1)}`
    }
    paths.push({ d, o: (0.07 + rnd() * 0.11).toFixed(3) })
  }
  return paths
}
const MC_PATHS = seededPaths()

export function SimulationPathsVisual() {
  return (
    <div>
      <svg viewBox="0 0 520 250" style={{ width: '100%', height: '250px', display: 'block' }}>
        {MC_PATHS.map((p, i) => (
          <path key={i} d={p.d} fill="none" stroke="var(--color-text-body)" strokeWidth="0.7" opacity={p.o} />
        ))}
        <path
          d="M0,150 L65,140 L130,132 L195,118 L260,104 L325,92 L390,76 L455,62 L520,48"
          fill="none"
          stroke="var(--color-accent-mint)"
          strokeWidth="2.2"
        />
        <path
          d="M0,150 L65,158 L130,164 L195,162 L260,170 L325,166 L390,174 L455,170 L520,178"
          fill="none"
          stroke="var(--color-signal-negative)"
          strokeWidth="1.4"
          strokeDasharray="4 4"
        />
      </svg>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontFamily: 'var(--font-mono)',
          fontSize: '10.5px',
          color: 'var(--color-text-ghost)',
          marginTop: 'var(--space-4)',
          paddingTop: 'var(--space-4)',
          borderTop: '1px solid var(--color-line-default)',
        }}
      >
        <span style={{ color: 'var(--color-signal-negative)' }}>Bad (5th) - $9,823</span>
        <span style={{ color: 'var(--color-accent-mint)' }}>Median - $13,603</span>
        <span>Good (95th) - $18,909</span>
      </div>
    </div>
  )
}

const STRESS_ROWS = [
  { label: '2008 - GLOBAL FINANCIAL CRISIS', pct: '−48.2%', width: '96%', deep: 'var(--color-signal-negativeDeep)', signal: 'var(--color-signal-negative)' },
  { label: '2020 - COVID CRASH', pct: '−31.7%', width: '63%', deep: 'var(--color-signal-negativeDeep)', signal: 'var(--color-signal-negative)' },
  { label: '2022 - RATE SHOCK', pct: '−24.1%', width: '48%', deep: 'var(--color-signal-warningDeep)', signal: 'var(--color-signal-warning)' },
]

export function StressBarsVisual() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>
      {STRESS_ROWS.map((row) => (
        <div key={row.label}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--space-3)' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--color-text-secondary)', letterSpacing: '0.08em' }}>{row.label}</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '15px', color: row.signal }}>{row.pct}</span>
          </div>
          <div style={{ height: '6px', background: 'var(--color-line-subtle)', borderRadius: 'var(--radius-xs)' }}>
            <div style={{ width: row.width, height: '6px', background: `linear-gradient(90deg, ${row.deep}, ${row.signal})`, borderRadius: 'var(--radius-xs)' }} />
          </div>
        </div>
      ))}
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-text-ghost)', borderTop: '1px solid var(--color-line-default)', paddingTop: 'var(--space-5)', lineHeight: 1.7 }}>
        Recovery to prior peak: <span style={{ color: 'var(--color-text-secondary)' }}>18mo · 5mo · 14mo</span>
      </div>
    </div>
  )
}

const FRONTIER_DOTS = [
  [96, 182], [140, 150], [188, 164], [214, 118],
  [266, 134], [298, 96], [342, 110], [378, 74],
]

export function FrontierVisual() {
  return (
    <svg viewBox="0 0 460 260" style={{ width: '100%', height: '260px', display: 'block' }}>
      <line x1="40" y1="230" x2="440" y2="230" stroke="var(--color-line-default)" strokeWidth="1" />
      <line x1="40" y1="20" x2="40" y2="230" stroke="var(--color-line-default)" strokeWidth="1" />
      <path d="M60,215 C140,120 240,68 430,42" fill="none" stroke="var(--color-accent-mint)" strokeWidth="2" />
      {FRONTIER_DOTS.map(([cx, cy], i) => (
        <circle key={i} cx={cx} cy={cy} r="2.5" fill="var(--color-chart-neutral100)" />
      ))}
      <circle cx="248" cy="152" r="6" fill="var(--color-signal-warning)" />
      <circle cx="248" cy="152" r="12" fill="none" stroke="var(--color-signal-warning)" strokeWidth="1" opacity="0.4" />
      <circle cx="286" cy="88" r="6" fill="var(--color-accent-mint)" />
      <circle cx="286" cy="88" r="12" fill="none" stroke="var(--color-accent-mint)" strokeWidth="1" opacity="0.4" />
      <text x="248" y="180" fill="var(--color-signal-warning)" fontFamily="var(--font-mono)" fontSize="10" textAnchor="middle">YOU</text>
      <text x="286" y="70" fill="var(--color-accent-mint)" fontFamily="var(--font-mono)" fontSize="10" textAnchor="middle">OPTIMAL</text>
      <text x="240" y="252" fill="var(--color-text-ghost)" fontFamily="var(--font-mono)" fontSize="9.5" textAnchor="middle">VOLATILITY</text>
    </svg>
  )
}

export const VISUALS = {
  distribution: DistributionVisual,
  simulationPaths: SimulationPathsVisual,
  stressBars: StressBarsVisual,
  frontier: FrontierVisual,
}
