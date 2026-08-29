// A hand-built replica of the real /app dashboard, tilted in 3D. `block.source`
// is 'mock' today; when the dashboard can render from a fixture without auth,
// swap in the real component inside a pointer-events:none shell and flip the
// flag to 'live' — see README > Roadmap note. Keep this mock as the SSR/no-JS
// fallback, don't delete it once that lands.
//
// Its internal micro-typography (tab labels, metric labels) replicates the
// real dashboard's own UI at miniature scale — it is illustrative content,
// not the landing page's own typographic voice, so those sizes are literal
// the way the growth-chart SVG path below is literal. Every colour still
// routes through the token custom properties.

function Dot() {
  return <span style={{ width: '9px', height: '9px', borderRadius: '50%', background: 'var(--color-chart-neutral300)', display: 'block' }} />
}

function WeightRow({ ticker, pct }) {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-text-muted)', marginBottom: '7px' }}>
        <span>{ticker}</span>
        <span style={{ color: 'var(--color-accent-mint)' }}>{pct}%</span>
      </div>
      <div style={{ height: '3px', background: 'var(--color-line-default)', borderRadius: 'var(--radius-xs)' }}>
        <div style={{ width: `${pct}%`, height: '3px', background: 'var(--color-accent-mint)', borderRadius: 'var(--radius-xs)' }} />
      </div>
    </div>
  )
}

function MetricTile({ label, value, color, accentTop }) {
  return (
    <div
      style={{
        border: '1px solid var(--color-line-default)',
        borderTop: accentTop ? `1px solid ${accentTop}` : undefined,
        borderRadius: 'var(--radius-xs)',
        padding: '14px 16px',
        background: 'var(--color-bg-inset)',
      }}
    >
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', letterSpacing: '0.14em', color: 'var(--color-text-ghost)', marginBottom: '8px' }}>{label}</div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '21px', color }}>{value}</div>
    </div>
  )
}

export default function ProductPlate({ block }) {
  return (
    <section style={{ position: 'relative', padding: '40px 40px var(--layout-sectionPadYBand)', perspective: '2200px' }}>
      <div data-parallax="0.22" style={{ maxWidth: 'var(--layout-maxWidth)', margin: '0 auto' }}>
        <div
          style={{
            transform: 'rotateX(11deg) scale(0.985)',
            transformOrigin: '50% 0%',
            border: '1px solid var(--color-line-strong)',
            borderRadius: 'var(--radius-2xl)',
            background: 'var(--color-bg-panel)',
            boxShadow: 'var(--shadow-plate)',
            overflow: 'hidden',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '26px', padding: '0 22px', height: '46px', borderBottom: '1px solid var(--color-line-default)', background: 'var(--color-bg-raised)' }}>
            <div style={{ display: 'flex', gap: '7px' }}>
              <Dot /><Dot /><Dot />
            </div>
            <div style={{ display: 'flex', gap: '26px', fontFamily: 'var(--font-mono)', fontSize: '11px', letterSpacing: '0.13em', textTransform: 'uppercase', color: 'var(--color-text-ghost)' }}>
              {block.tabs.map((tab) => (
                <span
                  key={tab}
                  style={
                    tab === block.activeTab
                      ? { color: 'var(--color-text-primary)', borderBottom: '2px solid var(--color-accent-mint)', paddingBottom: '2px' }
                      : undefined
                  }
                >
                  {tab}
                </span>
              ))}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 232px) minmax(0, 1fr)', minHeight: '460px' }}>
            <div style={{ borderRight: '1px solid var(--color-line-default)', padding: '22px', background: 'var(--color-bg-raised)', display: 'flex', flexDirection: 'column', gap: '22px' }}>
              <div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', letterSpacing: '0.16em', color: 'var(--color-text-ghost)', marginBottom: '10px' }}>1. STOCKS</div>
                <div style={{ border: '1px solid var(--color-line-strong)', borderRadius: 'var(--radius-xs)', padding: '11px 12px', fontFamily: 'var(--font-mono)', fontSize: '13px', color: 'var(--color-text-body)', background: 'var(--color-bg-inset)' }}>
                  AAPL, MSFT, GOOGL
                </div>
              </div>
              <div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', letterSpacing: '0.16em', color: 'var(--color-text-ghost)', marginBottom: '10px' }}>2. PORTFOLIO VALUE</div>
                <div style={{ border: '1px solid var(--color-line-strong)', borderRadius: 'var(--radius-xs)', padding: '11px 12px', fontFamily: 'var(--font-mono)', fontSize: '13px', color: 'var(--color-text-body)', background: 'var(--color-bg-inset)' }}>
                  $ 10000
                </div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', letterSpacing: '0.16em', color: 'var(--color-text-ghost)' }}>3. WEIGHTS</div>
                <WeightRow ticker="AAPL" pct={34} />
                <WeightRow ticker="MSFT" pct={33} />
                <WeightRow ticker="GOOGL" pct={33} />
              </div>
              <div
                style={{
                  marginTop: 'auto',
                  textAlign: 'center',
                  background: 'var(--color-accent-mint)',
                  color: 'var(--color-bg-base)',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '12.5px',
                  letterSpacing: '0.1em',
                  padding: '13px',
                  borderRadius: 'var(--radius-xs)',
                  fontWeight: 500,
                }}
              >
                RUN ANALYSIS
              </div>
            </div>

            <div style={{ padding: '22px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(112px, 1fr))', gap: '12px' }}>
                <MetricTile label="ANNUAL RETURN" value="33.6%" color="var(--color-accent-mint)" accentTop="var(--color-accent-mint)" />
                <MetricTile label="VOLATILITY" value="19.6%" color="var(--color-text-body)" />
                <MetricTile label="SORTINO" value="2.43" color="var(--color-text-body)" />
                <MetricTile label="MAX DRAWDOWN" value="−17.6%" color="var(--color-signal-warning)" accentTop="var(--color-signal-warning)" />
              </div>

              <div style={{ flex: 1, border: '1px solid var(--color-line-default)', borderRadius: 'var(--radius-xs)', background: 'var(--color-bg-inset)', padding: '18px 20px', display: 'flex', flexDirection: 'column' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', letterSpacing: '0.16em', color: 'var(--color-text-faint)' }}>PORTFOLIO GROWTH</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--color-text-ghost)', display: 'flex', gap: '16px' }}>
                    <span style={{ color: 'var(--color-accent-mint)' }}>— Your portfolio</span>
                    <span>--- S&amp;P 500</span>
                  </span>
                </div>
                <svg viewBox="0 0 800 230" preserveAspectRatio="none" style={{ width: '100%', flex: 1, minHeight: '200px', display: 'block' }}>
                  <defs>
                    <linearGradient id="vgrow" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--color-accent-mint)" stopOpacity="0.28" />
                      <stop offset="100%" stopColor="var(--color-accent-mint)" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  <path
                    d="M0,192 L20,184 L40,190 L60,170 L80,177 L100,156 L120,161 L140,145 L160,152 L180,131 L200,138 L220,122 L240,133 L260,112 L280,124 L300,105 L320,119 L340,101 L360,115 L380,138 L400,158 L420,178 L440,148 L460,132 L480,118 L500,100 L520,90 L540,97 L560,80 L580,87 L600,68 L620,75 L640,58 L660,65 L680,48 L700,55 L720,36 L740,43 L760,26 L780,31 L800,14 L800,230 L0,230 Z"
                    fill="url(#vgrow)"
                  />
                  <path
                    d="M0,192 L20,184 L40,190 L60,170 L80,177 L100,156 L120,161 L140,145 L160,152 L180,131 L200,138 L220,122 L240,133 L260,112 L280,124 L300,105 L320,119 L340,101 L360,115 L380,138 L400,158 L420,178 L440,148 L460,132 L480,118 L500,100 L520,90 L540,97 L560,80 L580,87 L600,68 L620,75 L640,58 L660,65 L680,48 L700,55 L720,36 L740,43 L760,26 L780,31 L800,14"
                    fill="none"
                    stroke="var(--color-accent-mint)"
                    strokeWidth="2"
                    vectorEffect="non-scaling-stroke"
                  />
                  <path
                    d="M0,200 L60,196 L120,188 L180,182 L240,176 L300,170 L360,166 L400,180 L440,192 L480,176 L540,164 L600,152 L660,140 L720,128 L800,116"
                    fill="none"
                    stroke="var(--color-text-ghost)"
                    strokeWidth="1.5"
                    strokeDasharray="5 5"
                    vectorEffect="non-scaling-stroke"
                  />
                </svg>
              </div>

              <div style={{ borderLeft: '2px solid var(--color-accent-mint)', background: 'var(--color-alpha-mintWash)', padding: '13px 18px' }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', letterSpacing: '0.16em', color: 'var(--color-accent-mint)', marginBottom: '6px' }}>RETURN</div>
                <div style={{ fontSize: '12.5px', color: 'var(--color-text-muted)' }}>
                  Portfolio grew at <span style={{ color: 'var(--color-text-body)' }}>33.6%/yr</span>. Beat the S&amp;P 500 by{' '}
                  <span style={{ color: 'var(--color-text-body)' }}>14.5%</span>. Median 1-year projection:{' '}
                  <span style={{ color: 'var(--color-text-body)' }}>$13,603</span>.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
