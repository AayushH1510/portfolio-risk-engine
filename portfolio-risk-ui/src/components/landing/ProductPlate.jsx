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
//
// The 232px + 1fr layout stacks below 768px — see product-plate.css.

import './product-plate.css'

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

          <div className="product-plate-grid">
            <div className="product-plate-sidebar" style={{ padding: '22px', background: 'var(--color-bg-raised)', display: 'flex', flexDirection: 'column', gap: '22px' }}>
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
                    <span style={{ color: 'var(--color-accent-mint)' }}>- Your portfolio</span>
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
                  {/* Points generated from a seeded random walk (scripts/
                      gen-chart.mjs — same deterministic-LCG technique as
                      this file's own visuals.jsx uses for the Monte Carlo
                      paths), then hard-coded here rather than computed at
                      runtime: irregular per-step volatility with slow
                      regime drift (quiet vs. choppy stretches), one
                      deliberate drawdown with a sharp decline and an
                      uneven, longer recovery (not a symmetric V), pinned to
                      the same start/end y-values and viewBox as the
                      original hand-authored sawtooth so nothing else in
                      the layout moves. Benchmark is its own independent
                      draw with a shallower trend and its own offset,
                      smaller dip — tracks loosely rather than running
                      parallel to the portfolio line. */}
                  <path
                    d="M0,192.0 L18,191.6 L36,186.4 L55,179.5 L73,176.4 L91,170.3 L109,163.5 L127,157.1 L145,148.7 L164,148.3 L182,144.2 L200,132.0 L218,121.4 L236,111.3 L255,117.5 L273,120.0 L291,111.0 L309,101.3 L327,96.3 L345,95.6 L364,104.3 L382,111.5 L400,125.2 L418,131.0 L436,150.9 L455,145.5 L473,138.4 L491,134.7 L509,121.4 L527,102.9 L545,93.5 L564,90.5 L582,78.0 L600,71.1 L618,64.3 L636,67.0 L655,73.1 L673,70.8 L691,55.3 L709,41.8 L727,21.8 L745,38.8 L764,14.0 L782,21.6 L800,14.0 L800,230 L0,230 Z"
                    fill="url(#vgrow)"
                  />
                  <path
                    d="M0,192.0 L18,191.6 L36,186.4 L55,179.5 L73,176.4 L91,170.3 L109,163.5 L127,157.1 L145,148.7 L164,148.3 L182,144.2 L200,132.0 L218,121.4 L236,111.3 L255,117.5 L273,120.0 L291,111.0 L309,101.3 L327,96.3 L345,95.6 L364,104.3 L382,111.5 L400,125.2 L418,131.0 L436,150.9 L455,145.5 L473,138.4 L491,134.7 L509,121.4 L527,102.9 L545,93.5 L564,90.5 L582,78.0 L600,71.1 L618,64.3 L636,67.0 L655,73.1 L673,70.8 L691,55.3 L709,41.8 L727,21.8 L745,38.8 L764,14.0 L782,21.6 L800,14.0"
                    fill="none"
                    stroke="var(--color-accent-mint)"
                    strokeWidth="2"
                    vectorEffect="non-scaling-stroke"
                  />
                  <path
                    d="M0,200.0 L18,195.4 L36,196.0 L55,191.7 L73,187.4 L91,189.4 L109,186.0 L127,183.1 L145,183.4 L164,182.3 L182,181.4 L200,178.6 L218,176.9 L236,175.7 L255,171.6 L273,167.6 L291,168.6 L309,166.6 L327,168.3 L345,163.9 L364,161.6 L382,157.5 L400,156.2 L418,158.1 L436,163.4 L455,167.7 L473,172.9 L491,180.5 L509,175.4 L527,170.8 L545,162.1 L564,153.2 L582,145.9 L600,141.8 L618,136.4 L636,132.1 L655,128.5 L673,125.7 L691,124.0 L709,122.1 L727,120.2 L745,121.0 L764,120.9 L782,118.0 L800,116.0"
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
