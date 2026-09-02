import { Link } from 'react-router-dom'
import Logo from './Logo'

export const LANDING_BG = 'var(--surface-canvas)'

export function LandingPage({ children }) {
  return (
    <div className="grain-canvas" style={{
      height: '100vh', overflowY: 'auto',
      background: LANDING_BG, color: 'var(--text-primary)',
      fontFamily: 'var(--font-primary)',
      display: 'flex', flexDirection: 'column', position: 'relative',
    }}>
      {children}
    </div>
  )
}

export function LandingHeader({ crossLinkLabel, crossLinkTo }) {
  return (
    <header style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '20px 48px', maxWidth: 1140, margin: '0 auto', width: '100%',
      flexShrink: 0,
    }}>
      <Link to="/" style={{ display: 'inline-block', textDecoration: 'none' }}>
        <Logo variant="horizontal" size={30} ink="var(--text-primary)" />
      </Link>

      <div style={{ display: 'flex', alignItems: 'center', gap: 28 }}>
        <Link
          to={crossLinkTo}
          style={{ fontSize: 13, color: 'var(--text-secondary)', textDecoration: 'none', fontWeight: 500 }}
          onMouseEnter={e => { e.currentTarget.style.color = 'var(--text-primary)' }}
          onMouseLeave={e => { e.currentTarget.style.color = 'var(--text-secondary)' }}
        >
          {crossLinkLabel}
        </Link>
        <Link
          to="/app"
          className="btn-primary"
          style={{ display: 'inline-block', width: 'auto', textDecoration: 'none', padding: '9px 18px' }}
        >
          Launch app
        </Link>
      </div>
    </header>
  )
}

export function LandingHero({ headline, subhead, ctaLabel = 'Launch app', ctaTo = '/app' }) {
  return (
    <div style={{ textAlign: 'center', padding: '72px 24px 60px', maxWidth: 780, margin: '0 auto', flexShrink: 0 }}>
      <h1 style={{
        fontSize: 'var(--text-display)', fontWeight: 'var(--weight-semibold)', color: 'var(--text-primary)',
        fontFamily: 'var(--font-primary)',
        lineHeight: 'var(--leading-display)', letterSpacing: 'var(--tracking-display)', margin: '0 0 20px',
      }}>
        {headline}
      </h1>
      <p style={{
        fontSize: 16, color: 'var(--text-secondary)', lineHeight: 1.65,
        maxWidth: 620, margin: '0 auto 32px',
      }}>
        {subhead}
      </p>
      <Link
        to={ctaTo}
        className="btn-primary"
        style={{ display: 'inline-block', width: 'auto', padding: '13px 30px', fontSize: 13, textDecoration: 'none' }}
      >
        {ctaLabel}
      </Link>
    </div>
  )
}

export function LandingFeatureGrid({ features }) {
  return (
    <div style={{
      display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16,
      maxWidth: 1080, margin: '0 auto', padding: '0 24px 64px', width: '100%',
    }}>
      {features.map(f => (
        <div key={f.title} className="card" style={{ padding: '22px 22px 24px' }}>
          <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 10, lineHeight: 1.3 }}>
            {f.title}
          </div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            {f.body}
          </div>
        </div>
      ))}
    </div>
  )
}

export function LandingHowItWorks({ steps }) {
  return (
    <div style={{ maxWidth: 880, margin: '0 auto', padding: '0 24px 80px', width: '100%' }}>
      <div style={{
        fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)',
        fontFamily: 'var(--font-primary)', color: 'var(--text-muted)', textAlign: 'center', marginBottom: 32,
      }}>
        How it works
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 28 }}>
        {steps.map((label, i) => (
          <div key={label} style={{ textAlign: 'center' }}>
            <div style={{
              width: 34, height: 34,
              background: 'rgba(var(--signal-positive-rgb),0.12)', border: '1px solid var(--signal-positive)',
              color: 'var(--signal-positive)', fontWeight: 700, fontSize: 14, fontFamily: 'var(--font-mono)',
              display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 14px',
            }}>
              {i + 1}
            </div>
            <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>
              {label}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export function LandingFooter() {
  return (
    <footer style={{
      textAlign: 'center', padding: '24px 24px', fontSize: 11, color: 'var(--text-muted)',
      borderTop: 'var(--border-default)', marginTop: 'auto', flexShrink: 0,
    }}>
      <div style={{ marginBottom: 8 }}>
        Varense - educational tool only. Not financial advice. Past performance does not guarantee future results.
      </div>
      <div style={{ display: 'flex', gap: 16, justifyContent: 'center' }}>
        <Link to="/privacy" style={{ color: 'var(--text-muted)', textDecoration: 'none' }}>Privacy</Link>
        <Link to="/terms" style={{ color: 'var(--text-muted)', textDecoration: 'none' }}>Terms</Link>
      </div>
    </footer>
  )
}

// ─── Shared layout for Privacy / Terms ─────────────────────────────────────

export function TrustPage({ title, updated, children }) {
  return (
    <LandingPage>
      <LandingHeader crossLinkLabel="← Back to Varense" crossLinkTo="/" />
      <div style={{ maxWidth: 700, margin: '0 auto', padding: '32px 24px 80px', width: '100%', flex: 1 }}>
        <h1 style={{ fontSize: 'var(--text-heading-lg)', fontWeight: 'var(--weight-semibold)', fontFamily: 'var(--font-primary)', color: 'var(--text-primary)', letterSpacing: 'var(--tracking-heading-lg)', margin: '0 0 6px' }}>
          {title}
        </h1>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 36 }}>
          Last updated {updated}
        </div>
        {children}
      </div>
      <LandingFooter />
    </LandingPage>
  )
}

export function TrustSection({ heading, children }) {
  return (
    <div style={{ marginBottom: 28 }}>
      <h2 style={{ fontSize: 15, fontWeight: 'var(--weight-semibold)', fontFamily: 'var(--font-primary)', color: 'var(--signal-positive)', letterSpacing: '-0.01em', margin: '0 0 10px' }}>
        {heading}
      </h2>
      <div style={{ fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.75 }}>
        {children}
      </div>
    </div>
  )
}
