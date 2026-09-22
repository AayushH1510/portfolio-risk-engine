import { Link } from 'react-router-dom'
import Logo from './Logo'
import { useWindowScroll } from '../hooks/useWindowScroll'

export const LANDING_BG = 'var(--surface-canvas)'

// height:100vh (this div was its own internally-scrolling box, overflowY:
// 'auto') capped this element's own box at exactly one viewport tall
// regardless of content — but .grain-canvas::before (index.css, the shared
// grain-texture pseudo-element) sizes itself via inset:0 against ITS
// CONTAINING BLOCK, which is this div. On a page short enough to fit one
// screen that was harmless; on /privacy, /terms, /pro — real content that
// runs 1160-1360px tall against a 900px viewport at 1440px, more at
// narrower widths — the texture stayed pinned to the original top 100vh of
// content while the box itself kept scrolling internally past it, so
// anything scrolled below that first viewport-height was painted with no
// grain at all: a real, visible seam where the texture (which reads as
// giving the flat --surface-canvas colour its depth) just stops and the
// page goes flat. Confirmed directly, not guessed: scrollHeight measured
// 1362/1277/1164px against a 900px box on those three routes.
//
// minHeight (not height) keeps the one thing height:100vh was actually
// for — LandingFooter's marginTop:'auto' pins it to the bottom of a SHORT
// page instead of leaving a gap under sparse content — while letting the
// box grow past 100vh for real content instead of clipping its own
// scrollable overflow.
//
// useWindowScroll() is required alongside this, not optional: index.css's
// `html, body, #root { height:100%; overflow:hidden }` (there for /app's
// fixed-viewport shell) means a div that merely stops capping its OWN
// height still can't be seen past one viewport — its ancestors clip it
// regardless. useWindowScroll neutralises exactly that constraint while a
// page using it is mounted (the same mechanism Landing.jsx/Methodology.jsx
// already use for the identical reason, restored on unmount so /app is
// unaffected) — this was the missing half of the fix, caught by actually
// re-measuring scrollHeight after the minHeight change alone, which came
// back clipped to the viewport (900/900, not scrollable at all) rather
// than reachable. With both pieces in place, .grain-canvas::before's
// inset:0 sizing covers the whole page at every scroll position, with no
// separate fade/taper needed to hide a boundary that no longer exists.
export function LandingPage({ children }) {
  useWindowScroll()
  return (
    <div className="grain-canvas" style={{
      minHeight: '100vh',
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
    // flexWrap + a clamp() gap — same approach as landing/Nav.jsx, which
    // already holds cleanly at 320px. Without it, the fixed 48px padding
    // plus logo + crosslink + button never fit one line below ~360px, and
    // the button was getting pushed 30px past the edge, truncated to
    // "LAUNC" (RESPONSIVE_AUDIT.md's landing survey). Two-level wrap, same
    // as Nav.jsx: the outer row can drop the button-group to its own line,
    // and that group can itself wrap crosslink/button onto separate lines.
    <header style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      flexWrap: 'wrap', rowGap: 12,
      padding: '20px clamp(20px, 6vw, 48px)', maxWidth: 1140, margin: '0 auto', width: '100%',
      flexShrink: 0,
    }}>
      <Link to="/" style={{ display: 'inline-block', textDecoration: 'none' }}>
        <Logo variant="horizontal" size={30} ink="var(--text-primary)" />
      </Link>

      <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 'clamp(14px, 4vw, 28px)' }}>
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
    // minmax(min(300px, 100%), 1fr) — the same container-starvation fix
    // RESPONSIVE_AUDIT.md's landing pass already applied to Pricing.jsx's
    // tier grid: a bare minmax(300px, 1fr) floor is wider than the ~272px
    // available at 320px (320 viewport - 24px padding either side), so the
    // single column that survives auto-fit still can't shrink below 300px
    // and overflows its own container by exactly that gap (measured: a
    // 300px-wide card against a 272px column, a real, reproducible 4px
    // page-level overflow at 320px, not a rounding artifact). min(300px,
    // 100%) caps the floor at whatever's actually available, so the column
    // genuinely fills 100% instead of forcing a wider one. This was never
    // caught before because it was silently absorbed by this page's own
    // now-fixed internally-scrolling container (LandingPage, see its own
    // comment) rather than measured as real document overflow — invisible
    // to any check that only ever looked at document.documentElement.
    <div style={{
      display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(300px, 100%), 1fr))', gap: 16,
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
