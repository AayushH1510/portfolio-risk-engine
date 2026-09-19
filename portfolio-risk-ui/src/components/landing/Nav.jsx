import { useRef, useState } from 'react'
import { typeStyle } from './tokens'
import Logo from '../Logo'
import useOverlay from '../../hooks/useOverlay'
import './tap-target.css'
import './nav-menu.css'

function NavLink({ href, children }) {
  const [hovered, setHovered] = useState(false)
  return (
    <a
      href={href}
      className="tap-target-navlink"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        ...typeStyle('monoNav'),
        // --color-text-faint reads fine on the flat card surfaces it's
        // designed for ("metric tile labels" per design/tokens.json), but
        // nav sits over Hero's animated smoke field — needs the brightest
        // text token to stay legible regardless of what's drifting behind
        // it, with the established accent-mint highlight on hover.
        color: hovered ? 'var(--color-accent-mint)' : 'var(--color-text-primary)',
        transition: `color var(--motion-surfaceTint-duration) var(--motion-surfaceTint-easing)`,
      }}
    >
      {children}
    </a>
  )
}

// Hamburger below 768px, morphing to an X while the menu is open — see
// nav-menu.css for the display toggle and tap-target.css for the 44x44 hit
// area. Rounded line caps and the landing palette match this component's
// own visual language (Newsreader/IBM Plex, rounded corners) rather than
// the app's sharp-cornered terminal aesthetic — DESIGN.md scopes those to
// the app only.
function MenuToggle({ open, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-expanded={open}
      aria-controls="nav-mobile-menu"
      aria-label={open ? 'Close menu' : 'Open menu'}
      className="nav-menu-toggle tap-target-menu-toggle"
      style={{
        alignItems: 'center',
        justifyContent: 'center',
        width: '36px',
        height: '36px',
        border: '1px solid var(--color-line-interactive)',
        borderRadius: 'var(--radius-sm)',
        background: 'transparent',
        color: 'var(--color-text-body)',
        cursor: 'pointer',
        flexShrink: 0,
      }}
    >
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
        {open ? (
          <>
            <line x1="3" y1="3" x2="13" y2="13" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            <line x1="13" y1="3" x2="3" y2="13" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </>
        ) : (
          <>
            <line x1="2" y1="4" x2="14" y2="4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            <line x1="2" y1="8" x2="14" y2="8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            <line x1="2" y1="12" x2="14" y2="12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </>
        )}
      </svg>
    </button>
  )
}

// The dropdown panel itself. useOverlay (hooks/useOverlay.js) supplies body
// scroll-lock, Tab focus-trap, and Escape-to-close — the same behaviour
// hook the app's sidebar drawer and Dashboard expand control use, reused
// here because it's exactly that: behaviour, not a visual token. The
// panel's own surface (rounded corners, --color-bg-panel, --shadow-card) is
// landing's own language — deliberately not the app's hairline-and-zero-
// radius card treatment.
function MobileMenu({ open, onClose, nav }) {
  const panelRef = useRef(null)
  useOverlay(open, panelRef, onClose)

  if (!open) return null

  return (
    <>
      <div
        onClick={onClose}
        style={{ position: 'fixed', inset: 0, zIndex: 60, background: 'var(--color-alpha-scrim)' }}
      />
      <div
        id="nav-mobile-menu"
        ref={panelRef}
        role="menu"
        style={{
          position: 'fixed',
          top: 'calc(var(--layout-navHeight) + 8px)',
          left: 'var(--layout-gutter)',
          right: 'var(--layout-gutter)',
          zIndex: 61,
          background: 'var(--color-bg-panel)',
          border: '1px solid var(--color-line-default)',
          borderRadius: 'var(--radius-lg)',
          boxShadow: 'var(--shadow-card)',
          padding: 'var(--space-4)',
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-1)',
        }}
      >
        {nav.map((link) => (
          <a
            key={link.href}
            href={link.href}
            role="menuitem"
            onClick={onClose}
            style={{
              ...typeStyle('monoNav'),
              fontSize: '14px',
              color: 'var(--color-text-primary)',
              padding: 'var(--space-4) var(--space-3)',
              borderRadius: 'var(--radius-xs)',
            }}
          >
            {link.label}
          </a>
        ))}
      </div>
    </>
  )
}

function NavAction({ link }) {
  const [hovered, setHovered] = useState(false)
  return (
    <a
      href={link.href}
      className="tap-target-navaction"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        ...typeStyle('monoNav'),
        letterSpacing: '0.06em',
        padding: '10px 20px',
        border: '1px solid var(--color-accent-mint)',
        borderRadius: 'var(--radius-xs)',
        background: hovered ? 'var(--color-accent-mint)' : 'transparent',
        color: hovered ? 'var(--color-bg-base)' : 'var(--color-accent-mint)',
        whiteSpace: 'nowrap',
        transition: `background var(--motion-surfaceTint-duration) var(--motion-surfaceTint-easing), color var(--motion-surfaceTint-duration) var(--motion-surfaceTint-easing)`,
      }}
    >
      {link.label}
    </a>
  )
}

export default function Nav({ brand, nav, navAction, onScrollRef }) {
  const [scrolled, setScrolled] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const ref = useRef(null)

  // Registered by the page via onScrollRef so a single scroll-motion tick
  // drives the nav crossfade too — see README > Interactions > Nav.
  if (onScrollRef) {
    onScrollRef.current = (scrollY) => setScrolled(scrollY > 24)
  }

  return (
    <nav
      ref={ref}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 50,
        minHeight: 'var(--layout-navHeight)',
        background: scrolled ? 'var(--color-alpha-scrim)' : 'transparent',
        backdropFilter: scrolled ? 'var(--motion-navTransition-blur)' : 'none',
        WebkitBackdropFilter: scrolled ? 'var(--motion-navTransition-blur)' : 'none',
        borderBottom: `1px solid ${scrolled ? 'var(--color-line-subtle)' : 'transparent'}`,
        transition: `background var(--motion-navTransition-duration) var(--motion-navTransition-easing), border-color var(--motion-navTransition-duration) var(--motion-navTransition-easing), backdrop-filter var(--motion-navTransition-duration) var(--motion-navTransition-easing)`,
      }}
    >
      <div
        style={{
          maxWidth: 'var(--layout-maxWidth)',
          margin: '0 auto',
          padding: `0 var(--layout-gutter)`,
          minHeight: 'var(--layout-navHeight)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 'clamp(16px, 2.5vw, 40px)',
          flexWrap: 'wrap',
        }}
      >
        <a href="#top" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', color: 'var(--color-text-body)' }}>
          <Logo variant="mark" size={26} ink="var(--color-text-body)" />
          <span className="nav-brand-name" style={{ ...typeStyle('monoAction'), fontSize: '15px', letterSpacing: '0.02em', color: 'var(--color-text-body)' }}>
            {brand.name}
          </span>
        </a>

        <div className="nav-links" style={{ alignItems: 'center', gap: 'clamp(16px, 2vw, 34px)', flexWrap: 'wrap' }}>
          {nav.map((link) => (
            <NavLink key={link.href} href={link.href}>{link.label}</NavLink>
          ))}
        </div>

        <MenuToggle open={menuOpen} onClick={() => setMenuOpen(o => !o)} />

        <NavAction link={navAction} />
      </div>

      <MobileMenu open={menuOpen} onClose={() => setMenuOpen(false)} nav={nav} />
    </nav>
  )
}
