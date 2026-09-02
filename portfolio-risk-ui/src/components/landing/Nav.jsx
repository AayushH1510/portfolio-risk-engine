import { useRef, useState } from 'react'
import { typeStyle } from './tokens'
import Logo from '../Logo'

function NavLink({ href, children }) {
  const [hovered, setHovered] = useState(false)
  return (
    <a
      href={href}
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

function NavAction({ link }) {
  const [hovered, setHovered] = useState(false)
  return (
    <a
      href={link.href}
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
          <span style={{ ...typeStyle('monoAction'), fontSize: '15px', letterSpacing: '0.02em', color: 'var(--color-text-body)' }}>
            {brand.name}
          </span>
        </a>

        <div style={{ display: 'flex', alignItems: 'center', gap: 'clamp(16px, 2vw, 34px)', flexWrap: 'wrap' }}>
          {nav.map((link) => (
            <NavLink key={link.href} href={link.href}>{link.label}</NavLink>
          ))}
        </div>

        <NavAction link={navAction} />
      </div>
    </nav>
  )
}
