import { useState } from 'react'

function FooterLink({ link }) {
  const [hovered, setHovered] = useState(false)
  return (
    <a
      href={link.href}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        fontSize: '14px',
        color: hovered ? 'var(--color-text-body)' : 'var(--color-text-dim)',
        transition: `color var(--motion-surfaceTint-duration) var(--motion-surfaceTint-easing)`,
      }}
    >
      {link.label}
    </a>
  )
}

export default function Footer({ brand, footer }) {
  return (
    <footer style={{ borderTop: '1px solid var(--color-line-subtle)', padding: '60px var(--layout-gutter) 46px' }}>
      <div
        style={{
          maxWidth: 'var(--layout-maxWidth)',
          margin: '0 auto',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          gap: 'var(--space-12)',
          flexWrap: 'wrap',
          minWidth: 0,
        }}
      >
        <div style={{ maxWidth: '380px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-5)' }}>
            <span
              style={{
                width: '22px',
                height: '22px',
                borderRadius: 'var(--radius-sm)',
                background: 'linear-gradient(145deg, var(--color-accent-mint), var(--color-accent-mintDeep))',
                display: 'block',
              }}
            />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '14px', color: 'var(--color-text-body)' }}>{brand.name}</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-text-quiet)' }}>{brand.version}</span>
          </div>
          <p style={{ fontSize: '13px', lineHeight: 1.7, color: 'var(--color-text-ghost)', margin: 0 }}>{footer.disclaimer}</p>
        </div>

        <div style={{ display: 'flex', gap: 'clamp(32px, 5vw, 70px)', flexWrap: 'wrap' }}>
          {footer.columns.map((col) => (
            <div key={col.heading} style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
              <span
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '10.5px',
                  letterSpacing: '0.16em',
                  textTransform: 'uppercase',
                  color: 'var(--color-text-quiet)',
                  marginBottom: 'var(--space-1)',
                }}
              >
                {col.heading}
              </span>
              {col.links.map((link) => (
                <FooterLink key={link.href} link={link} />
              ))}
            </div>
          ))}
        </div>
      </div>

      <div
        style={{
          maxWidth: 'var(--layout-maxWidth)',
          margin: '46px auto 0',
          paddingTop: 'var(--space-6)',
          borderTop: '1px solid var(--color-line-subtle)',
          fontFamily: 'var(--font-mono)',
          fontSize: '11.5px',
          color: 'var(--color-text-quiet)',
          display: 'flex',
          justifyContent: 'space-between',
          gap: 'var(--space-5)',
          flexWrap: 'wrap',
        }}
      >
        <span>{footer.copyright}</span>
        <span>{footer.tagline}</span>
      </div>
    </footer>
  )
}
