import { useState } from 'react'
import { Link } from 'react-router-dom'
import SectionFrame from './SectionFrame'
import { typeStyle } from './tokens'

// Internal paths (starting with "/") route client-side via react-router;
// anything else (anchors, external) stays a plain <a> — same convention as
// the rest of the landing page's content-driven links.
function MoreLink({ link }) {
  const [hovered, setHovered] = useState(false)
  const style = {
    ...typeStyle('monoAction'),
    color: hovered ? 'var(--color-accent-mint)' : 'var(--color-text-secondary)',
    transition: `color var(--motion-surfaceTint-duration) var(--motion-surfaceTint-easing)`,
    textDecoration: 'none',
  }
  const props = { onMouseEnter: () => setHovered(true), onMouseLeave: () => setHovered(false), style }
  return link.href.startsWith('/')
    ? <Link to={link.href} {...props}>{link.label}</Link>
    : <a href={link.href} {...props}>{link.label}</a>
}

export default function Pillars({ block }) {
  return (
    <section
      id={block.id}
      style={{ padding: 'var(--layout-sectionPadYBand) var(--layout-gutter)', borderTop: '1px solid var(--color-line-subtle)', background: 'var(--color-bg-raised)' }}
    >
      <SectionFrame index={block.index} label={block.label}>
        <h2 style={{ ...typeStyle('displayM'), margin: '0 0 50px', color: 'var(--color-text-primary)', maxWidth: '720px', textWrap: 'balance' }}>
          {block.heading}
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(255px, 1fr))', gap: '44px clamp(30px, 4vw, 70px)' }}>
          {block.pillars.map((pillar) => (
            <div key={pillar.title} data-reveal>
              <h3 style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--color-accent-mint)', margin: '0 0 var(--space-4)' }}>
                {pillar.title}
              </h3>
              <p style={{ ...typeStyle('bodyM'), color: 'var(--color-text-dim)', margin: 0 }}>{pillar.body}</p>
            </div>
          ))}
        </div>
        {block.moreLink && (
          <div style={{ marginTop: '50px', paddingTop: 'var(--space-8)', borderTop: '1px solid var(--color-line-subtle)' }}>
            <MoreLink link={block.moreLink} />
          </div>
        )}
      </SectionFrame>
    </section>
  )
}
