import { useState } from 'react'
import SectionFrame from './SectionFrame'
import { typeStyle } from './tokens'
import './tap-target.css'

// Same component/convention as Pillars.jsx's own MoreLink (quiet text,
// mint on hover, tap-target-morelink's expanded hit area) — not reused
// directly because Pillars.jsx's version branches to a React Router
// <Link> for any href starting with "/", and /overview is a static file
// (public/overview/index.html) outside the SPA's route table: a <Link>
// there would try to client-match it against <Routes> and render nothing.
// This version is always a plain <a>, which is also correct for every
// other prose block's moreLink since none of them are SPA routes either.
function MoreLink({ link }) {
  const [hovered, setHovered] = useState(false)
  return (
    <a
      href={link.href}
      className="tap-target-morelink"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        ...typeStyle('monoAction'),
        display: 'inline-block',
        color: hovered ? 'var(--color-accent-mint)' : 'var(--color-text-secondary)',
        textDecoration: 'none',
        transition: `color var(--motion-surfaceTint-duration) var(--motion-surfaceTint-easing)`,
      }}
    >
      {link.label}
    </a>
  )
}

// content.ts paragraphs use **bold** to mark spans that render in
// text.body colour (weight unchanged) — split on the marker pairs.
function renderInline(text) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return (
        <span key={i} style={{ color: 'var(--color-text-body)' }}>
          {part.slice(2, -2)}
        </span>
      )
    }
    return <span key={i}>{part}</span>
  })
}

export default function Prose({ block }) {
  return (
    <section id={block.id} style={{ padding: 'var(--layout-sectionPadY) var(--layout-gutter)' }}>
      <SectionFrame index={block.index} label={block.label}>
        <div data-reveal>
          <h2 style={{ ...typeStyle('displayM'), margin: '0 0 var(--space-8)', color: 'var(--color-text-primary)', maxWidth: '780px', textWrap: 'balance' }}>
            {block.heading}
          </h2>
          {block.paragraphs.map((p, i) => (
            <p
              key={i}
              style={{
                ...typeStyle('lead'),
                color: 'var(--color-text-muted)',
                maxWidth: '700px',
                margin: i === block.paragraphs.length - 1 ? 0 : '0 0 var(--space-6)',
                textWrap: 'pretty',
              }}
            >
              {renderInline(p)}
            </p>
          ))}
          {block.moreLink && (
            <div style={{ marginTop: 'var(--space-6)' }}>
              <MoreLink link={block.moreLink} />
            </div>
          )}
        </div>
      </SectionFrame>
    </section>
  )
}
