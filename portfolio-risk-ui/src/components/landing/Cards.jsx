import { useState } from 'react'
import SectionFrame from './SectionFrame'
import { typeStyle, toneColor } from './tokens'

function Card({ card }) {
  const [hovered, setHovered] = useState(false)
  return (
    <div
      data-reveal
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        // Own hairline ring, not a container background — a container bg
        // with gap:1px shows through an empty auto-fit cell. README > Known trap #2.
        boxShadow: 'var(--shadow-hairline)',
        background: hovered ? 'var(--color-bg-inset)' : 'var(--color-bg-panel)',
        padding: '38px 32px',
        transition: `background var(--motion-surfaceTint-duration) var(--motion-surfaceTint-easing)`,
      }}
    >
      <div style={{ ...typeStyle('monoLabel'), color: toneColor(card.tone), marginBottom: 'var(--space-6)' }}>
        {card.ordinal}
      </div>
      <h3 style={{ ...typeStyle('headingS'), margin: '0 0 var(--space-4)', color: 'var(--color-text-primary)' }}>
        {card.title}
      </h3>
      <p style={{ ...typeStyle('bodyS'), color: 'var(--color-text-dim)', margin: 0 }}>{card.body}</p>
    </div>
  )
}

export default function Cards({ block }) {
  return (
    <section id={block.id} style={{ padding: '0 var(--layout-gutter) var(--layout-sectionPadY)' }}>
      <SectionFrame index={block.index} label={block.label}>
        <h2 style={{ ...typeStyle('displayM'), margin: '0 0 var(--space-12)', color: 'var(--color-text-primary)', maxWidth: '800px', textWrap: 'balance' }}>
          {block.heading}
        </h2>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))',
            gap: '1px',
            border: '1px solid var(--color-line-default)',
            borderRadius: 'var(--radius-lg)',
            overflow: 'hidden',
          }}
        >
          {block.cards.map((card) => (
            <Card key={card.ordinal} card={card} />
          ))}
        </div>
      </SectionFrame>
    </section>
  )
}
