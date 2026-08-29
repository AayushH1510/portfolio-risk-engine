import SectionFrame from './SectionFrame'
import { typeStyle } from './tokens'

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
      </SectionFrame>
    </section>
  )
}
