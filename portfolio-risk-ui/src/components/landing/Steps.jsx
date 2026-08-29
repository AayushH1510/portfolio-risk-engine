import SectionFrame from './SectionFrame'
import { typeStyle } from './tokens'

function Step({ step, isLast }) {
  return (
    <div
      data-reveal
      style={{
        display: 'grid',
        gridTemplateColumns: '64px minmax(0, 1fr) minmax(0, 260px)',
        gap: 'clamp(18px, 2.5vw, 40px)',
        alignItems: 'baseline',
        padding: '40px 0',
        borderTop: '1px solid var(--color-line-default)',
        borderBottom: isLast ? '1px solid var(--color-line-default)' : undefined,
      }}
    >
      <div style={{ fontFamily: 'var(--font-display)', fontSize: '44px', fontWeight: 200, color: 'var(--color-accent-mint)', lineHeight: 1 }}>
        {step.ordinal}
      </div>
      <div>
        <h3 style={{ ...typeStyle('headingM'), margin: '0 0 var(--space-3)', color: 'var(--color-text-primary)' }}>{step.title}</h3>
        <p style={{ ...typeStyle('bodyM'), color: 'var(--color-text-dim)', margin: 0, maxWidth: '460px' }}>{step.body}</p>
      </div>
      <div style={{ ...typeStyle('monoCaption'), color: 'var(--color-text-ghost)', lineHeight: 1.9, borderLeft: '1px solid var(--color-line-default)', paddingLeft: 'var(--space-6)' }}>
        {step.annotation.map((line, i) => (
          <div key={i} style={i === step.annotation.length - 1 ? { color: 'var(--color-accent-mint)' } : undefined}>
            {line}
          </div>
        ))}
      </div>
    </div>
  )
}

export default function Steps({ block }) {
  return (
    <section
      id={block.id}
      style={{ padding: 'var(--layout-sectionPadYBand) var(--layout-gutter)', borderTop: '1px solid var(--color-line-subtle)', background: 'var(--color-bg-raised)' }}
    >
      <SectionFrame index={block.index} label={block.label}>
        <h2 style={{ ...typeStyle('displayM'), margin: '0 0 var(--space-16)', color: 'var(--color-text-primary)', maxWidth: '700px' }}>
          {block.heading}
        </h2>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {block.steps.map((step, i) => (
            <Step key={step.ordinal} step={step} isLast={i === block.steps.length - 1} />
          ))}
        </div>
      </SectionFrame>
    </section>
  )
}
