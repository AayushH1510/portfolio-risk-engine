import SectionFrame from './SectionFrame'
import { typeStyle } from './tokens'
import './steps.css'

function Step({ step, isLast }) {
  return (
    <div
      data-reveal
      className="step-row"
      style={{
        padding: '40px 0',
        borderTop: '1px solid var(--color-line-default)',
        borderBottom: isLast ? '1px solid var(--color-line-default)' : undefined,
      }}
    >
      <div className="step-ordinal" style={{ fontFamily: 'var(--font-display)', fontSize: '44px', fontWeight: 200, color: 'var(--color-accent-mint)', lineHeight: 1 }}>
        {step.ordinal}
      </div>
      <div className="step-title">
        <h3 style={{ ...typeStyle('headingM'), margin: '0 0 var(--space-3)', color: 'var(--color-text-primary)' }}>{step.title}</h3>
        <p style={{ ...typeStyle('bodyM'), color: 'var(--color-text-dim)', margin: 0, maxWidth: '460px' }}>{step.body}</p>
      </div>
      <div className="step-annotation" style={{ ...typeStyle('monoCaption'), color: 'var(--color-text-ghost)', lineHeight: 1.9 }}>
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
