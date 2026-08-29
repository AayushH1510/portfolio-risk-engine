import SectionFrame from './SectionFrame'
import { typeStyle } from './tokens'

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
        </div>
      </SectionFrame>
    </section>
  )
}
