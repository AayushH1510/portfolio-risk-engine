import SectionFrame from './SectionFrame'

// Native <details>/<summary> — no JS, no state; the browser's own disclosure
// marker rotates for free, and the page stays fully readable with JS disabled.
export default function Faq({ block }) {
  return (
    <section id={block.id} style={{ padding: '0 var(--layout-gutter) var(--layout-sectionPadY)' }}>
      <SectionFrame index={block.index} label={block.label}>
        <div>
          {block.items.map((item, i) => (
            <details
              key={item.question}
              style={{
                borderBottom: i === block.items.length - 1 ? undefined : '1px solid var(--color-line-default)',
                padding: 'var(--space-7) 0',
              }}
            >
              <summary style={{ cursor: 'pointer', color: 'var(--color-text-ghost)', fontSize: '15px' }}>
                <span style={{ fontFamily: 'var(--font-display)', fontSize: '24px', fontWeight: 300, color: 'var(--color-text-primary)' }}>
                  {item.question}
                </span>
              </summary>
              <p style={{ fontSize: '16.5px', lineHeight: 1.75, color: 'var(--color-text-dim)', fontWeight: 300, margin: '18px 0 0', maxWidth: '640px' }}>
                {item.answer}
              </p>
            </details>
          ))}
        </div>
      </SectionFrame>
    </section>
  )
}
