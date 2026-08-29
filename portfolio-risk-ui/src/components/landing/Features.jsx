import { typeStyle, toneColor } from './tokens'
import { VISUALS } from './visuals'

function FeatureRow({ feature, isLast }) {
  const Visual = VISUALS[feature.visual]
  const copyOrder = feature.visualSide === 'left' ? 1 : 0
  const visualOrder = feature.visualSide === 'left' ? 0 : 1

  return (
    <div
      data-reveal
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(330px, 1fr))',
        gap: 'clamp(40px, 5vw, 90px)',
        alignItems: 'center',
        padding: 'var(--space-18) 0',
        borderBottom: isLast ? undefined : '1px solid var(--color-line-subtle)',
      }}
    >
      <div style={{ order: copyOrder }}>
        <div style={{ ...typeStyle('monoLabel'), color: toneColor(feature.kickerTone), marginBottom: 'var(--space-5)' }}>
          {feature.kicker}
        </div>
        <h3 style={{ ...typeStyle('displayS'), margin: '0 0 var(--space-5)', color: 'var(--color-text-primary)' }}>
          {feature.heading}
        </h3>
        <p style={{ ...typeStyle('bodyL'), color: 'var(--color-text-dim)', margin: '0 0 var(--space-6)', maxWidth: '500px' }}>
          {feature.body}
        </p>
        {feature.tags && (
          <div style={{ display: 'flex', gap: 'var(--space-9)', ...typeStyle('monoCaption'), color: 'var(--color-text-ghost)', flexWrap: 'wrap' }}>
            {feature.tags.map((tag) => (
              <span key={tag}>{tag}</span>
            ))}
          </div>
        )}
      </div>

      <div
        data-parallax="0.06"
        style={{
          order: visualOrder,
          border: '1px solid var(--color-line-default)',
          borderRadius: 'var(--radius-lg)',
          background: 'var(--color-bg-panel)',
          padding: 'var(--space-8)',
          boxShadow: 'var(--shadow-card)',
        }}
      >
        <Visual />
      </div>
    </div>
  )
}

export default function Features({ block }) {
  return (
    <section id={block.id} style={{ padding: 'var(--layout-sectionPadY) var(--layout-gutter) var(--space-9)' }}>
      <div style={{ maxWidth: 'var(--layout-maxWidth)', margin: '0 auto' }}>
        <div
          style={{
            ...typeStyle('monoSection'),
            color: 'var(--color-text-ghost)',
            paddingTop: 'var(--space-4)',
            borderTop: '1px solid var(--color-line-default)',
            marginBottom: 'var(--space-9)',
          }}
        >
          {block.index} — {block.label}
        </div>
        {block.features.map((feature, i) => (
          <FeatureRow key={feature.heading} feature={feature} isLast={i === block.features.length - 1} />
        ))}
      </div>
    </section>
  )
}
