import { typeStyle } from './tokens'
import ActionButton from './ActionButton'

export default function Cta({ block }) {
  return (
    <section style={{ position: 'relative', padding: 'var(--space-28) var(--layout-gutter)', borderTop: '1px solid var(--color-line-subtle)', overflow: 'hidden', background: 'var(--color-bg-raised)' }}>
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background: 'radial-gradient(70% 130% at 50% 100%, var(--color-alpha-mintGlow) 0%, rgba(12,9,8,0) 62%)',
        }}
      />
      <div style={{ position: 'relative', maxWidth: '800px', margin: '0 auto', textAlign: 'center' }}>
        <h2 style={{ ...typeStyle('displayL'), margin: '0 0 var(--space-7)', color: 'var(--color-text-primary)', textWrap: 'balance' }}>
          {block.heading}
        </h2>
        <p style={{ fontSize: '18.5px', lineHeight: 1.7, color: 'var(--color-text-muted)', fontWeight: 300, margin: '0 0 var(--space-9)' }}>
          {block.lead}
        </p>
        <div style={{ display: 'flex', gap: 'var(--space-4)', justifyContent: 'center', flexWrap: 'wrap' }}>
          {block.actions.map((link) => (
            <ActionButton key={link.href} link={link} padding="17px 34px" />
          ))}
        </div>
      </div>
    </section>
  )
}
