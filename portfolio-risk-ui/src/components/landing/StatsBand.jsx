import { typeStyle } from './tokens'

export default function StatsBand({ block }) {
  return (
    <section style={{ borderTop: '1px solid var(--color-line-subtle)', borderBottom: '1px solid var(--color-line-subtle)', background: 'var(--color-bg-raised)' }}>
      <div
        style={{
          maxWidth: 'var(--layout-maxWidth)',
          margin: '0 auto',
          padding: `0 var(--layout-gutter)`,
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
        }}
      >
        {block.stats.map((stat) => (
          <div key={stat.caption} data-reveal style={{ padding: '46px 0' }}>
            <div style={{ ...typeStyle('statM'), color: 'var(--color-text-primary)' }}>
              {stat.figure}
              {stat.unit && <span style={{ fontSize: '24px', color: 'var(--color-text-ghost)' }}>{stat.unit}</span>}
            </div>
            <div style={{ ...typeStyle('monoLabel'), color: 'var(--color-text-faint)', marginTop: 'var(--space-3)' }}>
              {stat.caption}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
