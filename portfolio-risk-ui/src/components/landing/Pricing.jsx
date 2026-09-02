import { useState } from 'react'
import SectionFrame from './SectionFrame'
import { typeStyle } from './tokens'
import ActionButton from './ActionButton'

function TierAction({ link }) {
  const [hovered, setHovered] = useState(false)
  const primary = link.variant === 'primary'
  return (
    <a
      href={link.href}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        marginTop: 'auto',
        textAlign: 'center',
        fontFamily: 'var(--font-mono)',
        fontSize: '13px',
        letterSpacing: '0.06em',
        padding: '14px',
        borderRadius: 'var(--radius-sm)',
        transition: `background var(--motion-surfaceTint-duration) var(--motion-surfaceTint-easing), border-color var(--motion-surfaceTint-duration) var(--motion-surfaceTint-easing), color var(--motion-surfaceTint-duration) var(--motion-surfaceTint-easing)`,
        ...(primary
          ? { background: hovered ? 'var(--color-accent-mintHover)' : 'var(--color-accent-mint)', color: 'var(--color-bg-base)', fontWeight: 500 }
          : { border: `1px solid ${hovered ? 'var(--color-line-hover)' : 'var(--color-line-interactive)'}`, color: hovered ? 'var(--color-text-primary)' : 'var(--color-text-secondary)' }),
      }}
    >
      {link.label}
    </a>
  )
}

function TierCard({ tier }) {
  const featured = tier.featured
  return (
    <div
      data-reveal
      style={{
        position: 'relative',
        border: `1px solid ${featured ? 'var(--color-accent-mintBorder)' : 'var(--color-line-default)'}`,
        borderRadius: 'var(--radius-xl)',
        background: featured
          ? 'linear-gradient(180deg, color-mix(in srgb, var(--color-bg-panel), white 8%), var(--color-bg-panel))'
          : 'var(--color-bg-panel)',
        boxShadow: featured ? 'var(--shadow-tier)' : undefined,
        padding: '40px 34px',
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-6)',
      }}
    >
      {featured && (
        <div
          style={{
            position: 'absolute',
            top: '-1px',
            left: '34px',
            right: '34px',
            height: '1px',
            background: 'linear-gradient(90deg, transparent, var(--color-accent-mint), transparent)',
          }}
        />
      )}
      <div>
        <div
          style={{
            ...typeStyle('monoLabel'),
            color: featured ? 'var(--color-accent-mint)' : 'var(--color-text-faint)',
            marginBottom: 'var(--space-6)',
          }}
        >
          {tier.name}
        </div>
        <div style={{ ...typeStyle('statL'), color: 'var(--color-text-primary)' }}>
          {tier.price}
          {tier.period && <span style={{ fontFamily: 'var(--font-mono)', fontSize: '15px', color: 'var(--color-text-ghost)' }}>{tier.period}</span>}
        </div>
        <div style={{ fontSize: '14.5px', color: 'var(--color-text-ghost)', marginTop: 'var(--space-3)' }}>{tier.summary}</div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)', fontSize: '15px', color: 'var(--color-text-dim)', fontWeight: 300 }}>
        {tier.features.map((f) => (
          <div key={f}>{f}</div>
        ))}
      </div>
      <TierAction link={tier.action} />
    </div>
  )
}

// Single centered message in place of the tier grid — e.g. "free during
// beta." See content.ts > PricingBlock.notice. The two-column tier layout
// doesn't make sense for one message, so this doesn't reuse SectionFrame's
// label column; it's a simpler centered block, same section padding/rhythm.
function PricingNotice({ block }) {
  const { notice } = block
  return (
    <section id={block.id} style={{ padding: 'var(--layout-sectionPadY) var(--layout-gutter)' }}>
      <div style={{ maxWidth: '640px', margin: '0 auto', textAlign: 'center' }}>
        <div
          style={{
            ...typeStyle('monoSection'),
            color: 'var(--color-text-ghost)',
            marginBottom: 'var(--space-8)',
          }}
        >
          {block.index} - {block.label}
        </div>
        <h2 data-reveal style={{ ...typeStyle('displayM'), margin: '0 0 var(--space-6)', color: 'var(--color-text-primary)' }}>
          {block.heading}
        </h2>
        <p style={{ ...typeStyle('lead'), color: 'var(--color-text-muted)', margin: '0 0 var(--space-8)' }}>
          {notice.lead}
        </p>
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <ActionButton link={notice.action} />
        </div>
        {notice.note && (
          <div style={{ ...typeStyle('monoCaption'), color: 'var(--color-text-ghost)', marginTop: 'var(--space-6)' }}>
            {notice.note}
          </div>
        )}
      </div>
    </section>
  )
}

export default function Pricing({ block }) {
  if (block.notice) return <PricingNotice block={block} />

  return (
    <section id={block.id} style={{ padding: 'var(--layout-sectionPadY) var(--layout-gutter)' }}>
      <SectionFrame index={block.index} label={block.label}>
        <h2 style={{ ...typeStyle('displayM'), margin: '0 0 var(--space-16)', color: 'var(--color-text-primary)', maxWidth: '640px' }}>
          {block.heading}
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(255px, 1fr))', gap: 'var(--space-5)' }}>
          {block.tiers.map((tier) => (
            <TierCard key={tier.name} tier={tier} />
          ))}
        </div>
      </SectionFrame>
    </section>
  )
}
