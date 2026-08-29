import { useState } from 'react'
import { typeStyle } from './tokens'

// Primary (mint fill) / secondary (outlined) link button — used by Hero and
// the closing CTA. See README > Interactions > Hover states.
export default function ActionButton({ link, padding = '16px 30px' }) {
  const [hovered, setHovered] = useState(false)
  const primary = link.variant === 'primary'
  return (
    <a
      href={link.href}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        ...typeStyle('monoAction'),
        padding,
        borderRadius: 'var(--radius-sm)',
        transition: `transform var(--motion-hoverLift-duration) var(--motion-hoverLift-easing), box-shadow var(--motion-hoverLift-duration) var(--motion-hoverLift-easing), border-color var(--motion-hoverLift-duration) var(--motion-hoverLift-easing), color var(--motion-hoverLift-duration) var(--motion-hoverLift-easing)`,
        transform: primary && hovered ? 'translateY(var(--motion-hoverLift-translateY))' : 'none',
        ...(primary
          ? {
              background: 'var(--color-accent-mint)',
              color: 'var(--color-bg-base)',
              boxShadow: hovered ? 'var(--shadow-actionHover)' : 'var(--shadow-action)',
            }
          : {
              border: `1px solid ${hovered ? 'var(--color-line-hover)' : 'var(--color-line-interactive)'}`,
              color: hovered ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
            }),
      }}
    >
      {link.label}
    </a>
  )
}
