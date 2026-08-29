// Small helpers so components consume design/tokens.json (via the generated
// CSS custom properties, see src/landing-theme.css) instead of restating
// font-family/size/weight/etc per element. No hex codes or magic values
// should appear in a landing/* component — everything routes through here
// or a direct var(--...) reference.

export function typeStyle(role) {
  return {
    fontFamily: `var(--type-${role}-font)`,
    fontSize: `var(--type-${role}-size)`,
    fontWeight: `var(--type-${role}-weight)`,
    lineHeight: `var(--type-${role}-leading, normal)`,
    letterSpacing: `var(--type-${role}-tracking, normal)`,
    textTransform: `var(--type-${role}-transform, none)`,
  }
}

const TONE_COLOR = {
  positive: 'var(--color-accent-mint)',
  negative: 'var(--color-signal-negative)',
  warning: 'var(--color-signal-warning)',
  neutral: 'var(--color-text-body)',
}

export function toneColor(tone) {
  return TONE_COLOR[tone] ?? TONE_COLOR.neutral
}

const TONE_DEEP = {
  negative: 'var(--color-signal-negativeDeep)',
  warning: 'var(--color-signal-warningDeep)',
}

/** Gradient origin color for a loss/caution bar — see stressBars visual. */
export function toneDeep(tone) {
  return TONE_DEEP[tone] ?? TONE_DEEP.negative
}
