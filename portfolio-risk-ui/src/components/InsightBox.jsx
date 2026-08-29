export default function InsightBox({ label, text, tone = 'neutral', compact = false }) {
  const colors = {
    good:    { border: 'var(--signal-positive)', label: 'var(--signal-positive)', wash: 'var(--signal-positive-wash)' },
    bad:     { border: 'var(--signal-negative)', label: 'var(--signal-negative)', wash: 'var(--signal-negative-wash)' },
    warning: { border: 'var(--signal-caution)',  label: 'var(--signal-caution)',  wash: 'var(--signal-caution-wash)'  },
    neutral: { border: 'var(--signal-positive)', label: 'var(--signal-positive)', wash: 'var(--signal-positive-wash)' },
  }
  const c = colors[tone] || colors.neutral

  return (
    <div style={{
      borderLeft: `3px solid ${c.border}`,
      background: c.wash,
      padding: compact ? '8px 14px' : '12px 16px',
      marginTop: compact ? 6 : 8,
    }}>
      <div style={{
        fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', letterSpacing: 'var(--tracking-caption)',
        textTransform: 'uppercase', color: c.label, fontFamily: 'var(--font-primary)', marginBottom: compact ? 3 : 5,
      }}>
        {label}
      </div>
      <div
        style={{ fontSize: 'var(--text-body-sm)', fontFamily: 'var(--font-primary)', color: 'var(--text-secondary)', lineHeight: 1.6 }}
        dangerouslySetInnerHTML={{ __html: text }}
      />
    </div>
  )
}
