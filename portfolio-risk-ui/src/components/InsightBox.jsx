export default function InsightBox({ label, text, tone = 'neutral' }) {
  const colors = {
    good:    { border: 'var(--positive)', label: 'var(--positive)' },
    bad:     { border: 'var(--negative)', label: 'var(--negative)' },
    warning: { border: 'var(--warning)',  label: 'var(--warning)'  },
    neutral: { border: 'var(--accent)',   label: 'var(--accent)'   },
  }
  const c = colors[tone] || colors.neutral

  return (
    <div style={{
      borderLeft: `3px solid ${c.border}`,
      background: 'rgba(var(--white-rgb),0.03)',
      borderRadius: 'var(--radius-tab)',
      padding: '10px 14px',
      marginTop: 8,
    }}>
      <div style={{
        fontSize: 10, fontWeight: 700, letterSpacing: '0.08em',
        textTransform: 'uppercase', color: c.label, marginBottom: 5,
      }}>
        {label}
      </div>
      <div
        style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}
        dangerouslySetInnerHTML={{ __html: text }}
      />
    </div>
  )
}