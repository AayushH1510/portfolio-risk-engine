export default function MetricCard({ label, value, sub, tone = 'neutral', mono = true, small = false }) {
  const toneColor = {
    good:    'var(--signal-positive)',
    bad:     'var(--signal-negative)',
    warning: 'var(--signal-caution)',
    neutral: 'var(--text-primary)',
  }[tone] || 'var(--text-primary)'

  // DESIGN.md Metric Card spec: only warning/negative states get a signal-coloured
  // top border — good/neutral keep the default light hairline-top treatment.
  const topBorder = (tone === 'bad' || tone === 'warning') ? `2px solid ${toneColor}` : undefined

  return (
    <div className="card" style={{
      padding: small ? '12px 14px' : '14px 16px',
      ...(topBorder ? { borderTop: topBorder } : {}),
      display: 'flex',
      flexDirection: 'column',
      gap: 4,
    }}>
      <div style={{
        fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', letterSpacing: 'var(--tracking-caption)',
        textTransform: 'uppercase', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)',
      }}>
        {label}
      </div>
      <div style={{
        fontSize: small ? 18 : 22,
        fontWeight: 'var(--weight-semibold)',
        color: toneColor,
        fontFamily: mono ? 'var(--font-mono)' : 'var(--font-primary)',
        letterSpacing: '-0.02em',
        lineHeight: 1.1,
      }}>
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 'var(--text-micro)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)', marginTop: 1 }}>
          {sub}
        </div>
      )}
    </div>
  )
}
