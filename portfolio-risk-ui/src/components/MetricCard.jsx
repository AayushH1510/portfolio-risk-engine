export default function MetricCard({ label, value, sub, tone = 'neutral', mono = true, small = false }) {
  const toneColor = {
    good:    'var(--positive)',
    bad:     'var(--negative)',
    warning: 'var(--warning)',
    neutral: 'var(--accent)',
  }[tone] || 'var(--accent)'

  return (
    <div className="card" style={{
      padding: small ? '12px 14px' : '14px 16px',
      borderTop: `2px solid ${toneColor}`,
      display: 'flex',
      flexDirection: 'column',
      gap: 4,
    }}>
      <div style={{
        fontSize: 10, fontWeight: 700, letterSpacing: '0.08em',
        textTransform: 'uppercase', color: 'var(--text-muted)',
      }}>
        {label}
      </div>
      <div style={{
        fontSize: small ? 18 : 22,
        fontWeight: 700,
        color: tone === 'neutral' ? 'var(--text-primary)' : toneColor,
        fontFamily: mono ? 'monospace' : 'inherit',
        letterSpacing: '-0.02em',
        lineHeight: 1.1,
      }}>
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 1 }}>
          {sub}
        </div>
      )}
    </div>
  )
}