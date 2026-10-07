// Shown in place of the raw message when the market-data provider's rate
// limit has been hit (see lib/rateLimit.js). "Try again" calls onRetry, which
// re-runs the same request; nothing retries on its own, so a busy provider
// isn't hit again at the moment it's least able to take it.
export default function BusyNotice({ onRetry }) {
  return (
    <div role="status" style={{
      background: 'var(--signal-caution-wash)', borderLeft: '3px solid var(--signal-caution)',
      padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 8,
      fontFamily: 'var(--font-primary)', textAlign: 'left',
    }}>
      <div style={{ fontSize: 'var(--text-body)', fontWeight: 'var(--weight-medium)', color: 'var(--text-primary)' }}>
        Varense is busy right now.
      </div>
      <div style={{ fontSize: 'var(--text-body-sm)', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
        Lots of people are running analyses at the moment. Please try again in a minute.
      </div>
      <div>
        <button
          type="button"
          onClick={onRetry}
          style={{
            background: 'transparent', border: 'var(--border-default)', borderRadius: 0,
            color: 'var(--text-primary)', fontFamily: 'var(--font-primary)',
            fontSize: 'var(--text-body-sm)', fontWeight: 500, letterSpacing: '0.02em',
            textTransform: 'uppercase', padding: '10px 20px', cursor: 'pointer', marginTop: 4,
          }}
          onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--line-emphasis)'; e.currentTarget.style.background = 'var(--surface-elevated)' }}
          onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--line-hairline)'; e.currentTarget.style.background = 'transparent' }}
        >
          Try again
        </button>
      </div>
    </div>
  )
}
