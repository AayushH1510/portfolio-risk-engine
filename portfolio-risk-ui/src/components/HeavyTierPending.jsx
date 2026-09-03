// Scoped loading/error state for a tab whose data comes from the background
// /api/analyse-full call, not the fast /api/analyse-summary response the
// Dashboard renders from immediately. Used by MonteCarlo, Frontier, and
// Backtest — each still renders instantly once heavy data arrives, no
// manual refresh needed, since it's just a normal prop update from
// useAnalysis once setData(fullResponse) runs.
export default function HeavyTierPending({ label, error }) {
  if (error) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%',
        flexDirection: 'column', gap: 8, textAlign: 'center', padding: 24,
      }}>
        <div style={{ fontSize: 13, color: 'var(--signal-negative)' }}>{error}</div>
      </div>
    )
  }

  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%',
      flexDirection: 'column', gap: 12, textAlign: 'center', padding: 24,
    }}>
      <svg className="spin" width="28" height="28" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="10" stroke="var(--signal-positive)" strokeWidth="3" strokeDasharray="40 20" />
      </svg>
      <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>{label}</div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', opacity: 0.7 }}>
        Running the full simulation, this can take up to a minute.
      </div>
    </div>
  )
}
