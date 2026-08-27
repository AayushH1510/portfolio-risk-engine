import { useState } from 'react'

export default function SavedPortfolios({
  portfolios,
  onLoad,
  onDelete,
  onSave,
  currentTickers,
  currentWeights,
  currentPeriod,
  currentPortfolioValue,
}) {
  const [expanded, setExpanded]   = useState(true)
  const [saveName, setSaveName]   = useState('')
  const [saving, setSaving]       = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(null)

  const handleSave = () => {
    if (!saveName.trim()) return
    onSave({
      name:           saveName,
      tickers:        currentTickers,
      weights:        currentWeights,
      period:         currentPeriod,
      portfolioValue: currentPortfolioValue,
    })
    setSaveName('')
    setSaving(false)
  }

  const fmt = (iso) => {
    if (!iso) return ''
    const d = new Date(iso)
    if (isNaN(d.getTime())) return ''
    return `${d.getDate()} ${d.toLocaleString('en', { month: 'short' })} ${d.getFullYear()}`
  }

  return (
    <div style={{ marginBottom: 18 }}>

      {/* Header */}
      <div
        onClick={() => setExpanded(v => !v)}
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          cursor: 'pointer', marginBottom: expanded ? 8 : 0,
        }}
      >
        <div style={{
          fontSize: 10, fontWeight: 700, letterSpacing: '0.08em',
          textTransform: 'uppercase', color: 'var(--text-muted)',
        }}>
          Saved portfolios
          {portfolios.length > 0 && (
            <span style={{
              marginLeft: 6, fontSize: 9, fontWeight: 700,
              padding: '1px 5px', borderRadius: 'var(--radius-3)',
              background: 'rgba(var(--signal-positive-rgb),0.15)',
              color: 'var(--accent)',
            }}>
              {portfolios.length}
            </span>
          )}
        </div>
        <span style={{ fontSize: 9, color: 'var(--text-muted)', transition: 'transform 0.15s', display: 'inline-block', transform: expanded ? 'rotate(180deg)' : 'none' }}>▼</span>
      </div>

      {expanded && (
        <>
          {/* Save current button */}
          {!saving ? (
            <button
              onClick={() => setSaving(true)}
              style={{
                width: '100%', padding: '6px 0', borderRadius: 'var(--radius-6)',
                fontSize: 11, fontWeight: 600, letterSpacing: '0.04em',
                background: 'transparent',
                border: '1px dashed rgba(var(--signal-positive-rgb),0.4)',
                color: 'var(--accent)', cursor: 'pointer',
                marginBottom: portfolios.length > 0 ? 8 : 0,
                transition: 'all 0.15s',
              }}
              onMouseEnter={e => e.currentTarget.style.background = 'rgba(var(--signal-positive-rgb),0.08)'}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
            >
              + Save current portfolio
            </button>
          ) : (
            <div style={{ marginBottom: portfolios.length > 0 ? 8 : 0 }}>
              <input
                autoFocus
                value={saveName}
                onChange={e => setSaveName(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') handleSave(); if (e.key === 'Escape') setSaving(false) }}
                placeholder="Portfolio name..."
                style={{ marginBottom: 5, fontSize: 12 }}
              />
              <div style={{ display: 'flex', gap: 5 }}>
                <button
                  onClick={handleSave}
                  disabled={!saveName.trim()}
                  style={{
                    flex: 1, padding: '5px 0', borderRadius: 'var(--radius-5)', fontSize: 10,
                    fontWeight: 700, letterSpacing: '0.05em',
                    background: saveName.trim() ? 'var(--accent)' : 'var(--card-2)',
                    color: saveName.trim() ? 'var(--card)' : 'var(--text-muted)',
                    border: '1px solid var(--border)',
                    cursor: saveName.trim() ? 'pointer' : 'not-allowed',
                  }}
                >
                  Save
                </button>
                <button
                  onClick={() => setSaving(false)}
                  style={{
                    flex: 1, padding: '5px 0', borderRadius: 'var(--radius-5)', fontSize: 10,
                    fontWeight: 600, background: 'var(--card-2)',
                    color: 'var(--text-muted)', border: '1px solid var(--border)',
                    cursor: 'pointer',
                  }}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* Saved list */}
          {portfolios.length === 0 && !saving && (
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontStyle: 'italic', padding: '4px 0' }}>
              No saved portfolios yet
            </div>
          )}

          {portfolios.map(p => (
            <div
              key={p.id}
              style={{
                background: 'var(--card-2)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-6)',
                padding: '8px 10px',
                marginBottom: 5,
                cursor: 'pointer',
                transition: 'border-color 0.12s',
              }}
              onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--accent)'}
              onMouseLeave={e => {
                if (confirmDelete !== p.id) e.currentTarget.style.borderColor = 'var(--border)'
              }}
              onClick={() => onLoad(p)}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {p.name}
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {p.tickers.join(', ')} · {p.period}
                  </div>
                  <div style={{ fontSize: 9, color: 'var(--text-muted)', marginTop: 2 }}>
                    {fmt(p.savedAt)}
                  </div>
                </div>

                {/* Delete button */}
                {confirmDelete === p.id ? (
                  <div style={{ display: 'flex', gap: 4, flexShrink: 0 }} onClick={e => e.stopPropagation()}>
                    <button
                      onClick={() => { onDelete(p.id); setConfirmDelete(null) }}
                      style={{ fontSize: 9, padding: '2px 6px', borderRadius: 'var(--radius-3)', background: 'rgba(var(--signal-negative-rgb),0.2)', color: 'var(--negative)', border: '1px solid rgba(var(--signal-negative-rgb),0.4)', cursor: 'pointer', fontWeight: 700 }}
                    >Delete</button>
                    <button
                      onClick={() => setConfirmDelete(null)}
                      style={{ fontSize: 9, padding: '2px 6px', borderRadius: 'var(--radius-3)', background: 'var(--card-3)', color: 'var(--text-muted)', border: '1px solid var(--border)', cursor: 'pointer' }}
                    >Keep</button>
                  </div>
                ) : (
                  <button
                    onClick={e => { e.stopPropagation(); setConfirmDelete(p.id) }}
                    style={{ fontSize: 11, padding: '0 4px', background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', flexShrink: 0, lineHeight: 1, marginLeft: 6 }}
                    title="Delete"
                  >×</button>
                )}
              </div>
            </div>
          ))}
        </>
      )}
    </div>
  )
}