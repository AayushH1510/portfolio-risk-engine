import { useState, useEffect, useLayoutEffect } from 'react'

const STORAGE_KEY      = 'varense_has_seen_first_result'
const AUTO_DISMISS_MS  = 8000
const ANCHOR_SELECTOR  = '[data-tour="sharpe-card"]'
const POPOVER_WIDTH    = 250

export default function FirstResultCallout() {
  const [visible, setVisible] = useState(false)
  const [rect, setRect]       = useState(null)

  useEffect(() => {
    if (localStorage.getItem(STORAGE_KEY)) return
    // Small delay lets the Dashboard cards finish laying out first.
    const t = setTimeout(() => setVisible(true), 350)
    return () => clearTimeout(t)
  }, [])

  useLayoutEffect(() => {
    if (!visible) return
    const el = document.querySelector(ANCHOR_SELECTOR)
    if (el) setRect(el.getBoundingClientRect())
  }, [visible])

  useEffect(() => {
    if (!visible) return
    const t = setTimeout(dismiss, AUTO_DISMISS_MS)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible])

  const dismiss = () => {
    localStorage.setItem(STORAGE_KEY, 'true')
    setVisible(false)
  }

  if (!visible || !rect) return null

  // Same horizontal clamp as MetricTooltip.jsx — at a narrow single-column
  // mobile layout, a Sharpe card sitting anywhere past ~140px from the
  // right edge would otherwise push this callout off-screen.
  const left = Math.max(8, Math.min(rect.left, window.innerWidth - POPOVER_WIDTH - 8))

  return (
    <div style={{
      position: 'fixed', top: rect.bottom + 10, left,
      zIndex: 999, width: POPOVER_WIDTH,
      background: 'var(--surface-card)', border: '1px solid var(--signal-positive)',
      padding: '12px 14px',
    }}>
      <div style={{
        position: 'absolute', top: -7, left: 24,
        width: 0, height: 0,
        borderLeft: '7px solid transparent',
        borderRight: '7px solid transparent',
        borderBottom: '7px solid var(--signal-positive)',
      }} />
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 10 }}>
        <div style={{ fontSize: 12, color: 'var(--text-primary)', lineHeight: 1.55 }}>
          <strong style={{ color: 'var(--signal-positive)' }}>This is your Sharpe ratio.</strong>{' '}
          Above 1.0 means you're being paid well for the risk you're taking.
        </div>
        <button
          onClick={dismiss}
          aria-label="Dismiss"
          style={{
            background: 'transparent', color: 'var(--text-muted)',
            fontSize: 15, fontWeight: 400, padding: 0, lineHeight: 1, flexShrink: 0,
          }}
        >
          ×
        </button>
      </div>
    </div>
  )
}
