import { useState } from 'react'
import { COMPACT_VIEWPORT_QUERY } from '../lib/breakpoints'

// `priority` is opt-in and additive — every existing call site (Dashboard,
// Monte Carlo, Efficient Frontier, Valuation, Backtest, Compare) omits it
// entirely and keeps today's always-expanded, no-affordance behaviour
// exactly as before. Only a tab dense enough to need triage marks its boxes:
// 'secondary' collapses by default on a compact viewport (still toggleable
// everywhere); 'primary' gets the identical toggle affordance so a group of
// boxes reads as one consistent control, not "two have a button and one is
// just broken," but always DEFAULTS to expanded regardless of viewport —
// it's the one the reader shouldn't have to open. This is the generic
// mechanism RiskAnalysis needed rather than RiskAnalysis.jsx hardcoding
// which of its three boxes collapses — other dense tabs can reuse it the
// same way.
export default function InsightBox({ label, text, tone = 'neutral', compact = false, priority }) {
  const colors = {
    good:    { border: 'var(--signal-positive)', label: 'var(--signal-positive)', wash: 'var(--signal-positive-wash)' },
    bad:     { border: 'var(--signal-negative)', label: 'var(--signal-negative)', wash: 'var(--signal-negative-wash)' },
    warning: { border: 'var(--signal-caution)',  label: 'var(--signal-caution)',  wash: 'var(--signal-caution-wash)'  },
    neutral: { border: 'var(--signal-positive)', label: 'var(--signal-positive)', wash: 'var(--signal-positive-wash)' },
  }
  const c = colors[tone] || colors.neutral

  const collapsible = priority === 'primary' || priority === 'secondary'
  // Seeded once at mount from the current viewport, not tracked live —
  // "session-only" (no localStorage) is about not remembering a collapse
  // across visits, but a live matchMedia listener here would also fight a
  // manual toggle every time the box's own viewport crosses 768px mid-
  // session (e.g. a rotation), which is worse UX than just picking a
  // sensible default once. COMPACT_VIEWPORT_QUERY (lib/breakpoints.js) is
  // the same dual-axis "phone, either orientation" signal every other
  // phone-gated rule in the app uses — see RESPONSIVE_AUDIT.md's standing
  // decision on why a width-only check would miss a landscape phone.
  // Only 'secondary' ever seeds collapsed — 'primary' is collapsible (same
  // toggle, same tap target) but always starts expanded, viewport included.
  const [collapsed, setCollapsed] = useState(
    () => priority === 'secondary' && typeof window !== 'undefined' && window.matchMedia(COMPACT_VIEWPORT_QUERY).matches
  )

  return (
    <div style={{
      borderLeft: `3px solid ${c.border}`,
      background: c.wash,
      padding: compact ? '8px 14px' : '12px 16px',
      marginTop: compact ? 6 : 8,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginBottom: compact ? 3 : 5 }}>
        <div style={{
          fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', letterSpacing: 'var(--tracking-caption)',
          textTransform: 'uppercase', color: c.label, fontFamily: 'var(--font-primary)',
        }}>
          {label}
        </div>
        {collapsible && (
          // Mono glyph in a hairline box, 44x44 tap target via the same
          // ::before{inset:-11px} pattern as Dashboard's .chart-expand-btn
          // (index.css) — no new affordance vocabulary, just that one reused.
          <button
            type="button"
            className="insight-toggle-btn"
            onClick={() => setCollapsed(v => !v)}
            aria-expanded={!collapsed}
            aria-label={collapsed ? 'Expand explanation' : 'Collapse explanation'}
            style={{ color: c.label, borderColor: `rgba(var(--text-primary-rgb),0.15)` }}
          >
            {collapsed ? '▸' : '▾'}
          </button>
        )}
      </div>
      {!collapsed && (
        <div
          style={{ fontSize: 'var(--text-body-sm)', fontFamily: 'var(--font-primary)', color: 'var(--text-secondary)', lineHeight: 1.6 }}
          dangerouslySetInnerHTML={{ __html: text }}
        />
      )}
    </div>
  )
}
