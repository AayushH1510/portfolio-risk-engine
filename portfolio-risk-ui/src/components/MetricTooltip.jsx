import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { metricExplanations } from '../content/metricExplanations'

const POPOVER_WIDTH = 200
const GAP = 6

// Wraps a metric's label text with the same dotted-underline hover pattern
// used for tickers in Sidebar.jsx (TickerLabel) — no separate icon, the text
// itself is the trigger. Hover shows the popover on desktop; tap toggles it
// on mobile (where hover doesn't fire), and tapping anywhere else closes it.
//
// The popover renders through a portal into document.body, positioned with
// viewport-fixed coordinates computed from the trigger's own bounding rect.
// It can't just be position:absolute inside the trigger — several pages wrap
// their content in a container with overflow:hidden (App.jsx's <main>), which
// silently clips a relatively-positioned popover whenever the label sits near
// that container's edge.
// Touchscreens synthesize mouseenter/mouseleave around a tap alongside the
// click event — with both hover and click handlers wired to the same
// element, that synthetic pair can toggle `open` shut again right after the
// click opens it. Only bind the hover handlers on devices that actually
// support hover, so touch is left to the click-toggle alone.
const supportsHover = typeof window !== 'undefined' && window.matchMedia('(hover: hover)').matches

export default function MetricTooltip({ metricKey, children }) {
  const [open, setOpen] = useState(false)
  const [coords, setCoords] = useState(null)
  const triggerRef = useRef(null)

  const explanation = metricExplanations[metricKey]

  const updatePosition = () => {
    if (!triggerRef.current) return
    const rect = triggerRef.current.getBoundingClientRect()
    const spaceBelow = window.innerHeight - rect.bottom
    const placement = spaceBelow < 100 ? 'top' : 'bottom'
    let left = rect.left + rect.width / 2 - POPOVER_WIDTH / 2
    left = Math.max(8, Math.min(left, window.innerWidth - POPOVER_WIDTH - 8))
    setCoords({
      left,
      placement,
      y: placement === 'bottom' ? rect.bottom + GAP : window.innerHeight - rect.top + GAP,
    })
  }

  useLayoutEffect(() => {
    if (open) updatePosition()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  useEffect(() => {
    if (!open) return
    const handleOutside = e => {
      if (triggerRef.current && !triggerRef.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('click', handleOutside)
    window.addEventListener('scroll', updatePosition, true)
    window.addEventListener('resize', updatePosition)
    return () => {
      document.removeEventListener('click', handleOutside)
      window.removeEventListener('scroll', updatePosition, true)
      window.removeEventListener('resize', updatePosition)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  if (!explanation) return <>{children}</>

  return (
    <>
      <span
        ref={triggerRef}
        {...(supportsHover ? {
          onMouseEnter: () => setOpen(true),
          onMouseLeave: () => setOpen(false),
        } : {})}
        onClick={e => { e.stopPropagation(); setOpen(o => !o) }}
        style={{
          color: open ? 'var(--accent)' : 'inherit',
          borderBottom: open ? '1.5px solid var(--accent)' : '1.5px dashed rgba(var(--signal-positive-rgb),0.4)',
          paddingBottom: 1,
          cursor: 'help',
          transition: 'color 0.15s, border-color 0.15s',
        }}
      >
        {children}
      </span>

      {open && coords && createPortal(
        <div
          style={{
            position: 'fixed',
            [coords.placement === 'bottom' ? 'top' : 'bottom']: coords.y,
            left: coords.left,
            zIndex: 9999,
            width: POPOVER_WIDTH,
            background: 'var(--card)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            padding: '8px 10px',
            fontSize: 11,
            fontWeight: 400,
            textTransform: 'none',
            letterSpacing: 'normal',
            color: 'var(--text-primary)',
            lineHeight: 1.45,
            boxShadow: '0 10px 28px rgba(var(--black-rgb),0.4)',
          }}
        >
          {explanation}
        </div>,
        document.body
      )}
    </>
  )
}
