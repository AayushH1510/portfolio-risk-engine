import { useEffect, useRef } from 'react'

const FOCUSABLE_SELECTOR = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

// Shared overlay chrome for anything that temporarily covers part or all of
// the viewport above the page — the sidebar drawer (Sidebar.jsx) and the
// Dashboard growth chart's fullscreen expand both need the exact same three
// behaviours (body scroll lock, a Tab focus trap, Escape-to-close with focus
// restored to whatever triggered it) and none of the three is specific to
// being a drawer versus a fullscreen panel. Extracted out of Sidebar.jsx
// (where it originated) so the second caller reuses it instead of
// reimplementing it.
//
// `containerRef` must point at the element whose focusable descendants
// should be trapped — active on mount/activation, it moves focus into that
// element and won't release it until `active` goes false.
export default function useOverlay(active, containerRef, onClose) {
  const previouslyFocusedRef = useRef(null)

  useEffect(() => {
    if (!active) return

    previouslyFocusedRef.current = document.activeElement
    const prevBodyOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'

    const focusables = containerRef.current?.querySelectorAll(FOCUSABLE_SELECTOR)
    ;(focusables?.[0] || containerRef.current)?.focus()

    const handleKeydown = (e) => {
      if (e.key === 'Escape') {
        onClose?.()
        return
      }
      if (e.key !== 'Tab') return
      const list = containerRef.current?.querySelectorAll(FOCUSABLE_SELECTOR)
      if (!list || list.length === 0) return
      const first = list[0]
      const last  = list[list.length - 1]
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault(); last.focus()
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault(); first.focus()
      }
    }
    document.addEventListener('keydown', handleKeydown)

    return () => {
      document.removeEventListener('keydown', handleKeydown)
      document.body.style.overflow = prevBodyOverflow
      previouslyFocusedRef.current?.focus?.()
    }
  }, [active, containerRef, onClose])
}
