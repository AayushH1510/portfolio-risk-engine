import { useLayoutEffect } from 'react'
import { useLocation, useNavigationType } from 'react-router-dom'

// Route changes don't reset window scroll on their own. The landing and
// methodology pages scroll the document itself (useWindowScroll), so a
// client-side link from the bottom of one page lands partway down the next.
//
// Browser scroll restoration is switched off so this decides instead: the
// browser restores before the new route has rendered, and it doesn't scroll
// to a hash that arrives through pushState. Then:
//   - back/forward to an entry the router created: the position it was left at
//   - a URL with a hash: that element, the same place a plain anchor click goes
//   - anything else: the top of the page
// Positions are kept per history entry in sessionStorage, so reload also keeps
// where you were. A position is recorded when leaving a page (a link click,
// back/forward, unload), not on every scroll: the outgoing page's own cleanup
// clamps the window to 0 mid-transition, and a continuous recorder would save
// that 0. /app scrolls inside its own panels, not the window, so nothing is
// recorded or restored there.

const STORAGE_KEY = 'varense:scroll-positions'
const RESTORE_BUDGET_MS = 3000
const RESTORE_SETTLE_MS = 500
const INPUT_EVENTS = ['wheel', 'touchstart', 'keydown', 'mousedown', 'pointerdown']

let positions = null
let currentEntryKey = null
let initialLoadPending = true
let lastInputAt = 0
let lastPopAt = 0

function isAppRoute(pathname) {
  return pathname === '/app' || pathname.startsWith('/app/')
}

function loadPositions() {
  if (positions) return positions
  try {
    positions = JSON.parse(sessionStorage.getItem(STORAGE_KEY)) || {}
  } catch {
    positions = {}
  }
  return positions
}

function entryKey(location) {
  return `${location.pathname}|${location.key}`
}

function recordCurrent() {
  if (currentEntryKey === null) return
  loadPositions()[currentEntryKey] = window.scrollY
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(positions))
  } catch {
    // Storage full or disabled: in-memory positions still cover back/forward.
  }
}

// Module scope, not an effect: must be in place before the first render and
// before the browser's own load-time restore could run.
if (typeof window !== 'undefined') {
  if ('scrollRestoration' in window.history) window.history.scrollRestoration = 'manual'

  const markInput = () => { lastInputAt = performance.now() }
  INPUT_EVENTS.forEach((e) => window.addEventListener(e, markInput, { capture: true, passive: true }))

  // Registered before the router's own popstate listener (this module loads
  // first), so the outgoing page's scroll is read before the route re-renders.
  window.addEventListener('popstate', () => {
    recordCurrent()
    lastPopAt = performance.now()
  })
  window.addEventListener('pagehide', recordCurrent)
  document.addEventListener('click', (e) => {
    if (e.target instanceof Element && e.target.closest('a[href]')) recordCurrent()
  }, true)
}

export default function ScrollManager() {
  const location = useLocation()
  const navType = useNavigationType()
  const onApp = isAppRoute(location.pathname)

  useLayoutEffect(() => {
    currentEntryKey = onApp ? null : entryKey(location)
  }, [onApp, location.pathname, location.key])

  useLayoutEffect(() => {
    if (onApp) return

    // The first entry of a fresh document is only restored on reload or
    // back/forward; a new visit (typed URL, external link) starts at the top.
    const navEntry = performance.getEntriesByType?.('navigation')?.[0]
    const restoreFirst = navEntry?.type === 'reload' || navEntry?.type === 'back_forward'
    const isInitialLoad = initialLoadPending
    if (initialLoadPending) queueMicrotask(() => { initialLoadPending = false })

    // Input after a back/forward started belongs to the user: leave the page
    // where they're scrolling it.
    if (navType === 'POP' && lastPopAt > 0 && lastInputAt >= lastPopAt) return

    // A plain fragment click (e.g. the landing page's in-page nav) also
    // arrives as a POP, but it creates an entry without router state, so it
    // has no saved position and must not be treated as a back navigation.
    const fromRouter = window.history.state !== null
    const saved = (!isInitialLoad || restoreFirst) ? loadPositions()[entryKey(location)] : undefined
    const hash = location.hash
    const getTop = () => {
      if (navType === 'POP' && fromRouter && saved !== undefined) return saved
      if (hash) {
        const el = document.getElementById(decodeURIComponent(hash.slice(1)))
        return el ? el.getBoundingClientRect().top + window.scrollY : 0
      }
      return 0
    }

    // The outgoing page's cleanup (useWindowScroll) can constrain the document
    // after the target is reached and clamp scroll to 0, so the position must
    // hold for RESTORE_SETTLE_MS. Any input since the restore began stops it.
    const startedAt = performance.now()
    let settledAt = null
    let cancelled = false
    const apply = () => {
      if (cancelled) return
      if (lastInputAt >= startedAt) return
      const target = getTop()
      window.scrollTo({ top: target, behavior: 'instant' })
      const now = performance.now()
      if (Math.abs(window.scrollY - target) <= 1) settledAt = settledAt ?? now
      else settledAt = null
      const held = settledAt !== null && now - settledAt >= RESTORE_SETTLE_MS
      if (held || now - startedAt > RESTORE_BUDGET_MS) return
      requestAnimationFrame(apply)
    }
    requestAnimationFrame(apply)
    return () => {
      cancelled = true
    }
  }, [onApp, location.pathname, location.key, location.hash])

  return null
}
