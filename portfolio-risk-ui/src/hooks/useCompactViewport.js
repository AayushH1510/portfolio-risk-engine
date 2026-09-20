import { useState, useEffect } from 'react'
import { COMPACT_VIEWPORT_QUERY } from '../lib/breakpoints'

// The "phone, either orientation" live-tracked signal — third independent
// copy of this exact useState+useEffect+matchMedia pattern was about to
// land in Backtest.jsx (Dashboard.jsx and RiskAnalysis.jsx each already
// have their own inline copy), which is exactly the "one value silently
// encoded in N places" shape RESPONSIVE_AUDIT.md has flagged more than once
// as the recurring bug class in this project. Extracted here instead —
// COMPACT_VIEWPORT_QUERY (lib/breakpoints.js) stays the one canonical value,
// this is the one canonical way to subscribe to it from a component.
export default function useCompactViewport() {
  const [isCompactViewport, setIsCompactViewport] = useState(() =>
    typeof window !== 'undefined' && window.matchMedia(COMPACT_VIEWPORT_QUERY).matches
  )
  useEffect(() => {
    const mql = window.matchMedia(COMPACT_VIEWPORT_QUERY)
    const handler = e => setIsCompactViewport(e.matches)
    mql.addEventListener('change', handler)
    return () => mql.removeEventListener('change', handler)
  }, [])
  return isCompactViewport
}
