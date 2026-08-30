import { useEffect } from 'react'

// index.css sets `html, body, #root { height: 100%; overflow: hidden; }`
// for the /app dashboard's fixed-viewport shell (sidebar + internally-
// scrolling panels). Any long-scrolling marketing/editorial page (Landing,
// Methodology, ...) needs real window-level scrolling instead — scroll-
// motion.ts listens on `window.scrollY` / `window`'s scroll event, and even
// pages that don't use scroll-motion still need the page to actually be
// scrollable. Neutralise the constraint only while the page is mounted,
// restore it on unmount so /app is unaffected. Don't "fix" this by making
// the page's own root div scroll internally — window never fires a scroll
// event for a div's internal scroll.
export function useWindowScroll() {
  useEffect(() => {
    const els = [document.documentElement, document.body, document.getElementById('root')].filter(Boolean)
    const prev = els.map((el) => ({ height: el.style.height, overflow: el.style.overflow }))
    for (const el of els) {
      el.style.height = 'auto'
      el.style.overflow = 'visible'
    }
    return () => {
      els.forEach((el, i) => {
        el.style.height = prev[i].height
        el.style.overflow = prev[i].overflow
      })
    }
  }, [])
}
