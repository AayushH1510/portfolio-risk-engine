import { useEffect, useRef } from 'react'
import { landingPage } from '../content/landing'
import { BLOCK_RENDERERS } from '../components/landing/registry'
import { useScrollMotion } from '../hooks/useScrollMotion'
import Nav from '../components/landing/Nav'
import Footer from '../components/landing/Footer'

// index.css sets `html, body, #root { height: 100%; overflow: hidden; }`
// for the /app dashboard's fixed-viewport shell (sidebar + internally-
// scrolling panels). scroll-motion.ts is framework-agnostic and listens on
// `window.scrollY` / `window`'s scroll event — it needs real window-level
// scrolling, not a clipped, non-scrolling body. Neutralise the constraint
// only while this page is mounted, restore it on unmount so /app is
// unaffected. Don't "fix" this by making Landing's own root div scroll
// internally (the way LandingLayout.jsx's simpler pages do) — window never
// fires a scroll event for a div's internal scroll, and parallax/reveal
// would silently stop working.
function useWindowScroll() {
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

export default function Landing() {
  const { brand, nav, navAction, blocks, footer, motionIntensity } = landingPage
  const onScrollRef = useRef(null)

  useWindowScroll()

  // Must run after the blocks mount — initScrollMotion queries
  // [data-parallax]/[data-reveal] at call time, and a useEffect here runs
  // after children exist. See README > Rule 2.
  useScrollMotion({
    intensity: motionIntensity / 100,
    onScroll: (scrollY) => onScrollRef.current?.(scrollY),
  })

  return (
    <div
      style={{
        position: 'relative',
        background: 'var(--color-bg-base)',
        color: 'var(--color-text-body)',
        fontFamily: 'var(--font-body)',
        fontWeight: 300,
        overflowX: 'hidden',
      }}
    >
      <Nav brand={brand} nav={nav} navAction={navAction} onScrollRef={onScrollRef} />

      {blocks.map((block, i) => {
        const Renderer = BLOCK_RENDERERS[block.type]
        if (!Renderer) {
          if (import.meta.env.DEV) {
            throw new Error(`No renderer registered for block type "${block.type}"`)
          }
          console.error(`No renderer registered for block type "${block.type}"`)
          return null
        }
        return <Renderer key={block.id ?? `${block.type}-${i}`} block={block} />
      })}

      <Footer brand={brand} footer={footer} />
    </div>
  )
}
