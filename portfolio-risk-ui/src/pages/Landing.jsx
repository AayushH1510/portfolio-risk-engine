import { useRef } from 'react'
import { landingPage } from '../content/landing'
import { BLOCK_RENDERERS } from '../components/landing/registry'
import { useScrollMotion } from '../hooks/useScrollMotion'
import { useWindowScroll } from '../hooks/useWindowScroll'
import Nav from '../components/landing/Nav'
import Footer from '../components/landing/Footer'

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
