import { useCallback, useEffect, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'

// Temporary diagnostic overlay for chasing viewport/height discrepancies on
// real devices (Chrome toolbar collapse, dvh vs vh, etc). Gated behind
// ?debug=viewport so it never renders in normal use or in harness
// screenshots (scripts/shots.mjs never appends this param). Self-contained
// and trivially deletable — remove this file and its one mount point in
// main.jsx to fully undo.

function fmtNum(n) {
  return n == null ? 'n/a' : (Math.round(n * 100) / 100)
}

function readSnapshot() {
  const vv = window.visualViewport
  let orientation = 'n/a'
  try { orientation = screen.orientation?.type ?? 'n/a' } catch { orientation = 'n/a' }
  const viewportMeta = document.querySelector('meta[name="viewport"]')
  return {
    innerWidth: window.innerWidth,
    innerHeight: window.innerHeight,
    dpr: window.devicePixelRatio,
    vvWidth: vv ? vv.width : null,
    vvHeight: vv ? vv.height : null,
    screenWidth: screen.width,
    screenHeight: screen.height,
    hoverHover: window.matchMedia('(hover: hover)').matches,
    pointerCoarse: window.matchMedia('(pointer: coarse)').matches,
    reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    orientation,
    viewportMetaContent: viewportMeta ? viewportMeta.getAttribute('content') : 'n/a',
    userAgent: navigator.userAgent,
  }
}

export default function DebugViewportBadge() {
  const location = useLocation()
  const enabled = new URLSearchParams(location.search).get('debug') === 'viewport'

  const [snapshot, setSnapshot] = useState(null)
  const [probeHeights, setProbeHeights] = useState({ vh100: null, dvh100: null })
  const [copied, setCopied] = useState(false)
  const vhProbeRef = useRef(null)
  const dvhProbeRef = useRef(null)

  const measure = useCallback(() => {
    setSnapshot(readSnapshot())
    setProbeHeights({
      vh100: vhProbeRef.current ? vhProbeRef.current.getBoundingClientRect().height : null,
      dvh100: dvhProbeRef.current ? dvhProbeRef.current.getBoundingClientRect().height : null,
    })
  }, [])

  useEffect(() => {
    if (!enabled) return
    measure()
    const onChange = () => measure()
    window.addEventListener('resize', onChange)
    const vv = window.visualViewport
    vv?.addEventListener('resize', onChange)
    vv?.addEventListener('scroll', onChange)
    return () => {
      window.removeEventListener('resize', onChange)
      vv?.removeEventListener('resize', onChange)
      vv?.removeEventListener('scroll', onChange)
    }
  }, [enabled, measure])

  if (!enabled) return null

  const lines = snapshot ? [
    `innerWidth x innerHeight: ${snapshot.innerWidth} x ${snapshot.innerHeight}`,
    `devicePixelRatio: ${snapshot.dpr}`,
    `visualViewport: ${fmtNum(snapshot.vvWidth)} x ${fmtNum(snapshot.vvHeight)}`,
    `screen: ${snapshot.screenWidth} x ${snapshot.screenHeight}`,
    `hover:hover: ${snapshot.hoverHover}`,
    `pointer:coarse: ${snapshot.pointerCoarse}`,
    `prefers-reduced-motion:reduce: ${snapshot.reducedMotion}`,
    `orientation: ${snapshot.orientation}`,
    `viewport meta: ${snapshot.viewportMetaContent}`,
    `100vh / 100dvh / vv.height: ${fmtNum(probeHeights.vh100)} / ${fmtNum(probeHeights.dvh100)} / ${fmtNum(snapshot.vvHeight)}`,
    `userAgent: ${snapshot.userAgent}`,
  ] : []

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(lines.join('\n'))
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard API unavailable/denied — the numbers are still on screen to transcribe by hand.
    }
  }

  return (
    <>
      {/* Zero-footprint height probes — visibility:hidden + zero-width keep
          them out of layout/paint entirely, so they can't affect the very
          numbers being read from other elements on the page. */}
      <div ref={vhProbeRef} style={{ position: 'fixed', top: 0, left: 0, width: 0, height: '100vh', visibility: 'hidden', overflow: 'hidden', pointerEvents: 'none' }} />
      <div ref={dvhProbeRef} style={{ position: 'fixed', top: 0, left: 0, width: 0, height: '100dvh', visibility: 'hidden', overflow: 'hidden', pointerEvents: 'none' }} />

      <div
        style={{
          position: 'fixed', bottom: 'var(--space-3)', left: 'var(--space-3)', zIndex: 999999,
          maxWidth: 320,
          background: 'var(--surface-elevated)',
          border: 'var(--border-default)',
          borderRadius: 'var(--radius)',
          color: 'var(--text-primary)',
          fontFamily: 'var(--font-mono)',
          fontSize: 'var(--text-micro)',
          lineHeight: 1.5,
          padding: 'var(--space-2) var(--space-3)',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-all',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 'var(--space-3)', marginBottom: 'var(--space-1)' }}>
          <span style={{ color: 'var(--text-muted)', letterSpacing: 'var(--tracking-caption)', textTransform: 'uppercase', fontSize: 'var(--text-caption)' }}>debug: viewport</span>
          <button
            onClick={handleCopy}
            style={{
              fontFamily: 'var(--font-mono)', fontSize: 'var(--text-caption)',
              background: 'transparent', color: 'var(--text-primary)',
              border: 'var(--border-default)', borderRadius: 'var(--radius)',
              padding: '2px 8px', cursor: 'pointer',
            }}
          >
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
        {lines.map((line, i) => <div key={i}>{line}</div>)}
      </div>
    </>
  )
}
