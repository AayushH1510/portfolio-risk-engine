import { useEffect } from 'react'

// Shared canvas sizing for RiskGauge/ReturnHistogram/Frontier-style raw-canvas
// components. Per RESPONSIVE_AUDIT §5, none of them redraw on a pure resize —
// only on data change — so a sidebar drag, drawer open/close, or rotation
// leaves them stretched/blurry until the next data update. A ResizeObserver
// on the canvas itself (not a separate wrapper) fixes that: it fires on any
// layout-box change regardless of cause, and setting canvas.width/height
// doesn't itself trigger a layout resize, so there's no feedback loop.
//
// DPR handling rounds the backing-buffer size first (`bw`/`bh`) and derives
// the draw-space scale from that rounded value rather than the raw
// devicePixelRatio. On a fractional DPR (Galaxy A54: 2.8125) `W * dpr` is
// itself fractional and truncates when assigned to canvas.width, leaving the
// buffer a fraction of a px smaller than the CSS box — against hairline
// borders and zero radius that reads as softness. Rounding the buffer size
// first and scaling by bw/W (not dpr) keeps the two in exact agreement.
//
// `draw` should be memoized (useCallback) by the caller with its real data
// dependencies (e.g. [score]) — a new `draw` reference re-runs this effect
// (redrawing at the current size immediately), while the ResizeObserver
// keeps redrawing at whatever the latest `draw` closure is on a pure resize.
export default function useCanvasSize(canvasRef, draw) {
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const resize = () => {
      const W = canvas.offsetWidth
      const H = canvas.offsetHeight
      if (!W || !H) return
      const dpr = window.devicePixelRatio || 1
      const bw = Math.round(W * dpr)
      const bh = Math.round(H * dpr)
      canvas.width = bw
      canvas.height = bh
      const ctx = canvas.getContext('2d')
      ctx.scale(bw / W, bh / H)
      draw(ctx, W, H)
    }

    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(canvas)
    return () => ro.disconnect()
  }, [canvasRef, draw])
}
