import { useCallback, useEffect, useState } from 'react'

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
//
// Returns a callback ref — attach it to the <canvas> (`ref={setCanvasNode}`)
// instead of passing `canvasRef` straight to the element. `canvasRef.current`
// still ends up holding the node either way (this writes it), so any
// existing caller code that reads `canvasRef.current` elsewhere (hit-testing
// in mousemove handlers, tooltip positioning) is untouched. What changes is
// *how* the sizing effect below learns the node exists: a plain object ref
// can silently start pointing at a different DOM node (canvasRef.current
// reassigned during a commit) without canvasRef's own identity changing, and
// an effect keyed on [canvasRef, draw] has no way to notice that — it goes
// on observing whatever node it originally attached to, even after that
// node is removed from the DOM. That's exactly what happens when a
// <canvas>'s enclosing subtree gets unmounted and remounted elsewhere
// without the *component* that owns canvasRef unmounting with it — Frontier
// portalling its chart card to document.body on expand is the case this was
// found from: the div/canvas subtree remounts (fresh DOM nodes) but Frontier
// itself, and therefore its useCanvasSize call, does not, so [canvasRef,
// draw] never changes and the old ResizeObserver is left watching a
// detached node forever — canvas.width/height stuck at the browser's 300x150
// default, never resized or redrawn. Routing the node through state (via
// this callback ref) makes the *node itself* the effect's dependency, so a
// swap — remount-driven or otherwise — reliably retriggers setup.
export default function useCanvasSize(canvasRef, draw) {
  const [node, setNode] = useState(null)

  const setCanvasNode = useCallback(el => {
    canvasRef.current = el
    setNode(el)
  }, [canvasRef])

  useEffect(() => {
    if (!node) return

    const resize = () => {
      const W = node.offsetWidth
      const H = node.offsetHeight
      if (!W || !H) return
      const dpr = window.devicePixelRatio || 1
      const bw = Math.round(W * dpr)
      const bh = Math.round(H * dpr)
      node.width = bw
      node.height = bh
      const ctx = node.getContext('2d')
      ctx.scale(bw / W, bh / H)
      draw(ctx, W, H)
    }

    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(node)
    return () => ro.disconnect()
  }, [node, draw])

  return setCanvasNode
}
