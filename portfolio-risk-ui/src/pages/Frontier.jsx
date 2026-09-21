import { useCallback, useEffect, useRef, useState } from 'react'
import { cssVar } from '../lib/cssVar'
import HeavyTierPending from '../components/HeavyTierPending'
import InsightBox from '../components/InsightBox'
import useCanvasSize from '../hooks/useCanvasSize'
import { supportsHover } from '../lib/pointer'

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))

const fmt  = v => v != null ? `${(v * 100).toFixed(1)}%` : '-'
const fmtN = (v, d = 2) => v != null ? v.toFixed(d) : '-'

function sharpeColor(pct, alpha = 1) {
  const stops = [
    [0.00, [220, 60,  60]],
    [0.25, [220, 140, 40]],
    [0.50, [200, 190, 60]],
    [0.75, [80,  190, 130]],
    [1.00, [40,  180, 160]],
  ]
  let lo = stops[0], hi = stops[stops.length - 1]
  for (let i = 0; i < stops.length - 1; i++) {
    if (pct >= stops[i][0] && pct <= stops[i + 1][0]) { lo = stops[i]; hi = stops[i + 1]; break }
  }
  const t = (pct - lo[0]) / (hi[0] - lo[0])
  const r = Math.round(lo[1][0] + (hi[1][0] - lo[1][0]) * t)
  const g = Math.round(lo[1][1] + (hi[1][1] - lo[1][1]) * t)
  const b = Math.round(lo[1][2] + (hi[1][2] - lo[1][2]) * t)
  return `rgba(${r},${g},${b},${alpha})`
}

function drawDiamond(ctx, cx, cy, size, fill, strokeColor) {
  ctx.beginPath()
  ctx.moveTo(cx, cy - size); ctx.lineTo(cx + size, cy)
  ctx.lineTo(cx, cy + size); ctx.lineTo(cx - size, cy)
  ctx.closePath()
  ctx.fillStyle = cssVar(fill); ctx.fill()
  ctx.strokeStyle = cssVar(strokeColor); ctx.lineWidth = 1.5; ctx.stroke()
}

function drawCrosshair(ctx, cx, cy, size, color) {
  const gap = size + 4, len = 8
  ctx.strokeStyle = cssVar(color); ctx.lineWidth = 1; ctx.globalAlpha = 0.5
  ctx.beginPath(); ctx.moveTo(cx, cy - gap); ctx.lineTo(cx, cy - gap - len); ctx.stroke()
  ctx.beginPath(); ctx.moveTo(cx, cy + gap); ctx.lineTo(cx, cy + gap + len); ctx.stroke()
  ctx.beginPath(); ctx.moveTo(cx - gap, cy); ctx.lineTo(cx - gap - len, cy); ctx.stroke()
  ctx.beginPath(); ctx.moveTo(cx + gap, cy); ctx.lineTo(cx + gap + len, cy); ctx.stroke()
  ctx.globalAlpha = 1
}

function getInsight(gap, yourSharpe, max_sharpe_sharpe, max_sharpe_weights, tickers, weights) {
  if (gap < 0.05) return {
    tone: 'good', headline: 'Well-positioned',
    body: `Your portfolio is operating near peak efficiency with a Sharpe ratio of ${fmtN(yourSharpe)}. The gap to the theoretical optimum is negligible - your current allocation reflects strong risk-adjusted decision-making.`,
  }
  if (gap < 0.2) return {
    tone: 'good', headline: 'Strong allocation',
    body: `Your Sharpe ratio of ${fmtN(yourSharpe)} is close to the simulated optimum of ${fmtN(max_sharpe_sharpe)}. Minor weight adjustments could close the remaining gap, but your current portfolio already demonstrates disciplined risk management.`,
  }
  const suggestions = max_sharpe_weights
    ? Object.entries(max_sharpe_weights).map(([t, w]) => {
        const i = tickers.indexOf(t)
        const current = i >= 0 ? (weights[i] ?? 0) : 0
        return { ticker: t, current, optimal: w, delta: w - current }
      }).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
    : []
  const top = suggestions[0]
  const direction = top?.delta > 0 ? 'increase' : 'reduce'
  const change = top ? `${direction} ${top.ticker} from ${Math.round(top.current * 100)}% to ${Math.round(top.optimal * 100)}%` : 'rebalance your weights'
  if (gap < 0.5) return {
    tone: 'warning', headline: 'Room to optimise',
    body: `Your portfolio scores ${fmtN(yourSharpe)} against an achievable optimum of ${fmtN(max_sharpe_sharpe)}. The primary lever is to ${change}. This would improve return per unit of risk without necessarily increasing overall volatility.`,
  }
  return {
    tone: 'warning', headline: 'Rebalancing recommended',
    body: `A gap of ${fmtN(gap)} between your Sharpe (${fmtN(yourSharpe)}) and the optimum (${fmtN(max_sharpe_sharpe)}) suggests disproportionate risk relative to returns. The highest-impact adjustment is to ${change}. Refer to the optimal weights below.`,
  }
}

function WeightCard({ title, weights, color, highlightBorderColor, highlight }) {
  if (!weights?.length) return null
  return (
    <div className="card" style={{ padding: '10px 14px', border: highlight ? `1px solid ${highlightBorderColor}` : undefined, overflowY: 'auto' }}>
      <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', marginBottom: 10, color: highlight ? color : 'var(--text-muted)', fontFamily: 'var(--font-primary)' }}>
        {title}
      </div>
      {weights.map(({ ticker, w }) => (
        <div key={ticker} style={{ marginBottom: 7 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
            <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{ticker}</span>
            <span style={{ fontSize: 11, fontWeight: 700, color, fontFamily: 'var(--font-mono)' }}>{Math.round((w ?? 0) * 100)}%</span>
          </div>
          <div style={{ height: 2, background: 'rgba(var(--text-primary-rgb),0.06)', overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${(w ?? 0) * 100}%`, background: color, opacity: 0.8 }}/>
          </div>
        </div>
      ))}
    </div>
  )
}

export default function Frontier({ data, tickers, weights, heavyError }) {
  const canvasRef = useRef(null)
  const animRef   = useRef(null)
  const ptsRef    = useRef([])
  const specRef   = useRef([])
  const [tooltip, setTooltip] = useState(null)

  // Efficient frontier lives in the heavy tier — computed above hook calls
  // so the hooks below always run in the same order every render, whether
  // or not it's arrived yet (conditionally skipping useEffect based on data
  // readiness would violate the Rules of Hooks once this can legitimately
  // be absent on a render, which it couldn't before this endpoint split).
  const ef = data?.efficient_frontier
  const hasFrontier = !!(ef && ef.vols?.length && ef.returns?.length && ef.sharpes?.length)

  const yourVol    = data?.annualised_volatility ?? 0
  const yourReturn = data?.annualised_return     ?? 0
  const yourSharpe = data?.sharpe_ratio          ?? 0

  // draw() is memoized on [data] alone (not the derived ef/hasFrontier/
  // yourVol/yourReturn/yourSharpe below it) because all four are pure
  // functions of data with no other input — recomputed fresh every render,
  // so a data change is the only thing that could ever make a stale
  // closure over them wrong. Matches the original effect's own [data]
  // dependency exactly.
  //
  // Unmount cleanup for the rAF loop lives in its own tiny effect below,
  // not here — useCanvasSize's own effect cleanup (ResizeObserver.
  // disconnect) doesn't know about this component's animation frame, and
  // draw() itself is called repeatedly by useCanvasSize's resize handler
  // without the surrounding effect re-running, so there's no natural place
  // inside draw() for a once-per-unmount cleanup to live.
  const draw = useCallback((ctx, W, H) => {
    if (!hasFrontier) return
    const { vols, returns, sharpes, max_sharpe_vol, max_sharpe_return, max_sharpe_sharpe, min_vol_vol, min_vol_return } = ef

    // Proportional padding + axis font, replacing the fixed PAD/9px this
    // canvas used unconditionally before. Ceiling = the original fixed
    // values (32/32/48/54, 9px), reached at or above the smallest real
    // desktop canvas this renders at — measured directly (not assumed):
    // 1280x800's own canvas is 986px wide, 1366x768's is 373px tall, the
    // narrower dimension of each at the three desktop widths this app
    // targets. Any canvas at or past those resolves to exactly the
    // original numbers, not merely close. Floors keep axis labels legible
    // once .frontier-canvas-wrapper's own compact-viewport min-height
    // (320px) is the binding constraint — that floor pins canvas height to
    // a near-constant ~318px across every phone width (286-818px wide),
    // which is why the font scales with H, not W: H barely moves across
    // the phone range once floored, W moves a lot, but legibility is
    // fundamentally a height-of-canvas question (do two stacked lines of
    // axis text fit) more than a width one.
    const axisFontSize = clamp(H * (9/373), 7, 9)
    const PAD = {
      top:    clamp(H * (32/373), 20, 32),
      right:  clamp(W * (32/986), 14, 32),
      bottom: clamp(H * (48/373), 34, 48),
      left:   clamp(W * (54/986), 32, 54),
    }
    const PW = W - PAD.left - PAD.right
    const PH = H - PAD.top  - PAD.bottom

    // Always include your portfolio in axis range
    const allVols    = [...vols, yourVol]
    const allReturns = [...returns, yourReturn]

    const retRange = Math.max(...allReturns) - Math.min(...allReturns)
    const volRange = Math.max(...allVols)    - Math.min(...allVols)
    // Extra breathing room on the left / bottom: the efficient-frontier
    // markers that matter (min-vol, max-Sharpe, and often the user's own dot)
    // naturally bunch at the low-vol edge, so a tight pad jams them against
    // the axis. Only widens the view — point positions / curve shape unchanged.
    const minV = Math.min(...allVols)    - volRange * 0.11
    const maxV = Math.max(...allVols)    + volRange * 0.06
    const minR = Math.min(...allReturns) - retRange * 0.07
    const maxR = Math.max(...allReturns) + retRange * 0.11
    const minS = Math.min(...sharpes), maxS = Math.max(...sharpes)

    const toX = v => PAD.left + (v - minV) / (maxV - minV) * PW
    const toY = v => PAD.top  + (1 - (v - minR) / (maxR - minR)) * PH

    ctx.fillStyle = cssVar('var(--surface-card)'); ctx.fillRect(0, 0, W, H)

    // Grid
    ctx.strokeStyle = cssVar('rgba(var(--text-primary-rgb),0.04)'); ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) { const y = PAD.top + (i/4)*PH; ctx.beginPath(); ctx.moveTo(PAD.left,y); ctx.lineTo(PAD.left+PW,y); ctx.stroke() }
    for (let i = 0; i <= 5; i++) { const x = PAD.left + (i/5)*PW; ctx.beginPath(); ctx.moveTo(x,PAD.top); ctx.lineTo(x,PAD.top+PH); ctx.stroke() }

    // Axis labels
    ctx.fillStyle = cssVar('rgba(var(--chart-sage-dark-rgb),0.5)'); ctx.font = `${axisFontSize}px ${cssVar('var(--font-mono)')}`; ctx.textAlign = 'center'
    for (let i = 0; i <= 5; i++) { const v = minV+(i/5)*(maxV-minV); ctx.fillText(`${(v*100).toFixed(1)}%`, PAD.left+(i/5)*PW, H-12) }
    ctx.textAlign = 'right'
    for (let i = 0; i <= 4; i++) { const v = minR+(1-i/4)*(maxR-minR); ctx.fillText(`${(v*100).toFixed(0)}%`, PAD.left-8, PAD.top+(i/4)*PH+4) }
    ctx.fillStyle = cssVar('rgba(var(--chart-sage-dark-rgb),0.3)'); ctx.font = `${axisFontSize}px ${cssVar('var(--font-mono)')}`; ctx.textAlign = 'center'
    ctx.fillText('Risk (volatility) →', PAD.left+PW/2, H-2)
    ctx.save(); ctx.translate(12, PAD.top+PH/2); ctx.rotate(-Math.PI/2); ctx.fillText('Return →', 0, 0); ctx.restore()

    // Dynamic CELL size — scales with vol range so dots aren't over-merged
    // with 3 tickers: ~0.0028, with 5 tickers: slightly larger to reduce clutter
    const CELL       = 0.002
    const bucketSize = Math.max(0.002, volRange / 60)

    const grid = {}
    vols.forEach((v, i) => {
      const cx = Math.round(v/CELL), cy = Math.round(returns[i]/CELL), key = `${cx},${cy}`
      if (!grid[key] || sharpes[i] > grid[key].s) grid[key] = { v, r: returns[i], s: sharpes[i] }
    })
    const pts = Object.values(grid)
    ptsRef.current = pts.map(p => ({ ...p, px: toX(p.v), py: toY(p.r), pct: maxS !== minS ? (p.s-minS)/(maxS-minS) : 0.5 }))

    ptsRef.current.forEach(p => {
      const radius = 1.6 + p.pct * 1.2, alpha = 0.5 + p.pct * 0.4
      ctx.beginPath(); ctx.arc(p.px, p.py, radius+0.6, 0, Math.PI*2); ctx.fillStyle = cssVar('var(--canvas-gradient-6)'); ctx.fill()
      ctx.beginPath(); ctx.arc(p.px, p.py, radius, 0, Math.PI*2); ctx.fillStyle = sharpeColor(p.pct, alpha); ctx.fill()
    })

    // Frontier curve with dynamic bucket size + jump filter
    const buckets = {}
    pts.forEach(p => {
      const bk = Math.round(p.v / bucketSize) * bucketSize
      if (!buckets[bk] || p.r > buckets[bk]) buckets[bk] = p.r
    })
    const curve = Object.entries(buckets).map(([v,r]) => ({ v: parseFloat(v), r })).sort((a,b) => a.v-b.v)
    const maxRetJump  = retRange * 0.12
    const smoothCurve = curve.filter((pt, i) => {
      if (i === 0) return true
      return Math.abs(pt.r - curve[i-1].r) < maxRetJump
    })

    if (smoothCurve.length > 2) {
      ctx.beginPath(); ctx.setLineDash([4,5]); ctx.lineWidth = 1; ctx.strokeStyle = cssVar('rgba(var(--text-primary-rgb),0.1)')
      ctx.moveTo(toX(smoothCurve[0].v), toY(smoothCurve[0].r))
      for (let i = 1; i < smoothCurve.length; i++) {
        const prev = smoothCurve[i-1]
        const mx = (toX(prev.v)+toX(smoothCurve[i].v))/2
        const my = (toY(prev.r)+toY(smoothCurve[i].r))/2
        ctx.quadraticCurveTo(toX(prev.v), toY(prev.r), mx, my)
      }
      ctx.stroke(); ctx.setLineDash([])
    }

    // Min vol dot
    if (min_vol_vol != null) {
      ctx.beginPath(); ctx.arc(toX(min_vol_vol), toY(min_vol_return), 5, 0, Math.PI*2)
      ctx.fillStyle = cssVar('rgba(var(--chart-teal-alt-rgb),0.85)'); ctx.fill()
      ctx.strokeStyle = cssVar('var(--canvas-gradient-5)'); ctx.lineWidth = 1.2; ctx.stroke()
    }

    // Your portfolio marker
    const ypX = toX(yourVol), ypY = toY(yourReturn)
    for (let r = 18; r >= 12; r -= 3) {
      ctx.beginPath(); ctx.arc(ypX, ypY, r, 0, Math.PI*2)
      ctx.strokeStyle = cssVar(`rgba(var(--signal-positive-rgb),${0.06+(18-r)*0.015})`); ctx.lineWidth = 1.5; ctx.stroke()
    }
    ctx.beginPath(); ctx.arc(ypX, ypY, 8, 0, Math.PI*2); ctx.fillStyle = cssVar('var(--signal-positive)'); ctx.fill()
    ctx.strokeStyle = cssVar('var(--canvas-gradient-4)'); ctx.lineWidth = 1.5; ctx.stroke()
    ctx.beginPath(); ctx.arc(ypX, ypY, 3, 0, Math.PI*2); ctx.fillStyle = cssVar('var(--text-primary)'); ctx.fill()

    const ypText = 'Your portfolio'; ctx.font = `600 9px ${cssVar('var(--font-mono)')}`
    const ypLw = ctx.measureText(ypText).width + 16
    const ypLx = Math.max(6, Math.min(ypX - ypLw/2, W - ypLw - 6))
    const labelAbove = ypY > 40
    const ypLy = labelAbove ? ypY - 26 : ypY + 32
    ctx.fillStyle = cssVar('var(--canvas-gradient-1)'); ctx.beginPath()
    ctx.rect(ypLx, ypLy-9, ypLw, 16)
    ctx.fill(); ctx.strokeStyle = cssVar('rgba(var(--signal-positive-rgb),0.5)'); ctx.lineWidth = 1; ctx.stroke()
    ctx.fillStyle = cssVar('var(--signal-positive)'); ctx.textAlign = 'center'
    ctx.fillText(ypText, ypLx+ypLw/2, ypLy+3)

    specRef.current = [{ px: ypX, py: ypY, radius: 12, type: 'yours', label: 'Your Portfolio', vol: yourVol, ret: yourReturn, sharpe: yourSharpe, color: 'var(--signal-positive)', glow: 'rgba(var(--signal-positive-rgb),0.19)' }]
    if (max_sharpe_vol != null) {
      specRef.current.push({ px: toX(max_sharpe_vol), py: toY(max_sharpe_return), radius: 12, type: 'optimal', label: 'Optimal Portfolio', vol: max_sharpe_vol, ret: max_sharpe_return, sharpe: max_sharpe_sharpe, color: 'var(--signal-caution)', glow: 'rgba(var(--signal-caution-rgb),0.19)' })
      const opX = toX(max_sharpe_vol), opY = toY(max_sharpe_return)
      cancelAnimationFrame(animRef.current)
      // Same reduced-motion check used in lib/motion/scroll-motion.ts and
      // Hero.jsx — reused as-is rather than a second detection path.
      const prefersReducedMotion = typeof window !== 'undefined' &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches
      const drawOptimalMarker = (r, a) => {
        if (!prefersReducedMotion) {
          ctx.beginPath(); ctx.arc(opX,opY,r,0,Math.PI*2); ctx.strokeStyle=cssVar(`rgba(var(--signal-caution-rgb),${a})`); ctx.lineWidth=2; ctx.stroke()
          ctx.beginPath(); ctx.arc(opX,opY,r*0.6,0,Math.PI*2); ctx.strokeStyle=cssVar(`rgba(var(--signal-caution-rgb),${a*1.5})`); ctx.lineWidth=1; ctx.stroke()
        }
        drawCrosshair(ctx,opX,opY,5,'var(--signal-caution)'); drawDiamond(ctx,opX,opY,6,'var(--signal-caution)','rgba(var(--text-primary-rgb),0.65)')
        ctx.beginPath(); ctx.arc(opX,opY,2,0,Math.PI*2); ctx.fillStyle=cssVar('var(--text-primary)'); ctx.fill()
      }
      if (prefersReducedMotion) {
        // Resting state — the base radius/alpha the pulse oscillates around
        // (frame=0 in the loop below), drawn once with no rings and no rAF loop.
        drawOptimalMarker(10, 0.06)
      } else {
        let frame = 0
        const animate = () => {
          const a = 0.06 + 0.05*Math.sin(frame*0.04), r = 10+Math.sin(frame*0.04)*2
          drawOptimalMarker(r, a)
          frame++; animRef.current = requestAnimationFrame(animate)
        }
        animRef.current = requestAnimationFrame(animate)
      }
    }
  }, [data])

  useCanvasSize(canvasRef, draw)

  // Unmount-only cleanup for the pulsing "Optimal" marker's rAF loop — see
  // draw()'s own comment above for why this can't live inside draw() or
  // rely on useCanvasSize's effect cleanup. draw() itself already cancels
  // and restarts the loop on every call (the cancelAnimationFrame(animRef.
  // current) right before starting a new one, above), so this only ever
  // needs to fire once, when Frontier unmounts entirely.
  useEffect(() => () => cancelAnimationFrame(animRef.current), [])

  if (!data) return null

  // Efficient frontier is heavy-tier — not in the summary /api/analyse-
  // summary renders from, filled in by the background /api/analyse-full
  // call. Re-renders on its own once useAnalysis replaces `data`.
  if (!hasFrontier) {
    return <HeavyTierPending label="Building the efficient frontier..." error={heavyError} />
  }

  const { max_sharpe_sharpe, max_sharpe_weights } = ef
  const gap        = (max_sharpe_sharpe ?? 0) - yourSharpe
  const insight    = getInsight(gap, yourSharpe, max_sharpe_sharpe, max_sharpe_weights, tickers, weights)
  const accentColor = insight.tone === 'good' ? 'var(--signal-positive)' : 'var(--signal-caution)'
  const accentBorderColor = insight.tone === 'good' ? 'rgba(var(--signal-positive-rgb),0.15)' : 'rgba(var(--signal-caution-rgb),0.15)'

  const handleMouseMove = e => {
    const canvas = canvasRef.current; if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const mx = e.clientX - rect.left, my = e.clientY - rect.top
    for (const s of specRef.current) {
      if (Math.sqrt((s.px-mx)**2+(s.py-my)**2) < s.radius) {
        setTooltip({ left: s.px, top: s.py, vol: s.vol, ret: s.ret, sharpe: s.sharpe, pct: s.type==='yours'?0.85:1, label: s.label, color: s.color, glow: s.glow, special: true }); return
      }
    }
    let nearest = null, minDist = 12
    ptsRef.current.forEach(p => { const d = Math.sqrt((p.px-mx)**2+(p.py-my)**2); if (d < minDist) { minDist = d; nearest = p } })
    setTooltip(nearest ? { left: nearest.px, top: nearest.py, vol: nearest.v, ret: nearest.r, sharpe: nearest.s, pct: nearest.pct, label: 'Portfolio', color: null, special: false } : null)
  }

  // Touch path — bound instead of handleMouseMove/handleMouseLeave when
  // supportsHover is false (lib/pointer.js — the same detection MetricTooltip
  // and ReturnHistogram already use, not a new check). Deliberately narrower
  // than the mouse path, not a touch-sized version of it: at a plot that can
  // be ~200px wide on a phone, the deduplicated point cloud packs points well
  // under 44px apart, so a fingertip (~44px contact patch) would sit
  // ambiguously over dozens of candidates and occlude the one meant to be
  // hit — a larger hit radius on the same nearest-point logic just produces
  // confident wrong answers, it doesn't fix the ambiguity. "Your Portfolio"
  // and "Optimal" are what the chart actually communicates, the cloud is
  // context for them, so touch only ever resolves to one of those two, each
  // getting a real 44px tap target — checked in that order, matching the
  // markers' own communicated priority — with no cloud fallback at all: a
  // miss clears the tooltip rather than guessing a nearest point.
  const TOUCH_HIT_RADIUS = 44
  const handleTap = e => {
    const canvas = canvasRef.current; if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const mx = e.clientX - rect.left, my = e.clientY - rect.top
    const yours   = specRef.current.find(s => s.type === 'yours')
    const optimal = specRef.current.find(s => s.type === 'optimal')
    for (const s of [yours, optimal]) {
      if (!s) continue
      if (Math.sqrt((s.px-mx)**2+(s.py-my)**2) < TOUCH_HIT_RADIUS) {
        setTooltip({ left: s.px, top: s.py, vol: s.vol, ret: s.ret, sharpe: s.sharpe, pct: s.type==='yours'?0.85:1, label: s.label, color: s.color, glow: s.glow, special: true })
        return
      }
    }
    setTooltip(null)
  }

  const optimalWeights = max_sharpe_weights ? Object.entries(max_sharpe_weights).map(([t,w]) => ({ ticker: t, w })) : []

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, height: '100%', overflowY: 'auto' }}>

      {/* Why the mix matters — the only InsightBox on this tab, so primary
          by default (same reasoning as Backtest's one box). */}
      <InsightBox
        label="Why the mix matters"
        compact
        priority="primary"
        text="You already picked the stocks. This shows if you picked the right mix. Somewhere on this curve is the version of your portfolio that gets more return for the same risk, or the same return for less. See how close you already are."
      />

      {/* Header — flexWrap on both rows, same as Dashboard's/Backtest's/
          MonteCarlo's legend rows. Traced directly to this being the actual
          source of the tab's opening "403px, 83px past 320" figure (found
          by measurement, not assumed to be the canvas just because it's the
          more prominent element on the tab): the legend's own rightmost
          badge ("Min vol") measured a right edge of exactly 403px, matching
          that number precisely — the canvas itself is fully contained by
          this pass's other fixes and contributes zero overflow. */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0, flexWrap: 'wrap', gap: 8 }}>
        <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)' }}>
          Efficient Frontier
          <span style={{ fontWeight: 'var(--weight-regular)', marginLeft: 8, opacity: 0.6 }}>5,000 simulated portfolios</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, fontSize: 10, color: 'var(--text-muted)', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            {/* Continuous Sharpe heat-map swatch — flagged exception to the single-signal-colour
                rule, see chat: mirrors the frontier scatter's own gradient legend, which needs
                a 5-stop ramp to communicate a continuous metric, not a single signal state. */}
            <div style={{ width: 60, height: 6, background: 'linear-gradient(to right, var(--signal-negative), var(--signal-caution), var(--chart-highlight), var(--signal-positive), var(--chart-teal))' }}/>
            <span style={{ opacity: 0.6 }}>Low → High Sharpe</span>
          </div>
          <span style={{ opacity: 0.2 }}>|</span>
          {[{ color: 'var(--signal-positive)', label: 'Your portfolio' }, { color: 'var(--signal-caution)', label: 'Optimal' }, { color: 'rgba(var(--chart-teal-alt-rgb),0.9)', label: 'Min vol' }].map(l => (
            <span key={l.label} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <div style={{ width: 7, height: 7, background: l.color }}/>{l.label}
            </span>
          ))}
        </div>
      </div>

      {/* Canvas — min-height lives in CSS (.frontier-canvas-wrapper,
          index.css), not inline, for the same inline-always-wins-over-a-
          stylesheet-rule reason as every other chart floor this project has
          needed. Found empirically while verifying the flex-wrap change
          above, not anticipated: the bottom row wrapping to three stacked
          full-width cards at phone width (versus one ~150px-tall row
          before) starves this canvas of the same flex:1 leftover space
          Dashboard's, Backtest's, and MonteCarlo's own charts already hit —
          worse here, measuring literal 0px tall at 320x568, 360x740, and
          852x393 before this floor existed, not just thin. See that class's
          own comment for the floor value and its derivation. */}
      <div className="frontier-canvas-wrapper" style={{ flex: 1, background: 'var(--surface-card)', border: 'var(--border-default)', overflow: 'hidden', position: 'relative' }}>
        <canvas ref={canvasRef}
          {...(supportsHover
            ? { onMouseMove: handleMouseMove, onMouseLeave: () => setTooltip(null) }
            : { onClick: handleTap })}
          style={{ width: '100%', height: '100%', display: 'block', cursor: supportsHover && tooltip ? 'crosshair' : 'default' }} />
        {tooltip && (
          <div style={{
            position: 'absolute',
            left: Math.min(tooltip.left+14, (canvasRef.current?.offsetWidth||999)-170),
            top:  Math.max(tooltip.top-90, 8),
            background: 'var(--surface-elevated)',
            border: `1px solid ${tooltip.color || sharpeColor(tooltip.pct, 0.5)}`,
            padding: '10px 14px', fontSize: 'var(--text-body-sm)', fontFamily: 'var(--font-mono)',
            pointerEvents: 'none', zIndex: 10, minWidth: 148,
          }}>
            <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', letterSpacing: 'var(--tracking-caption)', textTransform: 'uppercase', marginBottom: 8, color: tooltip.color||sharpeColor(tooltip.pct), fontFamily: 'var(--font-primary)', display: 'flex', alignItems: 'center', gap: 5 }}>
              {tooltip.special && <div style={{ width: 6, height: 6, background: tooltip.color, flexShrink: 0 }}/>}
              {tooltip.label}
            </div>
            {[{ k: 'Return', v: fmt(tooltip.ret) }, { k: 'Risk', v: fmt(tooltip.vol) }, { k: 'Sharpe', v: fmtN(tooltip.sharpe), accent: tooltip.color||sharpeColor(tooltip.pct) }].map(row => (
              <div key={row.k} style={{ display: 'flex', justifyContent: 'space-between', gap: 16, marginBottom: 4, paddingBottom: 4, borderBottom: '1px solid rgba(var(--text-primary-rgb),0.04)' }}>
                <span style={{ color: 'rgba(var(--chart-sage-rgb),0.5)' }}>{row.k}</span>
                <span style={{ color: row.accent||'var(--text-primary)', fontWeight: row.accent?700:400 }}>{row.v}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Bottom row — insight + weight cards. flex-wrap, not grid
          1fr 1fr 1fr: see .frontier-bottom-row's own comment in index.css
          for the measured flex-basis and why it's larger than Dashboard/
          MonteCarlo's 88px. */}
      <div className="frontier-bottom-row" style={{ display: 'flex', flexWrap: 'wrap', gap: 8, flexShrink: 0 }}>
        <div style={{
          display: 'flex', flexDirection: 'column', justifyContent: 'center',
          padding: '10px 14px',
          background: 'var(--surface-elevated)',
          border: 'var(--border-faint)',
          gap: 6,
        }}>
          <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)' }}>
            Sharpe ratio
          </div>
          {/* flexWrap + minWidth:0 on the badge — a defensive backstop, not
              the primary fix (that's .frontier-bottom-row's own flex-basis
              below, sized specifically so this never has to engage): the
              badge text is deliberately whiteSpace:'nowrap' (a headline
              like "Rebalancing recommended" measures ~164px including its
              own padding — the actual worst-case content, not a guess: this
              fixture renders that exact string), so without either fix a
              narrow enough panel would force the badge past the card's own
              edge rather than shrinking, since nowrap text has no min-
              content floor below its full width. Wrapping the badge below
              the value if it ever comes to that is a clean degradation. */}
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap' }}>
            <div style={{ fontSize: 'var(--text-heading)', fontWeight: 'var(--weight-semibold)', fontFamily: 'var(--font-mono)', color: accentColor, lineHeight: 1 }}>
              {fmtN(yourSharpe)}
            </div>
            <div style={{
              fontSize: 9, fontWeight: 600, padding: '2px 7px',
              background: insight.tone === 'good' ? 'rgba(var(--signal-positive-rgb),0.1)' : 'var(--signal-caution-wash)',
              color: accentColor, border: `1px solid ${accentBorderColor}`,
              letterSpacing: '0.05em', textTransform: 'uppercase', whiteSpace: 'nowrap',
            }}>
              {insight.headline}
            </div>
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.6 }}>
            {insight.body}
          </div>
        </div>

        <WeightCard
          title="Your weights"
          weights={tickers.map((t, i) => ({ ticker: t, w: weights[i] ?? 0 }))}
          color="var(--signal-positive)"
        />
        <WeightCard
          title={`Optimal - Sharpe ${fmtN(max_sharpe_sharpe)}`}
          weights={optimalWeights}
          color="var(--signal-caution)"
          highlightBorderColor="rgba(var(--signal-caution-rgb),0.19)"
          highlight
        />
      </div>

    </div>
  )
}