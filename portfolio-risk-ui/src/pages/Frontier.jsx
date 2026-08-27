import { useEffect, useRef, useState } from 'react'
import { cssVar } from '../lib/cssVar'

const fmt  = v => v != null ? `${(v * 100).toFixed(1)}%` : '—'
const fmtN = (v, d = 2) => v != null ? v.toFixed(d) : '—'

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
    body: `Your portfolio is operating near peak efficiency with a Sharpe ratio of ${fmtN(yourSharpe)}. The gap to the theoretical optimum is negligible — your current allocation reflects strong risk-adjusted decision-making.`,
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
      <div style={{ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 10, color: highlight ? color : 'var(--text-muted)' }}>
        {title}
      </div>
      {weights.map(({ ticker, w }) => (
        <div key={ticker} style={{ marginBottom: 7 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
            <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{ticker}</span>
            <span style={{ fontSize: 11, fontWeight: 700, color, fontFamily: 'var(--font-mono)' }}>{Math.round((w ?? 0) * 100)}%</span>
          </div>
          <div style={{ height: 2, background: 'rgba(var(--white-rgb),0.06)', borderRadius: 'var(--radius-xs)', overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${(w ?? 0) * 100}%`, background: color, borderRadius: 'var(--radius-xs)', opacity: 0.8 }}/>
          </div>
        </div>
      ))}
    </div>
  )
}

export default function Frontier({ data, tickers, weights }) {
  const canvasRef = useRef(null)
  const animRef   = useRef(null)
  const ptsRef    = useRef([])
  const specRef   = useRef([])
  const [tooltip, setTooltip] = useState(null)

  if (!data) return null
  const ef = data.efficient_frontier
  if (!ef) return null

  const { vols, returns, sharpes, max_sharpe_vol, max_sharpe_return, max_sharpe_sharpe, max_sharpe_weights, min_vol_vol, min_vol_return } = ef
  if (!vols?.length || !returns?.length || !sharpes?.length) return null

  const yourVol    = data.annualised_volatility ?? 0
  const yourReturn = data.annualised_return     ?? 0
  const yourSharpe = data.sharpe_ratio          ?? 0
  const gap        = (max_sharpe_sharpe ?? 0) - yourSharpe
  const insight    = getInsight(gap, yourSharpe, max_sharpe_sharpe, max_sharpe_weights, tickers, weights)
  const accentColor = insight.tone === 'good' ? 'var(--signal-positive)' : 'var(--chart-highlight-alt)'
  const accentBorderColor = insight.tone === 'good' ? 'rgba(var(--signal-positive-rgb),0.15)' : 'rgba(var(--chart-highlight-alt-rgb),0.15)'

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    const W   = canvas.offsetWidth
    const H   = canvas.offsetHeight
    canvas.width  = W * dpr; canvas.height = H * dpr
    canvas.style.width = W + 'px'; canvas.style.height = H + 'px'
    ctx.scale(dpr, dpr)

    const PAD = { top: 32, right: 32, bottom: 48, left: 54 }
    const PW = W - PAD.left - PAD.right
    const PH = H - PAD.top  - PAD.bottom

    // Always include your portfolio in axis range
    const allVols    = [...vols, yourVol]
    const allReturns = [...returns, yourReturn]

    const retRange = Math.max(...allReturns) - Math.min(...allReturns)
    const volRange = Math.max(...allVols)    - Math.min(...allVols)
    const minV = Math.min(...allVols)    - volRange * 0.04
    const maxV = Math.max(...allVols)    + volRange * 0.04
    const minR = Math.min(...allReturns) - retRange * 0.02
    const maxR = Math.max(...allReturns) + retRange * 0.08
    const minS = Math.min(...sharpes), maxS = Math.max(...sharpes)

    const toX = v => PAD.left + (v - minV) / (maxV - minV) * PW
    const toY = v => PAD.top  + (1 - (v - minR) / (maxR - minR)) * PH

    ctx.fillStyle = cssVar('var(--text-on-accent)'); ctx.fillRect(0, 0, W, H)

    // Grid
    ctx.strokeStyle = cssVar('rgba(var(--white-rgb),0.04)'); ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) { const y = PAD.top + (i/4)*PH; ctx.beginPath(); ctx.moveTo(PAD.left,y); ctx.lineTo(PAD.left+PW,y); ctx.stroke() }
    for (let i = 0; i <= 5; i++) { const x = PAD.left + (i/5)*PW; ctx.beginPath(); ctx.moveTo(x,PAD.top); ctx.lineTo(x,PAD.top+PH); ctx.stroke() }

    // Axis labels
    ctx.fillStyle = cssVar('rgba(var(--chart-sage-dark-rgb),0.5)'); ctx.font = '9px monospace'; ctx.textAlign = 'center'
    for (let i = 0; i <= 5; i++) { const v = minV+(i/5)*(maxV-minV); ctx.fillText(`${(v*100).toFixed(1)}%`, PAD.left+(i/5)*PW, H-12) }
    ctx.textAlign = 'right'
    for (let i = 0; i <= 4; i++) { const v = minR+(1-i/4)*(maxR-minR); ctx.fillText(`${(v*100).toFixed(0)}%`, PAD.left-8, PAD.top+(i/4)*PH+4) }
    ctx.fillStyle = cssVar('rgba(var(--chart-sage-dark-rgb),0.3)'); ctx.font = '9px monospace'; ctx.textAlign = 'center'
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
      ctx.beginPath(); ctx.setLineDash([4,5]); ctx.lineWidth = 1; ctx.strokeStyle = cssVar('rgba(var(--white-rgb),0.1)')
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
    ctx.beginPath(); ctx.arc(ypX, ypY, 3, 0, Math.PI*2); ctx.fillStyle = cssVar('var(--white)'); ctx.fill()

    const ypText = 'Your portfolio'; ctx.font = '600 9px monospace'
    const ypLw = ctx.measureText(ypText).width + 16
    const ypLx = Math.max(6, Math.min(ypX - ypLw/2, W - ypLw - 6))
    const labelAbove = ypY > 40
    const ypLy = labelAbove ? ypY - 26 : ypY + 32
    ctx.fillStyle = cssVar('var(--canvas-gradient-1)'); ctx.beginPath()
    if (ctx.roundRect) ctx.roundRect(ypLx, ypLy-9, ypLw, 16, 3); else ctx.rect(ypLx, ypLy-9, ypLw, 16)
    ctx.fill(); ctx.strokeStyle = cssVar('rgba(var(--signal-positive-rgb),0.5)'); ctx.lineWidth = 1; ctx.stroke()
    ctx.fillStyle = cssVar('var(--signal-positive)'); ctx.textAlign = 'center'
    ctx.fillText(ypText, ypLx+ypLw/2, ypLy+3)

    specRef.current = [{ px: ypX, py: ypY, radius: 12, type: 'yours', label: 'Your Portfolio', vol: yourVol, ret: yourReturn, sharpe: yourSharpe, color: 'var(--signal-positive)', glow: 'rgba(var(--signal-positive-rgb),0.19)' }]
    if (max_sharpe_vol != null) {
      specRef.current.push({ px: toX(max_sharpe_vol), py: toY(max_sharpe_return), radius: 12, type: 'optimal', label: 'Optimal Portfolio', vol: max_sharpe_vol, ret: max_sharpe_return, sharpe: max_sharpe_sharpe, color: 'var(--signal-caution)', glow: 'rgba(var(--signal-caution-rgb),0.19)' })
      const opX = toX(max_sharpe_vol), opY = toY(max_sharpe_return)
      let frame = 0; cancelAnimationFrame(animRef.current)
      const animate = () => {
        const a = 0.06 + 0.05*Math.sin(frame*0.04), r = 10+Math.sin(frame*0.04)*2
        ctx.beginPath(); ctx.arc(opX,opY,r,0,Math.PI*2); ctx.strokeStyle=cssVar(`rgba(var(--signal-caution-rgb),${a})`); ctx.lineWidth=2; ctx.stroke()
        ctx.beginPath(); ctx.arc(opX,opY,r*0.6,0,Math.PI*2); ctx.strokeStyle=cssVar(`rgba(var(--signal-caution-rgb),${a*1.5})`); ctx.lineWidth=1; ctx.stroke()
        drawCrosshair(ctx,opX,opY,5,'var(--signal-caution)'); drawDiamond(ctx,opX,opY,6,'var(--signal-caution)','rgba(var(--white-rgb),0.65)')
        ctx.beginPath(); ctx.arc(opX,opY,2,0,Math.PI*2); ctx.fillStyle=cssVar('var(--white)'); ctx.fill()
        frame++; animRef.current = requestAnimationFrame(animate)
      }
      animRef.current = requestAnimationFrame(animate)
    }
    return () => cancelAnimationFrame(animRef.current)
  }, [data])

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

  const optimalWeights = max_sharpe_weights ? Object.entries(max_sharpe_weights).map(([t,w]) => ({ ticker: t, w })) : []

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, height: '100%' }}>

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0 }}>
        <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)' }}>
          Efficient Frontier
          <span style={{ fontWeight: 400, marginLeft: 8, opacity: 0.6 }}>5,000 simulated portfolios</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, fontSize: 10, color: 'var(--text-muted)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 60, height: 6, borderRadius: 'var(--radius-3)', background: 'linear-gradient(to right, var(--signal-negative), var(--signal-caution), var(--chart-highlight), var(--signal-positive), var(--chart-teal))' }}/>
            <span style={{ opacity: 0.6 }}>Low → High Sharpe</span>
          </div>
          <span style={{ opacity: 0.2 }}>|</span>
          {[{ color: 'var(--signal-positive)', label: 'Your portfolio' }, { color: 'var(--signal-caution)', label: 'Optimal' }, { color: 'rgba(var(--chart-teal-alt-rgb),0.9)', label: 'Min vol' }].map(l => (
            <span key={l.label} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <div style={{ width: 7, height: 7, borderRadius: 'var(--radius-full)', background: l.color }}/>{l.label}
            </span>
          ))}
        </div>
      </div>

      {/* Canvas */}
      <div style={{ flex: 1, minHeight: 0, borderRadius: 'var(--radius-10)', background: 'var(--text-on-accent)', border: '1px solid rgba(var(--white-rgb),0.05)', overflow: 'hidden', position: 'relative' }}>
        <canvas ref={canvasRef} onMouseMove={handleMouseMove} onMouseLeave={() => setTooltip(null)}
          style={{ width: '100%', height: '100%', display: 'block', cursor: tooltip ? 'crosshair' : 'default' }} />
        {tooltip && (
          <div style={{
            position: 'absolute',
            left: Math.min(tooltip.left+14, (canvasRef.current?.offsetWidth||999)-170),
            top:  Math.max(tooltip.top-90, 8),
            background: tooltip.special ? 'var(--canvas-gradient-3)' : 'var(--canvas-gradient-2)',
            border: `1px solid ${tooltip.color||sharpeColor(tooltip.pct,0.5)}`,
            borderRadius: 'var(--radius-sm)', padding: '10px 14px', fontSize: 11, fontFamily: 'var(--font-mono)',
            pointerEvents: 'none', zIndex: 10, minWidth: 148,
            boxShadow: tooltip.special ? `0 0 20px ${tooltip.glow},0 4px 20px rgba(var(--black-rgb),0.5)` : '0 4px 16px rgba(var(--black-rgb),0.4)',
          }}>
            <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.07em', textTransform: 'uppercase', marginBottom: 8, color: tooltip.color||sharpeColor(tooltip.pct), display: 'flex', alignItems: 'center', gap: 5 }}>
              {tooltip.special && <div style={{ width: 6, height: 6, borderRadius: 'var(--radius-full)', background: tooltip.color, flexShrink: 0 }}/>}
              {tooltip.label}
            </div>
            {[{ k: 'Return', v: fmt(tooltip.ret) }, { k: 'Risk', v: fmt(tooltip.vol) }, { k: 'Sharpe', v: fmtN(tooltip.sharpe), accent: tooltip.color||sharpeColor(tooltip.pct) }].map(row => (
              <div key={row.k} style={{ display: 'flex', justifyContent: 'space-between', gap: 16, marginBottom: 4, paddingBottom: 4, borderBottom: '1px solid rgba(var(--white-rgb),0.04)' }}>
                <span style={{ color: 'rgba(var(--chart-sage-rgb),0.5)' }}>{row.k}</span>
                <span style={{ color: row.accent||'var(--white)', fontWeight: row.accent?700:400 }}>{row.v}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Bottom row — insight + weight cards */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, flexShrink: 0 }}>
        <div style={{
          display: 'flex', flexDirection: 'column', justifyContent: 'center',
          padding: '10px 14px', borderRadius: 'var(--radius-sm)',
          background: 'rgba(var(--white-rgb),0.03)',
          border: '1px solid rgba(var(--white-rgb),0.07)',
          gap: 6,
        }}>
          <div style={{ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)' }}>
            Sharpe ratio
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
            <div style={{ fontSize: 24, fontWeight: 700, fontFamily: 'var(--font-mono)', color: accentColor, lineHeight: 1 }}>
              {fmtN(yourSharpe)}
            </div>
            <div style={{
              fontSize: 9, fontWeight: 600, padding: '2px 7px', borderRadius: 'var(--radius-20)',
              background: insight.tone === 'good' ? 'rgba(var(--signal-positive-rgb),0.1)' : 'var(--chart-gold-wash)',
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
          title={`Optimal — Sharpe ${fmtN(max_sharpe_sharpe)}`}
          weights={optimalWeights}
          color="var(--signal-caution)"
          highlightBorderColor="rgba(var(--signal-caution-rgb),0.19)"
          highlight
        />
      </div>

    </div>
  )
}