import { useEffect, useRef } from 'react'
import { cssVar } from '../lib/cssVar'

export default function ReturnHistogram({ portfolioReturns, varPct, cvarPct, confidence }) {
  const canvasRef = useRef(null)

  if (!portfolioReturns?.length) return null

  const returns = portfolioReturns
  const mean    = returns.reduce((a, b) => a + b, 0) / returns.length
  const std     = Math.sqrt(returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length)
  const minR    = Math.min(...returns)
  const maxR    = Math.max(...returns)
  const posPct  = returns.filter(r => r > 0).length / returns.length
  const fmt     = v => `${(v * 100).toFixed(2)}%`
  const fmtS    = v => `${(v * 100).toFixed(1)}%`

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W   = canvas.offsetWidth
    const H   = canvas.offsetHeight
    canvas.width  = W * window.devicePixelRatio
    canvas.height = H * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    const PAD = { top: 16, right: 16, bottom: 36, left: 36 }
    const PW  = W - PAD.left - PAD.right
    const PH  = H - PAD.top  - PAD.bottom

    const N_BINS = 38
    const binW   = (maxR - minR) / N_BINS
    const bins   = Array(N_BINS).fill(0)
    returns.forEach(r => {
      const idx = Math.min(Math.floor((r - minR) / binW), N_BINS - 1)
      bins[idx]++
    })
    const maxCount = Math.max(...bins)

    const toX = v => PAD.left + ((v - minR) / (maxR - minR)) * PW
    const toY = v => PAD.top  + (1 - v / maxCount) * PH

    // Clean dark background
    ctx.fillStyle = cssVar('var(--text-on-accent)')
    ctx.fillRect(0, 0, W, H)

    // Subtle grid lines
    ctx.strokeStyle = cssVar('rgba(var(--white-rgb),0.04)')
    ctx.lineWidth   = 1
    for (let i = 1; i <= 4; i++) {
      const y = PAD.top + (i / 4) * PH
      ctx.beginPath()
      ctx.moveTo(PAD.left, y)
      ctx.lineTo(PAD.left + PW, y)
      ctx.stroke()
    }

    // Zero line
    const zeroX = toX(0)
    ctx.strokeStyle = cssVar('rgba(var(--white-rgb),0.12)')
    ctx.lineWidth   = 1
    ctx.setLineDash([3, 4])
    ctx.beginPath(); ctx.moveTo(zeroX, PAD.top); ctx.lineTo(zeroX, PAD.top + PH); ctx.stroke()
    ctx.setLineDash([])

    // Tail shading (left of VaR)
    const varX = toX(varPct)
    ctx.fillStyle = cssVar('rgba(var(--signal-negative-rgb),0.05)')
    ctx.fillRect(PAD.left, PAD.top, varX - PAD.left, PH)

    // VaR line
    ctx.strokeStyle = cssVar('rgba(var(--signal-negative-rgb),0.6)')
    ctx.lineWidth   = 1.5
    ctx.setLineDash([5, 4])
    ctx.beginPath(); ctx.moveTo(varX, PAD.top); ctx.lineTo(varX, PAD.top + PH); ctx.stroke()
    ctx.setLineDash([])

    // CVaR line
    const cvarX = toX(cvarPct)
    ctx.strokeStyle = cssVar('rgba(var(--signal-caution-rgb),0.55)')
    ctx.lineWidth   = 1.5
    ctx.setLineDash([5, 4])
    ctx.beginPath(); ctx.moveTo(cvarX, PAD.top); ctx.lineTo(cvarX, PAD.top + PH); ctx.stroke()
    ctx.setLineDash([])

    // Bars
    const bW = PW / N_BINS
    bins.forEach((count, i) => {
      if (count === 0) return
      const binMid = minR + (i + 0.5) * binW
      const x      = PAD.left + (i / N_BINS) * PW
      const barH   = (count / maxCount) * PH
      const y      = PAD.top + PH - barH

      let baseColor
      if (binMid < varPct)  baseColor = [224, 92,  92]
      else if (binMid < 0)  baseColor = [200, 150, 60]
      else                   baseColor = [82,  183, 136]

      const [r, g, b] = baseColor
      const alpha = binMid < varPct ? 0.7 : 0.6

      // Bar body
      ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`
      ctx.beginPath()
      ctx.roundRect?.(x + 1, y, bW - 2, barH, [2, 2, 0, 0])
      ctx.fill()

      // Top cap
      ctx.fillStyle = `rgba(${r},${g},${b},1)`
      ctx.fillRect(x + 1, y, bW - 2, 1.5)
    })

    // Normal distribution curve
    const gaussian = x => {
      return Math.exp(-0.5 * ((x - mean) / std) ** 2) / (std * Math.sqrt(2 * Math.PI))
    }
    const scaleFactor = maxCount / (gaussian(mean) * returns.length * binW)

    ctx.beginPath()
    ctx.strokeStyle = cssVar('rgba(var(--white-rgb),0.18)')
    ctx.lineWidth   = 1.5
    for (let i = 0; i <= PW; i++) {
      const rv = minR + (i / PW) * (maxR - minR)
      const gv = gaussian(rv) * returns.length * binW * scaleFactor
      const x  = PAD.left + i
      const y  = PAD.top + PH - (gv / maxCount) * PH
      if (i === 0) ctx.moveTo(x, y)
      else         ctx.lineTo(x, y)
    }
    ctx.stroke()

    // X-axis labels
    ctx.fillStyle = cssVar('rgba(var(--chart-sage-dark-rgb),0.55)')
    ctx.font      = '9px monospace'
    ctx.textAlign = 'center'
    const xTicks  = 6
    for (let i = 0; i <= xTicks; i++) {
      const v = minR + (i / xTicks) * (maxR - minR)
      ctx.fillText(`${(v * 100).toFixed(1)}%`, PAD.left + (i / xTicks) * PW, H - 8)
    }

    // Y-axis labels
    ctx.textAlign = 'right'
    ctx.fillStyle = cssVar('rgba(var(--chart-sage-dark-rgb),0.45)')
    const yTicks  = 3
    for (let i = 1; i <= yTicks; i++) {
      const v = Math.round((i / yTicks) * maxCount)
      ctx.fillText(v, PAD.left - 6, PAD.top + (1 - i / yTicks) * PH + 4)
    }

    // Labels on lines
    ctx.font      = '9px monospace'
    ctx.textAlign = 'center'
    ctx.fillStyle = cssVar('rgba(var(--signal-negative-rgb),0.85)')
    ctx.fillText(`VaR ${(confidence * 100).toFixed(0)}%`, varX, PAD.top + 10)

    ctx.fillStyle = cssVar('rgba(var(--signal-caution-rgb),0.85)')
    ctx.fillText('CVaR', cvarX, PAD.top + 10)

    ctx.fillStyle = cssVar('rgba(var(--white-rgb),0.3)')
    ctx.fillText('Mean', toX(mean), PAD.top + 10)

  }, [portfolioReturns, varPct, cvarPct])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>

      {/* Legend */}
      <div style={{ display: 'flex', gap: 16, fontSize: 10, color: 'var(--text-muted)' }}>
        {[
          { color: 'rgba(var(--signal-positive-rgb),0.7)',  label: 'Gain days' },
          { color: 'var(--chart-amber-dark-wash)', label: 'Small loss' },
          { color: 'rgba(var(--signal-negative-rgb),0.75)',  label: 'Tail risk' },
          { color: null,                    label: 'VaR / CVaR', dashed: true },
          { color: 'rgba(var(--white-rgb),0.2)', label: 'Normal curve', line: true },
        ].map(l => (
          <span key={l.label} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            {l.dashed ? (
              <div style={{ width: 16, height: 0, borderBottom: '1.5px dashed rgba(var(--signal-negative-rgb),0.7)' }}/>
            ) : l.line ? (
              <div style={{ width: 16, height: 0, borderBottom: `1.5px solid ${l.color}` }}/>
            ) : (
              <div style={{ width: 8, height: 8, background: l.color, borderRadius: 'var(--radius-xs)' }}/>
            )}
            {l.label}
          </span>
        ))}
      </div>

      {/* Canvas */}
      <div style={{
        borderRadius: 'var(--radius-10)', overflow: 'hidden',
        border: '1px solid rgba(var(--white-rgb),0.05)',
      }}>
        <canvas ref={canvasRef} style={{ width: '100%', height: 200, display: 'block' }} />
      </div>

      {/* Stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
        {[
          { label: 'Mean daily',   value: fmtS(mean),  color: mean >= 0 ? 'var(--signal-positive)' : 'var(--signal-negative)' },
          { label: 'Std dev',      value: fmtS(std),   color: 'var(--text-primary)' },
          { label: 'Positive days',value: `${(posPct * 100).toFixed(1)}%`, color: posPct > 0.5 ? 'var(--signal-positive)' : 'var(--signal-negative)' },
          { label: 'Observations', value: returns.length, color: 'var(--text-muted)' },
        ].map(s => (
          <div key={s.label} style={{
            padding: '8px 12px',
            background: 'rgba(var(--white-rgb),0.03)',
            border: '1px solid rgba(var(--white-rgb),0.05)',
            borderRadius: 'var(--radius-sm)',
          }}>
            <div style={{ fontSize: 9, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.07em', fontWeight: 600 }}>
              {s.label}
            </div>
            <div style={{ fontSize: 13, fontWeight: 700, fontFamily: 'var(--font-mono)', color: s.color }}>
              {s.value}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}