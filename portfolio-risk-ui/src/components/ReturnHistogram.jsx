import { useEffect, useRef, useState } from 'react'
import { cssVar } from '../lib/cssVar'
import MetricTooltip from './MetricTooltip'

export default function ReturnHistogram({ portfolioReturns, varPct, cvarPct, varDollar, cvarDollar, confidence }) {
  const canvasRef = useRef(null)
  const geomRef   = useRef(null)
  const [tooltip, setTooltip] = useState(null)

  if (!portfolioReturns?.length) return null

  const returns = portfolioReturns
  const mean    = returns.reduce((a, b) => a + b, 0) / returns.length
  const std     = Math.sqrt(returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length)
  const minR    = Math.min(...returns)
  const maxR    = Math.max(...returns)
  const posPct  = returns.filter(r => r > 0).length / returns.length
  const fmt     = v => `${(v * 100).toFixed(2)}%`
  const fmtS    = v => `${(v * 100).toFixed(1)}%`

  // Annualised so it's on the same scale as the Volatility metric card's own
  // < 0.20 / < 0.35 "good/warning/bad" cutoffs (Dashboard.jsx) — reusing that
  // established threshold here rather than inventing a new one for what's
  // fundamentally the same underlying spread, just daily instead of annual.
  const annualisedStd = std * Math.sqrt(252)
  const tightSpread    = annualisedStd < 0.20

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
    ctx.fillStyle = cssVar('var(--surface-card)')
    ctx.fillRect(0, 0, W, H)

    // Subtle grid lines — DESIGN.md Charts spec: --line-faint, 1px, solid
    ctx.strokeStyle = cssVar('var(--line-faint)')
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
    ctx.strokeStyle = cssVar('rgba(var(--text-primary-rgb),0.12)')
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

      const rgb = binMid < varPct
        ? cssVar('var(--signal-negative-rgb)')
        : binMid < 0
          ? cssVar('var(--signal-caution-rgb)')
          : cssVar('var(--signal-positive-rgb)')
      const alpha = binMid < varPct ? 0.7 : 0.6

      // Bar body — square corners, no rounding (DESIGN.md: zero radius, no exceptions)
      ctx.fillStyle = `rgba(${rgb},${alpha})`
      ctx.fillRect(x + 1, y, bW - 2, barH)

      // Top cap
      ctx.fillStyle = `rgba(${rgb},1)`
      ctx.fillRect(x + 1, y, bW - 2, 1.5)
    })

    // Normal distribution curve
    const gaussian = x => {
      return Math.exp(-0.5 * ((x - mean) / std) ** 2) / (std * Math.sqrt(2 * Math.PI))
    }
    const scaleFactor = maxCount / (gaussian(mean) * returns.length * binW)

    ctx.beginPath()
    ctx.strokeStyle = cssVar('rgba(var(--text-primary-rgb),0.18)')
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
    ctx.font      = `9px ${cssVar('var(--font-mono)')}`
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
    const meanX = toX(mean)
    ctx.font      = `9px ${cssVar('var(--font-mono)')}`
    ctx.textAlign = 'center'
    ctx.fillStyle = cssVar('rgba(var(--signal-negative-rgb),0.85)')
    ctx.fillText(`VaR ${(confidence * 100).toFixed(0)}%`, varX, PAD.top + 10)

    ctx.fillStyle = cssVar('rgba(var(--signal-caution-rgb),0.85)')
    ctx.fillText('CVaR', cvarX, PAD.top + 10)

    ctx.fillStyle = cssVar('rgba(var(--text-primary-rgb),0.3)')
    ctx.fillText('Mean', meanX, PAD.top + 10)

    // Hit-test geometry for hover — recomputed every draw so mouse moves
    // always test against the coordinates actually on screen right now,
    // not a stale layout from a previous render (matches the pattern in
    // Frontier.jsx's canvas hover, the only other raw-canvas chart with
    // interactive hover in this app).
    geomRef.current = {
      W, H, PAD, PW, PH,
      meanX, varX, cvarX,
      bins: bins.map((count, i) => ({
        x: PAD.left + (i / N_BINS) * PW,
        width: bW,
        binStart: minR + i * binW,
        binEnd: minR + (i + 1) * binW,
        count,
      })),
    }

  }, [portfolioReturns, varPct, cvarPct])

  const handleMouseMove = e => {
    const canvas = canvasRef.current
    const g = geomRef.current
    if (!canvas || !g) return
    const rect = canvas.getBoundingClientRect()
    const mx = e.clientX - rect.left
    const my = e.clientY - rect.top

    // Bottom strip — x-axis tick labels live here (H - PAD.bottom to H).
    if (my >= g.PAD.top + g.PH && mx >= g.PAD.left && mx <= g.W - g.PAD.right) {
      setTooltip({ kind: 'xaxis', left: mx, top: g.PAD.top + g.PH })
      return
    }
    // Left strip — y-axis tick labels live here (0 to PAD.left).
    if (mx <= g.PAD.left && my >= g.PAD.top && my <= g.PAD.top + g.PH) {
      setTooltip({ kind: 'yaxis', left: g.PAD.left, top: my })
      return
    }
    // Outside the plot area and both axis strips (top/right margins, or off
    // the bottom-left corner) — nothing meaningful to show.
    if (mx < g.PAD.left || mx > g.W - g.PAD.right || my < g.PAD.top || my > g.PAD.top + g.PH) {
      setTooltip(null)
      return
    }

    // Inside the main plot — lines get first claim within a tight pixel
    // tolerance (they're precise 1.5px targets), bars are the fallback,
    // covering the rest of the plot as one continuous hoverable surface.
    const LINE_TOL = 5
    if (Math.abs(mx - g.varX) < LINE_TOL) { setTooltip({ kind: 'var', left: g.varX, top: my }); return }
    if (Math.abs(mx - g.cvarX) < LINE_TOL) { setTooltip({ kind: 'cvar', left: g.cvarX, top: my }); return }
    if (Math.abs(mx - g.meanX) < LINE_TOL) { setTooltip({ kind: 'mean', left: g.meanX, top: my }); return }

    const bin = g.bins.find(b => mx >= b.x && mx < b.x + b.width)
    setTooltip(bin ? { kind: 'bar', left: mx, top: my, bin } : null)
  }

  const TOOLTIP_LABEL = { fontSize: 9, fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase', fontFamily: 'var(--font-primary)', color: 'var(--text-muted)', marginBottom: 3 }
  const TOOLTIP_VALUE = { fontSize: 12, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }

  const renderTooltipContent = () => {
    switch (tooltip.kind) {
      case 'mean':
        return (<>
          <div style={{ ...TOOLTIP_LABEL, color: 'rgba(var(--text-primary-rgb),0.5)' }}>Mean daily return</div>
          <div style={TOOLTIP_VALUE}>{fmt(mean)}</div>
        </>)
      case 'var':
        return (<>
          <div style={{ ...TOOLTIP_LABEL, color: 'var(--signal-negative)' }}>VaR {(confidence * 100).toFixed(0)}%</div>
          <div style={{ ...TOOLTIP_VALUE, color: 'var(--signal-negative)' }}>{fmt(varPct)}</div>
          {varDollar != null && <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>${Math.abs(varDollar).toFixed(0)} per day</div>}
        </>)
      case 'cvar':
        return (<>
          <div style={{ ...TOOLTIP_LABEL, color: 'var(--signal-caution)' }}>CVaR {(confidence * 100).toFixed(0)}%</div>
          <div style={{ ...TOOLTIP_VALUE, color: 'var(--signal-caution)' }}>{fmt(cvarPct)}</div>
          {cvarDollar != null && <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>${Math.abs(cvarDollar).toFixed(0)} avg</div>}
        </>)
      case 'bar':
        return (<>
          <div style={TOOLTIP_LABEL}>{fmt(tooltip.bin.binStart)} to {fmt(tooltip.bin.binEnd)}</div>
          <div style={TOOLTIP_VALUE}>{tooltip.bin.count} day{tooltip.bin.count === 1 ? '' : 's'}</div>
        </>)
      case 'xaxis':
        return <div style={{ fontSize: 10, color: 'var(--text-secondary)', maxWidth: 140 }}>The size of a day's gain or loss</div>
      case 'yaxis':
        return <div style={{ fontSize: 10, color: 'var(--text-secondary)', maxWidth: 140 }}>How many days fell in that range</div>
      default:
        return null
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>

      {/* Legend */}
      <div style={{ display: 'flex', gap: 16, fontSize: 10, color: 'var(--text-muted)' }}>
        {[
          { color: 'rgba(var(--signal-positive-rgb),0.7)',  label: 'Gain days' },
          { color: 'var(--chart-amber-dark-wash)', label: 'Small loss' },
          { color: 'rgba(var(--signal-negative-rgb),0.75)',  label: 'Tail risk' },
          { color: null,                    label: 'VaR / CVaR', dashed: true },
          { color: 'rgba(var(--text-primary-rgb),0.2)', label: 'Normal curve', line: true },
        ].map(l => (
          <span key={l.label} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            {l.dashed ? (
              <div style={{ width: 16, height: 0, borderBottom: '1.5px dashed rgba(var(--signal-negative-rgb),0.7)' }}/>
            ) : l.line ? (
              <div style={{ width: 16, height: 0, borderBottom: `1.5px solid ${l.color}` }}/>
            ) : (
              <div style={{ width: 8, height: 8, background: l.color }}/>
            )}
            {l.label}
          </span>
        ))}
      </div>

      {/* Canvas */}
      <div style={{
        position: 'relative',
        overflow: 'hidden',
        border: '1px solid rgba(var(--text-primary-rgb),0.05)',
      }}>
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setTooltip(null)}
          style={{ width: '100%', height: 200, display: 'block', cursor: tooltip ? 'crosshair' : 'default' }}
        />
        {tooltip && (
          <div style={{
            position: 'absolute',
            left: Math.min(Math.max(tooltip.left - 60, 4), (canvasRef.current?.offsetWidth || 999) - 140),
            top: Math.max(tooltip.top - 54, 4),
            background: 'var(--surface-elevated)',
            border: 'var(--border-emphasis)',
            padding: '8px 10px',
            pointerEvents: 'none',
            zIndex: 10,
            minWidth: 120,
          }}>
            {renderTooltipContent()}
          </div>
        )}
      </div>

      {/* Stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
        {[
          { key: 'mean_daily',    label: 'Mean daily',    value: fmtS(mean),  color: mean >= 0 ? 'var(--signal-positive)' : 'var(--signal-negative)' },
          { key: 'std_dev',       label: 'Std dev',       value: fmtS(std),   color: 'var(--text-primary)' },
          { key: 'positive_days', label: 'Positive days', value: `${(posPct * 100).toFixed(1)}%`, color: posPct > 0.5 ? 'var(--signal-positive)' : 'var(--signal-negative)' },
          { key: 'observations',  label: 'Observations',  value: returns.length, color: 'var(--text-muted)' },
        ].map(s => (
          <div key={s.label} style={{
            padding: '8px 12px',
            background: 'var(--surface-elevated)',
            border: 'var(--border-faint)',
          }}>
            <div style={{ fontSize: 'var(--text-caption)', color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', fontWeight: 'var(--weight-medium)', fontFamily: 'var(--font-primary)' }}>
              <MetricTooltip metricKey={s.key}>{s.label}</MetricTooltip>
            </div>
            <div style={{ fontSize: 13, fontWeight: 700, fontFamily: 'var(--font-mono)', color: s.color }}>
              {s.value}
            </div>
          </div>
        ))}
      </div>

      <div style={{ fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.5 }}>
        {tightSpread
          ? 'Most days cluster tightly around the average, a sign of predictable performance.'
          : 'Returns are spread widely, including some sharp outlier days.'}
      </div>
    </div>
  )
}
