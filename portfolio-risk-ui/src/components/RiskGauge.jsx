import { useEffect, useRef, useState } from 'react'

function computeRiskScore(vol, drawdown, varPct) {
  const v  = isFinite(vol)      ? vol      : 0.15
  const dd = isFinite(drawdown) ? drawdown : -0.10
  const vp = isFinite(varPct)   ? varPct   : -0.02
  const volScore = Math.min((v / 0.40) * 100, 100)
  const ddScore  = Math.min((Math.abs(dd) / 0.60) * 100, 100)
  const varScore = Math.min((Math.abs(vp) / 0.05) * 100, 100)
  return Math.round(volScore * 0.4 + ddScore * 0.35 + varScore * 0.25)
}

function riskLabel(score) {
  if (score < 25) return { label: 'Low',      color: '#52b788' }
  if (score < 50) return { label: 'Moderate', color: '#d4c84a' }
  if (score < 75) return { label: 'Elevated', color: '#e09a30' }
  return              { label: 'High',      color: '#e05c5c' }
}

export default function RiskGauge({ vol, drawdown, varPct }) {
  const canvasRef  = useRef(null)
  const [hovered, setHovered] = useState(false)

  const score = computeRiskScore(vol, drawdown, varPct)
  const { label, color } = riskLabel(score)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W   = canvas.offsetWidth
    const H   = canvas.offsetHeight
    canvas.width  = W * window.devicePixelRatio
    canvas.height = H * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    const cx     = W / 2
    const cy     = H * 0.68
    const R      = Math.min(W * 0.44, H * 0.82)
    // 9 o'clock (180°) → clockwise 180° → 3 o'clock (0°/360°)
    const START  = Math.PI        // 180° = 9 o'clock
    const SWEEP  = Math.PI        // 180° sweep
    const arcEnd = START + SWEEP  // = 360° = 3 o'clock

    // Match card background
    ctx.fillStyle = '#1e2420'
    ctx.fillRect(0, 0, W, H)

    // Track arc — subtle background
    ctx.beginPath()
    ctx.arc(cx, cy, R * 0.78, START, arcEnd)
    ctx.strokeStyle = 'rgba(255,255,255,0.05)'
    ctx.lineWidth   = R * 0.10
    ctx.lineCap     = 'round'
    ctx.stroke()

    // Colour zones — thin, low opacity reference
    const zones = [
      { from: 0,    to: 0.25, color: '#52b788' },
      { from: 0.25, to: 0.50, color: '#d4c84a' },
      { from: 0.50, to: 0.75, color: '#e09a30' },
      { from: 0.75, to: 1.00, color: '#e05c5c' },
    ]
    zones.forEach(z => {
      ctx.beginPath()
      ctx.arc(cx, cy, R * 0.78, START + z.from * SWEEP, START + z.to * SWEEP)
      ctx.strokeStyle = z.color
      ctx.lineWidth   = R * 0.10
      ctx.globalAlpha = 0.18
      ctx.stroke()
      ctx.globalAlpha = 1
    })

    // Active arc
    const scoreAngle = START + (score / 100) * SWEEP
    const actGrad = ctx.createLinearGradient(
      cx + R * Math.cos(START), cy + R * Math.sin(START),
      cx + R * Math.cos(scoreAngle), cy + R * Math.sin(scoreAngle)
    )
    actGrad.addColorStop(0, '#52b788')
    actGrad.addColorStop(0.33, '#d4c84a')
    actGrad.addColorStop(0.66, '#e09a30')
    actGrad.addColorStop(1, color)

    ctx.beginPath()
    ctx.arc(cx, cy, R * 0.78, START, scoreAngle)
    ctx.strokeStyle = actGrad
    ctx.lineWidth   = R * 0.10
    ctx.lineCap     = 'round'
    ctx.stroke()

    // Minimal tick marks — just 5 markers, no labels
    for (let t = 0; t <= 100; t += 25) {
      const a  = START + (t / 100) * SWEEP
      const r1 = R * 0.68
      const r2 = R * 0.75
      ctx.beginPath()
      ctx.moveTo(cx + r1 * Math.cos(a), cy + r1 * Math.sin(a))
      ctx.lineTo(cx + r2 * Math.cos(a), cy + r2 * Math.sin(a))
      ctx.strokeStyle = 'rgba(255,255,255,0.2)'
      ctx.lineWidth   = 1
      ctx.stroke()
    }

    // Needle
    const na   = START + (score / 100) * SWEEP
    const nLen = R * 0.70
    const nTip = { x: cx + nLen * Math.cos(na), y: cy + nLen * Math.sin(na) }
    ctx.beginPath()
    ctx.moveTo(cx, cy); ctx.lineTo(nTip.x, nTip.y)
    ctx.strokeStyle = 'rgba(255,255,255,0.9)'
    ctx.lineWidth   = 1.5
    ctx.lineCap     = 'round'
    ctx.stroke()

    // Needle pivot
    ctx.beginPath()
    ctx.arc(cx, cy, R * 0.06, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(255,255,255,0.15)'
    ctx.fill()
    ctx.beginPath()
    ctx.arc(cx, cy, R * 0.03, 0, Math.PI * 2)
    ctx.fillStyle = '#ffffff'
    ctx.fill()

    // Score — small, muted, no glow
    ctx.fillStyle    = 'rgba(180,210,180,0.6)'
    ctx.font         = `500 ${R * 0.14}px "IBM Plex Mono", monospace`
    ctx.textAlign    = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(score, cx, cy + R * 0.32)

  }, [score])

  return (
    <div style={{ display:'flex', flexDirection:'column' }}>
      <div style={{
        fontSize:10, fontWeight:700, textTransform:'uppercase',
        letterSpacing:'0.08em', color:'var(--text-muted)', marginBottom:6,
      }}>
        Risk Gauge
      </div>
      <div
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        style={{
          borderRadius:12, overflow:'hidden',
          border:`1px solid ${hovered ? color : 'rgba(255,255,255,0.06)'}`,
          transition:'border-color 0.2s',
          background:'var(--card)',
        }}
      >
        <canvas
          ref={canvasRef}
          style={{ width:'100%', aspectRatio:'1.4/1', display:'block' }}
        />
      </div>
    </div>
  )
}