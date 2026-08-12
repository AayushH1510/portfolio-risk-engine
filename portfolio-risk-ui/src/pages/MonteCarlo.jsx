import { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine, Customized,
} from 'recharts'
import MetricCard from '../components/MetricCard'
import InsightBox from '../components/InsightBox'

const fmtD = v => `$${v.toLocaleString('en-US', { maximumFractionDigits: 0 })}`

const SCENARIOS = [
  { key: 'bear', label: 'Bear', color: '#e05c5c', sub: '−10%/yr headwind' },
  { key: 'base', label: 'Base', color: '#52b788', sub: 'historical drift' },
  { key: 'bull', label: 'Bull', color: '#b7e4c7', sub: '+10%/yr tailwind' },
]

function ScenarioToggle({ scenario, onChange }) {
  return (
    <div className="card" style={{ display: 'flex', gap: 8, padding: 8, flexShrink: 0 }}>
      {SCENARIOS.map(s => {
        const active = scenario === s.key
        return (
          <button
            key={s.key}
            onClick={() => onChange(s.key)}
            style={{
              flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2,
              padding: '8px 0', borderRadius: 8,
              border: `1px solid ${active ? s.color : 'var(--border)'}`,
              background: active ? `${s.color}22` : 'rgba(255,255,255,0.03)',
              cursor: 'pointer', transition: 'all 0.15s',
            }}
          >
            <span style={{
              fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em',
              color: active ? s.color : 'var(--text-muted)',
            }}>
              {s.label}
            </span>
            <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>{s.sub}</span>
          </button>
        )
      })}
    </div>
  )
}

// Canvas layer that draws the 1,000 faint grey paths
// Custom Recharts layer — renders sampled paths using the chart's own coordinate system
function SimulatedPaths({ allPaths, xAxisMap, yAxisMap }) {
  if (!allPaths || !xAxisMap || !yAxisMap) return null

  const xAxis = Object.values(xAxisMap)[0]
  const yAxis = Object.values(yAxisMap)[0]
  if (!xAxis || !yAxis) return null

  const { scale: xScale } = xAxis
  const { scale: yScale } = yAxis
  if (!xScale || !yScale) return null

  const nDays = allPaths[0]?.length - 1 || 252
  const step  = Math.max(1, Math.floor(allPaths.length / 150))

  const paths = allPaths
    .filter((_, i) => i % step === 0)
    .map((path, pi) => {
      const d = path.map((v, i) => {
        const x = xScale(i)
        const y = yScale(v)
        return `${i === 0 ? 'M' : 'L'}${x},${y}`
      }).join(' ')
      return <path key={pi} d={d} stroke="rgba(180,210,180,0.09)" strokeWidth="0.7" fill="none" />
    })

  return <g>{paths}</g>
}

export default function MonteCarlo({ data }) {
  const [scenario, setScenario] = useState('base')

  if (!data) return null

  const mc = data[`monte_carlo_${scenario}`] || data.monte_carlo
  const {
    percentile_5: p5, percentile_50: p50, percentile_95: p95,
    p5_final, p50_final, p95_final,
    prob_profit, prob_loss_10pct, portfolio_value, n_simulations,
    all_paths,
  } = mc

  const chartData = p50.map((val, i) => ({
    day: i, p5: p5[i], p50: val, p95: p95[i],
  }))

  const gain    = p50_final - portfolio_value
  const gainPct = ((p50_final / portfolio_value) - 1) * 100
  const tone    = prob_profit > 0.75 ? 'good' : prob_profit > 0.5 ? 'warning' : 'bad'

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:12, height:'100%' }}>

      {/* Scenario toggle */}
      <ScenarioToggle scenario={scenario} onChange={setScenario} />

      {/* Metrics row */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(4, 1fr)', gap:10, flexShrink:0 }}>
        <MetricCard label="Starting value"      value={fmtD(portfolio_value)} tone="neutral" />
        <MetricCard label="Median outcome"      value={fmtD(p50_final)} sub={`${gainPct > 0 ? '+' : ''}${gainPct.toFixed(1)}%`} tone={gain > 0 ? 'good' : 'bad'} />
        <MetricCard label="Chance of profit"    value={`${Math.round(prob_profit * 100)}%`} tone={prob_profit > 0.6 ? 'good' : 'warning'} sub={`${n_simulations.toLocaleString()} simulations`} />
        <MetricCard label="Chance of -10% loss" value={`${Math.round(prob_loss_10pct * 100)}%`} tone={prob_loss_10pct < 0.1 ? 'good' : prob_loss_10pct < 0.25 ? 'warning' : 'bad'} />
      </div>

      {/* Chart */}
      <div className="card" style={{ padding:'16px', flex:1, minHeight:0, display:'flex', flexDirection:'column' }}>

        {/* Header */}
        <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:12, flexShrink:0 }}>
          <div style={{ fontSize:11, fontWeight:700, textTransform:'uppercase', letterSpacing:'0.06em', color:'var(--text-secondary)' }}>
            {n_simulations.toLocaleString()} simulated futures — next 12 months
          </div>
          <div style={{ display:'flex', gap:14, fontSize:10 }}>
            {[
              { color:'#e05c5c', label:`Bad (5th) — ${fmtD(p5_final)}`,   dash:true },
              { color:'#52b788', label:`Median — ${fmtD(p50_final)}`,      dash:false },
              { color:'#b7e4c7', label:`Good (95th) — ${fmtD(p95_final)}`, dash:true },
            ].map(l => (
              <span key={l.label} style={{ display:'flex', alignItems:'center', gap:5, color:'var(--text-muted)' }}>
                <svg width="20" height="2" style={{ flexShrink:0 }}>
                  <line x1="0" y1="1" x2="20" y2="1" stroke={l.color} strokeWidth="2" strokeDasharray={l.dash ? '4 3' : 'none'}/>
                </svg>
                {l.label}
              </span>
            ))}
            <span style={{ display:'flex', alignItems:'center', gap:5, color:'var(--text-muted)' }}>
              <div style={{ width:16, height:1.5, background:'rgba(180,210,180,0.2)', borderRadius:1 }}/>
              Simulated paths
            </span>
          </div>
        </div>

        {/* Chart */}
        <div style={{ flex:1, minHeight:0 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top:8, right:8, bottom:0, left:8 }}>
              <XAxis
                dataKey="day"
                tick={{ fill:'var(--text-muted)', fontSize:10 }}
                tickLine={false} axisLine={false}
                label={{ value:'Trading days (252 = 1 year)', position:'insideBottom', offset:-2, fill:'var(--text-muted)', fontSize:10 }}
              />
              <YAxis
                tickFormatter={v => `$${(v/1000).toFixed(0)}k`}
                tick={{ fill:'var(--text-muted)', fontSize:10 }}
                tickLine={false} axisLine={false}
                width={48}
              />
              <Tooltip
                formatter={(v, n) => [fmtD(v), n === 'p5' ? 'Bad scenario' : n === 'p50' ? 'Median' : 'Good scenario']}
                labelFormatter={l => `Day ${l}`}
                contentStyle={{ background:'var(--card)', border:'1px solid var(--border)', borderRadius:8, fontSize:11 }}
              />
              <ReferenceLine y={portfolio_value} stroke="rgba(255,255,255,0.1)" strokeDasharray="4 4"
                label={{ value:'Start', fill:'var(--text-muted)', fontSize:9, position:'right' }} />
              <Line type="monotone" dataKey="p5"  stroke="#e05c5c" strokeWidth={1.5} dot={false} strokeDasharray="5 4" />
              <Line type="monotone" dataKey="p50" stroke="#52b788" strokeWidth={2}   dot={false} />
              <Line type="monotone" dataKey="p95" stroke="#b7e4c7" strokeWidth={1.5} dot={false} strokeDasharray="5 4" />
              {all_paths && (
                <Customized component={<SimulatedPaths allPaths={all_paths} />} />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <InsightBox
          label="Monte Carlo insight"
          tone={tone}
          text={`Based on ${n_simulations.toLocaleString()} simulations, <strong>${Math.round(prob_profit*100)}%</strong> of futures end the year profitable. The median outcome is <strong>${fmtD(p50_final)}</strong> — a ${gainPct > 0 ? 'gain' : 'loss'} of <strong>${fmtD(Math.abs(gain))}</strong>. Good year: <strong>${fmtD(p95_final)}</strong>. Bad year: <strong>${fmtD(p5_final)}</strong>. ${prob_loss_10pct > 0.2 ? `Note: <strong>${Math.round(prob_loss_10pct*100)}% chance of losing 10%+</strong> — ensure you can hold through that.` : 'Past performance does not guarantee future results.'}`}
        />
      </div>
    </div>
  )
}