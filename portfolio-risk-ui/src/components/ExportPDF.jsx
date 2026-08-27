import {
  Area, LineChart, Line, ComposedChart,
  XAxis, YAxis, ResponsiveContainer, ReferenceLine,
} from 'recharts'

const fmt  = v => v != null ? `${(v * 100).toFixed(1)}%` : 'N/A'
const fmtN = v => v != null ? v.toFixed(2) : 'N/A'
const fmtD = v => v != null ? `$${Math.abs(v).toLocaleString('en-US', { maximumFractionDigits:0 })}` : 'N/A'

function ReportTemplate({ data, tickers, weights }) {
  const cum  = data.cumulative_returns
  const bench= data.benchmark_cumulative
  const mc   = data.monte_carlo
  const vc   = data.var_cvar
  const ba   = data.beta_alpha
  const corr = data.correlation_matrix

  const growthData = cum.dates.map((d, i) => ({
    date: d.slice(5),
    portfolio: cum.values[i],
    ...(bench?.values[i] != null ? { benchmark: bench.values[i] } : {}),
  }))

  const mcData = mc.percentile_50.map((v, i) => ({
    day: i, median: v, bad: mc.percentile_5[i], good: mc.percentile_95[i],
  }))

  const metrics = [
    { label:'Annual Return',  val: fmt(data.annualised_return),       good: data.annualised_return > 0 },
    { label:'Volatility',     val: fmt(data.annualised_volatility),    good: data.annualised_volatility < 0.25 },
    { label:'Sharpe Ratio',   val: fmtN(data.sharpe_ratio),           good: data.sharpe_ratio > 1 },
    { label:'Sortino Ratio',  val: fmtN(data.sortino_ratio),          good: data.sortino_ratio > 1 },
    { label:'Max Drawdown',   val: fmt(data.max_drawdown),            good: false },
    { label:'VaR 95%',        val: fmt(vc.var_pct),                   good: false },
    { label:'CVaR 95%',       val: fmt(vc.cvar_pct),                  good: false },
    ...(ba ? [
      { label:'Beta',  val: fmtN(ba.beta),  good: ba.beta < 1.2 },
      { label:'Alpha', val: fmt(ba.alpha),  good: ba.alpha > 0 },
    ] : []),
  ]

  const allocColors = ['var(--accent-dark)','var(--accent)','var(--signal-positive-soft)','var(--chart-mint)','var(--accent-light)']
  const corrColor   = v => v >= 0.7 ? 'var(--report-heat-high)' : v >= 0.4 ? 'var(--report-heat-mid)' : 'var(--report-heat-low)'
  const date        = new Date().toLocaleDateString('en-GB', { day:'numeric', month:'long', year:'numeric' })

  // All styles are inline — no <style> tag that would leak into the parent app
  return (
    <div style={{ width:794, fontFamily:'var(--font-primary)', fontSize:12, color:'var(--report-text-dark)', background:'var(--report-bg)' }}>

      {/* Header */}
      <div style={{ background:"var(--card)", padding:"24px 32px", display:"flex", justifyContent:"space-between", alignItems:"flex-start" }}>
        <div>
          <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:8 }}>
            <div style={{ width:30, height:30, background:'var(--accent)', borderRadius:'var(--radius-6)', display:'flex', alignItems:'center', justifyContent:'center', fontWeight:900, color:'var(--card)', fontSize:15 }}>V</div>
            <div>
              <div style={{ fontSize:18, fontWeight:700, color:'var(--white)', letterSpacing:'-0.02em' }}>varense</div>
              <div style={{ fontSize:10, color:'var(--accent)', letterSpacing:'0.05em' }}>variance, made sense of</div>
            </div>
          </div>
          <div style={{ marginTop:10 }}>
            {tickers.map((t,i) => <span key={t} style={{ display:"inline-block", background:"rgba(var(--signal-positive-rgb),0.15)", border:"1px solid rgba(var(--signal-positive-rgb),0.3)", borderRadius:'var(--radius-4)', padding:"2px 9px", fontSize:11, fontWeight:700, color:"var(--accent)", fontFamily:"var(--font-mono)", marginRight:5 }}>{t} {Math.round(weights[i]*100)}%</span>)}
          </div>
        </div>
        <div style={{ textAlign:'right' }}>
          <div style={{ fontSize:16, fontWeight:700, color:'var(--white)' }}>Varense Portfolio Report</div>
          <div style={{ fontSize:11, color:'var(--text-secondary)', marginTop:3 }}>{date}</div>
          <div style={{ fontSize:10, color:'var(--text-muted)', marginTop:5, fontFamily:'var(--font-mono)' }}>
            {data.period.start} — {data.period.end} · {data.period.n_days} days
          </div>
        </div>
      </div>

      <div style={{ padding:"20px 32px", display:"flex", flexDirection:"column", gap:18 }}>

        {/* Metrics */}
        <div>
          <div style={{ fontSize:10, fontWeight:700, textTransform:"uppercase", letterSpacing:"0.08em", color:"var(--text-muted)", marginBottom:8 }}>Key Metrics</div>
          <div style={{ display:"grid", gridTemplateColumns:"repeat(5,1fr)", gap:7 }}>
            {metrics.map(m => (
              <div key={m.label} style={{ background:"var(--white)", borderRadius:'var(--radius-7)', padding:"10px 12px", boxShadow:"0 1px 3px rgba(var(--black-rgb),0.06)", borderTop:`3px solid ${m.good ? 'var(--accent)' : 'var(--negative)'}` }}>
                <div style={{ fontSize:9, fontWeight:700, textTransform:"uppercase", letterSpacing:"0.06em", color:"var(--text-secondary)", marginBottom:4 }}>{m.label}</div>
                <div style={{ fontSize:16, fontWeight:700, fontFamily:"var(--font-mono)" }} style={{ color: m.good ? 'var(--accent-dark)' : 'var(--report-red)' }}>{m.val}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Growth chart */}
        <div>
          <div style={{ fontSize:10, fontWeight:700, textTransform:"uppercase", letterSpacing:"0.08em", color:"var(--text-muted)", marginBottom:8 }}>Portfolio Growth vs S&P 500</div>
          <div style={{ background:"var(--white)", borderRadius:'var(--radius-sm)', padding:14, boxShadow:"0 1px 3px rgba(var(--black-rgb),0.06)" }}>
            <div style={{ display:'flex', justifyContent:'space-between', marginBottom:8 }}>
              <span style={{ fontSize:11, color:'var(--text-secondary)' }}>Cumulative return from {data.period.start}</span>
              <span style={{ fontWeight:700, color: data.annualised_return > 0 ? 'var(--accent-dark)':'var(--report-red)', fontFamily:'var(--font-mono)', fontSize:12 }}>
                {fmt(cum.values[cum.values.length-1])} total
              </span>
            </div>
            <ResponsiveContainer width="100%" height={160}>
              <ComposedChart data={growthData} margin={{ top:4, right:4, bottom:0, left:40 }}>
                <defs>
                  <linearGradient id="rg1" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="var(--accent)" stopOpacity={0.2}/>
                    <stop offset="95%" stopColor="var(--accent)" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="date" tick={{ fontSize:9, fill:'var(--text-secondary)' }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
                <YAxis tickFormatter={v => `${(v*100).toFixed(0)}%`} tick={{ fontSize:9, fill:'var(--text-secondary)' }} tickLine={false} axisLine={false} />
                <ReferenceLine y={0} stroke="var(--report-gridline)" strokeDasharray="4 4" />
                <Area type="monotone" dataKey="portfolio" stroke="var(--accent)" strokeWidth={2} fill="url(#rg1)" dot={false} name="Portfolio" />
                {bench && <Line type="monotone" dataKey="benchmark" stroke="var(--text-secondary)" strokeWidth={1.5} strokeDasharray="5 4" dot={false} name="S&P 500" />}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Allocation + Monte Carlo */}
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14 }}>
          <div>
            <div style={{ fontSize:10, fontWeight:700, textTransform:"uppercase", letterSpacing:"0.08em", color:"var(--text-muted)", marginBottom:8 }}>Allocation</div>
            <div style={{ background:"var(--white)", borderRadius:'var(--radius-sm)', padding:14, boxShadow:"0 1px 3px rgba(var(--black-rgb),0.06)" }}>
              {tickers.map((t,i) => (
                <div key={t} style={{ marginBottom:9 }}>
                  <div style={{ display:'flex', justifyContent:'space-between', marginBottom:3 }}>
                    <span style={{ fontWeight:700, fontFamily:'var(--font-mono)', fontSize:12 }}>{t}</span>
                    <span style={{ fontWeight:700, color:'var(--accent-dark)', fontSize:12 }}>{Math.round(weights[i]*100)}%</span>
                  </div>
                  <div style={{ height:5, background:"var(--report-track)", borderRadius:'var(--radius-3)' }}>
                    <div style={{ height:5, borderRadius:'var(--radius-3)' }} style={{ width:`${weights[i]*100}%`, background: allocColors[i % allocColors.length] }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div>
            <div style={{ fontSize:10, fontWeight:700, textTransform:"uppercase", letterSpacing:"0.08em", color:"var(--text-muted)", marginBottom:8 }}>Monte Carlo — 1 Year</div>
            <div style={{ background:"var(--white)", borderRadius:'var(--radius-sm)', padding:14, boxShadow:"0 1px 3px rgba(var(--black-rgb),0.06)" }}>
              <ResponsiveContainer width="100%" height={90}>
                <LineChart data={mcData} margin={{ top:4, right:4, bottom:0, left:38 }}>
                  <XAxis dataKey="day" tick={{ fontSize:8, fill:'var(--text-secondary)' }} tickLine={false} axisLine={false} interval={63} />
                  <YAxis tickFormatter={v => `$${(v/1000).toFixed(0)}k`} tick={{ fontSize:8, fill:'var(--text-secondary)' }} tickLine={false} axisLine={false} />
                  <ReferenceLine y={mc.portfolio_value} stroke="var(--report-gridline)" strokeDasharray="3 3" />
                  <Line type="monotone" dataKey="good"   stroke="var(--accent-light)" strokeWidth={1.5} dot={false} strokeDasharray="4 3" />
                  <Line type="monotone" dataKey="median" stroke="var(--accent)" strokeWidth={2}   dot={false} />
                  <Line type="monotone" dataKey="bad"    stroke="var(--negative)" strokeWidth={1.5} dot={false} strokeDasharray="4 3" />
                </LineChart>
              </ResponsiveContainer>
              <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:5, marginTop:8 }}>
                {[
                  { label:'Median',       val: fmtD(mc.p50_final),                          color:'var(--accent-dark)' },
                  { label:'Good year',    val: fmtD(mc.p95_final),                          color:'var(--accent)' },
                  { label:'Bad year',     val: fmtD(mc.p5_final),                           color:'var(--report-red)' },
                  { label:'Profit chance',val: `${Math.round(mc.prob_profit*100)}%`,         color: mc.prob_profit > 0.6 ? 'var(--accent-dark)':'var(--report-amber)' },
                ].map(m => (
                  <div key={m.label} style={{ background:'var(--report-card-alt)', borderRadius:'var(--radius-5)', padding:'6px 8px' }}>
                    <div style={{ fontSize:9, color:'var(--text-secondary)', fontWeight:700, textTransform:'uppercase', marginBottom:2 }}>{m.label}</div>
                    <div style={{ fontSize:13, fontWeight:700, color:m.color, fontFamily:'var(--font-mono)' }}>{m.val}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Risk + Correlation */}
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14 }}>
          <div>
            <div style={{ fontSize:10, fontWeight:700, textTransform:"uppercase", letterSpacing:"0.08em", color:"var(--text-muted)", marginBottom:8 }}>Downside Risk</div>
            <div style={{ background:"var(--white)", borderRadius:'var(--radius-sm)', padding:14, boxShadow:"0 1px 3px rgba(var(--black-rgb),0.06)" }}>
              {[
                { label:'VaR 95%',       a: fmt(vc.var_pct),    b: `${fmtD(vc.var_dollar)}/day` },
                { label:'CVaR 95%',      a: fmt(vc.cvar_pct),   b: `${fmtD(vc.cvar_dollar)} avg` },
                { label:'Max Drawdown',  a: fmt(data.max_drawdown), b:'peak to trough' },
                ...(ba ? [
                  { label:'Beta',  a: fmtN(ba.beta),  b:'vs S&P 500' },
                  { label:'Alpha', a: fmt(ba.alpha),  b:"Jensen's" },
                ] : []),
              ].map(r => (
                <div key={r.label} style={{ display:"flex", justifyContent:"space-between", padding:"6px 0", borderBottom:"1px solid var(--report-bg)" }}>
                  <span style={{ color:'var(--text-muted)', fontSize:11 }}>{r.label}</span>
                  <div style={{ textAlign:'right' }}>
                    <div style={{ fontWeight:700, color:'var(--report-red)', fontFamily:'var(--font-mono)', fontSize:11 }}>{r.a}</div>
                    <div style={{ fontSize:9, color:'var(--text-secondary)' }}>{r.b}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div>
            <div style={{ fontSize:10, fontWeight:700, textTransform:"uppercase", letterSpacing:"0.08em", color:"var(--text-muted)", marginBottom:8 }}>Correlation Matrix</div>
            <div style={{ background:"var(--white)", borderRadius:'var(--radius-sm)', padding:14, boxShadow:"0 1px 3px rgba(var(--black-rgb),0.06)" }}>
              <table>
                <thead>
                  <tr>
                    <th style={{ width:40 }} />
                    {corr.tickers.map(t => <th key={t}>{t}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {corr.tickers.map((row, ri) => (
                    <tr key={row}>
                      <td style={{ textAlign:'right', color:'var(--text-secondary)', fontSize:10, paddingRight:5, background:'transparent' }}>{row}</td>
                      {corr.values[ri].map((val, ci) => (
                        <td key={ci} style={{ background: corrColor(val) }}>{val.toFixed(2)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Watermark */}
        <div style={{ textAlign:"center" }}>
          <div style={{ display:"inline-block", border:"1px solid var(--report-border)", borderRadius:'var(--radius-5)', padding:"4px 14px", fontSize:11, color:"var(--text-secondary)" }}>
            Generated by <strong style={{ color:'var(--accent-dark)' }}>varense</strong> · Free tier · {date}
          </div>
        </div>

      </div>

      {/* Footer */}
      <div style={{ background:"var(--card)", padding:"12px 32px", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <div style={{ fontSize:10, color:'var(--text-muted)', maxWidth:460, lineHeight:1.5 }}>
          This report is for educational purposes only and does not constitute financial advice.
          Past performance does not guarantee future results. Always do your own research.
        </div>
        <div style={{ fontSize:12, fontWeight:700, color:'var(--accent)' }}>varense.com</div>
      </div>

    </div>
  )
}

export default function ExportPDF({ data, tickers, weights, portfolioValue }) {
  if (!data) return null

  const buildReportHTML = () => {
    const d    = data
    const cum  = d.cumulative_returns
    const mc   = d.monte_carlo
    const vc   = d.var_cvar
    const ba   = d.beta_alpha
    const corr = d.correlation_matrix
    const date = new Date().toLocaleDateString('en-GB', { day:'numeric', month:'long', year:'numeric' })

    const metricRows = [
      { label:'Annual Return',  val: fmt(d.annualised_return),       good: d.annualised_return > 0 },
      { label:'Volatility',     val: fmt(d.annualised_volatility),    good: d.annualised_volatility < 0.25 },
      { label:'Sharpe Ratio',   val: fmtN(d.sharpe_ratio),           good: d.sharpe_ratio > 1 },
      { label:'Sortino Ratio',  val: fmtN(d.sortino_ratio),          good: d.sortino_ratio > 1 },
      { label:'Max Drawdown',   val: fmt(d.max_drawdown),            good: false },
      { label:'VaR 95%',        val: fmt(vc.var_pct),                good: false },
      { label:'CVaR 95%',       val: fmt(vc.cvar_pct),               good: false },
      ...(ba ? [
        { label:'Beta',  val: fmtN(ba.beta),  good: ba.beta < 1.2 },
        { label:'Alpha', val: fmt(ba.alpha),  good: ba.alpha > 0 },
      ] : []),
    ]

    const allocColors = ['var(--accent-dark)','var(--accent)','var(--signal-positive-soft)','var(--chart-mint)','var(--accent-light)']

    const metricCards = metricRows.map(m => `
      <div style="background:var(--white);border-radius:var(--radius-7);padding:10px 12px;border-top:3px solid ${m.good ? 'var(--accent)':'var(--negative)'}">
        <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:var(--text-secondary);margin-bottom:4px">${m.label}</div>
        <div style="font-size:18px;font-weight:700;font-family:var(--font-mono);color:${m.good ? 'var(--accent-dark)':'var(--report-red)'}">${m.val}</div>
      </div>
    `).join('')

    const tickerPills = tickers.map((t, i) => `
      <span style="display:inline-block;background:rgba(var(--signal-positive-rgb),0.15);border:1px solid rgba(var(--signal-positive-rgb),0.3);border-radius:var(--radius-4);padding:2px 9px;font-size:11px;font-weight:700;color:var(--accent);font-family:var(--font-mono);margin-right:5px">${t} ${Math.round(weights[i]*100)}%</span>
    `).join('')

    const allocBars = tickers.map((t, i) => `
      <div style="margin-bottom:10px">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
          <span style="font-weight:700;font-family:var(--font-mono);font-size:13px">${t}</span>
          <span style="font-weight:700;color:var(--accent-dark);font-size:13px">${Math.round(weights[i]*100)}%</span>
        </div>
        <div style="height:6px;background:var(--report-track);border-radius:var(--radius-3)">
          <div style="height:6px;width:${weights[i]*100}%;background:${allocColors[i % allocColors.length]};border-radius:var(--radius-3)"></div>
        </div>
      </div>
    `).join('')

    const mcItems = [
      { label:'Median',        val: fmtD(mc.p50_final),                       color:'var(--accent-dark)' },
      { label:'Good year',     val: fmtD(mc.p95_final),                       color:'var(--accent)' },
      { label:'Bad year',      val: fmtD(mc.p5_final),                        color:'var(--report-red)' },
      { label:'Profit chance', val: `${Math.round(mc.prob_profit*100)}%`,      color: mc.prob_profit > 0.6 ? 'var(--accent-dark)':'var(--report-amber)' },
    ].map(m => `
      <div style="background:var(--report-card-alt);border-radius:var(--radius-5);padding:7px 10px">
        <div style="font-size:9px;color:var(--text-secondary);font-weight:700;text-transform:uppercase;margin-bottom:2px">${m.label}</div>
        <div style="font-size:14px;font-weight:700;color:${m.color};font-family:var(--font-mono)">${m.val}</div>
      </div>
    `).join('')

    const riskRows = [
      { label:'VaR 95%',      a: fmt(vc.var_pct),    b:`${fmtD(vc.var_dollar)}/day` },
      { label:'CVaR 95%',     a: fmt(vc.cvar_pct),   b:`${fmtD(vc.cvar_dollar)} avg` },
      { label:'Max Drawdown', a: fmt(d.max_drawdown), b:'peak to trough' },
      ...(ba ? [
        { label:'Beta',  a: fmtN(ba.beta),  b:'vs S&P 500' },
        { label:'Alpha', a: fmt(ba.alpha),  b:"Jensen's" },
      ] : []),
    ].map(r => `
      <div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid var(--report-bg)">
        <span style="color:var(--text-muted);font-size:12px">${r.label}</span>
        <div style="text-align:right">
          <div style="font-weight:700;color:var(--report-red);font-family:var(--font-mono);font-size:12px">${r.a}</div>
          <div style="font-size:10px;color:var(--text-secondary)">${r.b}</div>
        </div>
      </div>
    `).join('')

    const corrHeaders = corr.tickers.map(t => `<th style="font-size:10px;color:var(--text-secondary);font-family:var(--font-mono);text-align:center;padding:0 4px 6px">${t}</th>`).join('')
    const corrRows = corr.tickers.map((row, ri) => {
      const cells = corr.values[ri].map((val, ci) => {
        const bg = val >= 0.7 ? 'var(--report-heat-high)' : val >= 0.4 ? 'var(--report-heat-mid)' : 'var(--report-heat-low)'
        return `<td style="text-align:center;font-size:11px;font-family:var(--font-mono);font-weight:600;padding:6px 4px;border-radius:var(--radius-3);background:${bg}">${val.toFixed(2)}</td>`
      }).join('')
      return `<tr><td style="font-size:10px;color:var(--text-secondary);font-family:var(--font-mono);text-align:right;padding-right:8px">${row}</td>${cells}</tr>`
    }).join('')

    // Growth chart as SVG sparkline
    const cumVals = cum.values
    const minC = Math.min(...cumVals), maxC = Math.max(...cumVals)
    const rngC = maxC - minC || 0.01
    const svgW = 700, svgH = 120
    const points = cumVals.map((v, i) =>
      `${(i / (cumVals.length-1)) * svgW},${svgH - ((v - minC) / rngC) * svgH}`
    ).join(' ')
    const zeroY = svgH - ((0 - minC) / rngC) * svgH

    return `
      <div style="background:var(--card);padding:24px 32px;display:flex;justify-content:space-between;align-items:flex-start">
        <div>
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
            <div style="width:32px;height:32px;background:var(--accent);border-radius:var(--radius-7);display:flex;align-items:center;justify-content:center;font-weight:900;color:var(--card);font-size:16px">V</div>
            <div>
              <div style="font-size:20px;font-weight:700;color:var(--white);letter-spacing:-0.02em">varense</div>
              <div style="font-size:10px;color:var(--accent);letter-spacing:0.05em">variance, made sense of</div>
            </div>
          </div>
          <div>${tickerPills}</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:18px;font-weight:700;color:var(--white)">Varense Portfolio Report</div>
          <div style="font-size:12px;color:var(--text-secondary);margin-top:4px">${date}</div>
          <div style="font-size:11px;color:var(--text-muted);margin-top:5px;font-family:var(--font-mono)">${d.period.start} — ${d.period.end} · ${d.period.n_days} days</div>
        </div>
      </div>

      <div style="padding:22px 32px;display:flex;flex-direction:column;gap:18px;background:var(--report-bg)">

        <div>
          <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);margin-bottom:8px">Key Metrics</div>
          <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:7px">${metricCards}</div>
        </div>

        <div>
          <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);margin-bottom:8px">Portfolio Growth</div>
          <div style="background:var(--white);border-radius:var(--radius-sm);padding:16px">
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;font-size:12px">
              <span style="color:var(--text-secondary)">Cumulative return from ${d.period.start}</span>
              <span style="font-weight:700;color:${d.annualised_return > 0 ? 'var(--accent-dark)':'var(--report-red)'};font-family:var(--font-mono)">${fmt(cumVals[cumVals.length-1])} total</span>
            </div>
            <svg width="100%" height="${svgH}" viewBox="0 0 ${svgW} ${svgH}" preserveAspectRatio="none">
              <defs>
                <linearGradient id="sg1" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="var(--accent)" stop-opacity="0.2"/>
                  <stop offset="100%" stop-color="var(--accent)" stop-opacity="0"/>
                </linearGradient>
              </defs>
              ${zeroY > 0 && zeroY < svgH ? `<line x1="0" y1="${zeroY}" x2="${svgW}" y2="${zeroY}" stroke="var(--report-gridline)" stroke-dasharray="6,4" stroke-width="1"/>` : ''}
              <polygon points="${points} ${svgW},${svgH} 0,${svgH}" fill="url(#sg1)"/>
              <polyline points="${points}" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
            </svg>
          </div>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
          <div>
            <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);margin-bottom:8px">Allocation</div>
            <div style="background:var(--white);border-radius:var(--radius-sm);padding:16px">${allocBars}</div>
          </div>
          <div>
            <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);margin-bottom:8px">Monte Carlo — 1 Year</div>
            <div style="background:var(--white);border-radius:var(--radius-sm);padding:16px">
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">${mcItems}</div>
              <div style="margin-top:10px;font-size:10px;color:var(--text-secondary)">Based on ${mc.n_simulations?.toLocaleString() || '1,000'} simulations. Median: ${fmtD(mc.p50_final)} · Good: ${fmtD(mc.p95_final)} · Bad: ${fmtD(mc.p5_final)}</div>
            </div>
          </div>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
          <div>
            <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);margin-bottom:8px">Downside Risk</div>
            <div style="background:var(--white);border-radius:var(--radius-sm);padding:16px">${riskRows}</div>
          </div>
          <div>
            <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);margin-bottom:8px">Correlation Matrix</div>
            <div style="background:var(--white);border-radius:var(--radius-sm);padding:16px">
              <table style="width:100%;border-collapse:separate;border-spacing:3px">
                <thead><tr><th style="width:50px"></th>${corrHeaders}</tr></thead>
                <tbody>${corrRows}</tbody>
              </table>
            </div>
          </div>
        </div>

        <div style="text-align:center">
          <div style="display:inline-block;border:1px solid var(--report-border);border-radius:var(--radius-5);padding:5px 16px;font-size:11px;color:var(--text-secondary)">
            Generated by <strong style="color:var(--accent-dark)">varense</strong> · Free tier · ${date}
          </div>
        </div>

      </div>

      <div style="background:var(--card);padding:12px 32px;display:flex;justify-content:space-between;align-items:center">
        <div style="font-size:10px;color:var(--text-muted);line-height:1.5;max-width:460px">
          This report is for educational purposes only and does not constitute financial advice.
          Past performance does not guarantee future results. Always do your own research.
        </div>
        <div style="font-size:13px;font-weight:700;color:var(--accent)">varense.com</div>
      </div>
    `
  }

  const handlePrint = () => {
    const win = window.open('', '_blank', 'width=1100,height=900')
    const reportHTML = buildReportHTML()
    win.document.write(`
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8" />
          <title>Varense Portfolio Report</title>
          <link rel="preconnect" href="https://fonts.googleapis.com">
          <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
          <style>
            /* This popup is a separate document (window.open + document.write) that
               does NOT inherit the app's index.css, so the design tokens it uses below
               are duplicated here — values must stay in sync with index.css by hand. */
            :root {
              --accent-dark: #2d6a4f; --accent: #52b788; --signal-positive-soft: #74c69d;
              --chart-mint: #95d5b2; --accent-light: #b7e4c7; --report-text-dark: #1a2a1a;
              --card: #1e2420; --white: #ffffff; --text-muted: #5a7a5a; --text-secondary: #8aaa8a;
              --negative: #e05c5c; --report-red: #c0392b; --report-bg: #f0f4f0;
              --report-track: #e8ede8; --report-gridline: #e0e8e0; --report-heat-high: #fdd;
              --report-heat-mid: #fef3d0; --report-heat-low: #e8f5ee; --report-card-alt: #f8faf8;
              --report-card-soft: #f5faf5; --report-amber: #b7791f; --report-border: #c8d8c8;
              --black-rgb: 0,0,0; --signal-positive-rgb: 82,183,136;
              --font-mono: monospace; --font-report: Inter, sans-serif;
              --radius-14: 14px; --radius-3: 3px; --radius-4: 4px; --radius-5: 5px;
              --radius-7: 7px; --radius-sm: 8px; --radius-full: 50%;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { background: var(--report-bg); font-family: var(--font-report); }

            @media print {
              #modal-overlay { display: none !important; }
              @page { margin: 0; size: A4 landscape; }
              body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
            }
          </style>
        </head>
        <body>
          <!-- Instruction modal -->
          <div id="modal-overlay" style="position:fixed;inset:0;background:rgba(var(--black-rgb),0.55);z-index:9999;display:flex;align-items:center;justify-content:center;">
            <div style="background:var(--white);border-radius:var(--radius-14);padding:32px 36px;max-width:460px;width:90%;box-shadow:0 24px 64px rgba(var(--black-rgb),0.3);font-family:var(--font-report);">
              <div style="display:flex;align-items:center;gap:10;margin-bottom:20px;">
                <div style="width:34px;height:34px;background:var(--accent-dark);border-radius:var(--radius-sm);display:flex;align-items:center;justify-content:center;font-weight:900;color:var(--white);font-size:16px;flex-shrink:0;">V</div>
                <div>
                  <div style="font-size:16px;font-weight:700;color:var(--report-text-dark);">Save your PDF report</div>
                  <div style="font-size:12px;color:var(--text-secondary);margin-top:1px;">Follow these steps for a full-colour PDF</div>
                </div>
              </div>
              <div style="display:flex;flex-direction:column;gap:12px;margin-bottom:24px;">
                <div style="display:flex;align-items:flex-start;gap:12px;padding:12px 14px;background:var(--report-card-soft);border-radius:var(--radius-sm);border-left:3px solid var(--accent);">
                  <div style="width:22px;height:22px;background:var(--accent-dark);border-radius:var(--radius-full);display:flex;align-items:center;justify-content:center;color:var(--white);font-size:11px;font-weight:700;flex-shrink:0;margin-top:1px;">1</div>
                  <div>
                    <div style="font-size:13px;font-weight:600;color:var(--report-text-dark);">Click <strong>"More settings"</strong> in the print dialog</div>
                    <div style="font-size:11px;color:var(--text-secondary);margin-top:2px;">Found at the bottom of the left panel</div>
                  </div>
                </div>
                <div style="display:flex;align-items:flex-start;gap:12px;padding:12px 14px;background:var(--report-card-soft);border-radius:var(--radius-sm);border-left:3px solid var(--accent);">
                  <div style="width:22px;height:22px;background:var(--accent-dark);border-radius:var(--radius-full);display:flex;align-items:center;justify-content:center;color:var(--white);font-size:11px;font-weight:700;flex-shrink:0;margin-top:1px;">2</div>
                  <div>
                    <div style="font-size:13px;font-weight:600;color:var(--report-text-dark);">Enable <strong>"Background graphics"</strong></div>
                    <div style="font-size:11px;color:var(--text-secondary);margin-top:2px;">This preserves all colours and dark backgrounds</div>
                  </div>
                </div>
                <div style="display:flex;align-items:flex-start;gap:12px;padding:12px 14px;background:var(--report-card-soft);border-radius:var(--radius-sm);border-left:3px solid var(--accent);">
                  <div style="width:22px;height:22px;background:var(--accent-dark);border-radius:var(--radius-full);display:flex;align-items:center;justify-content:center;color:var(--white);font-size:11px;font-weight:700;flex-shrink:0;margin-top:1px;">3</div>
                  <div>
                    <div style="font-size:13px;font-weight:600;color:var(--report-text-dark);">Set layout to <strong>"Landscape"</strong></div>
                    <div style="font-size:11px;color:var(--text-secondary);margin-top:2px;">Fits the report width perfectly on the page</div>
                  </div>
                </div>
                <div style="display:flex;align-items:flex-start;gap:12px;padding:12px 14px;background:var(--report-card-soft);border-radius:var(--radius-sm);border-left:3px solid var(--accent);">
                  <div style="width:22px;height:22px;background:var(--accent-dark);border-radius:var(--radius-full);display:flex;align-items:center;justify-content:center;color:var(--white);font-size:11px;font-weight:700;flex-shrink:0;margin-top:1px;">4</div>
                  <div>
                    <div style="font-size:13px;font-weight:600;color:var(--report-text-dark);">Set destination to <strong>"Save as PDF"</strong></div>
                    <div style="font-size:11px;color:var(--text-secondary);margin-top:2px;">Then click Save</div>
                  </div>
                </div>
              </div>
              <button onclick="document.getElementById('modal-overlay').style.display='none';window.print();"
                style="width:100%;padding:12px;background:var(--accent-dark);color:var(--white);border:none;border-radius:var(--radius-sm);font-size:14px;font-weight:700;cursor:pointer;letter-spacing:0.03em;">
                Got it — open print dialog
              </button>
            </div>
          </div>

          <div>${reportHTML}</div>
          <script>
            // No auto-print — user clicks the button in the modal
          </script>
        </body>
      </html>
    `)
    win.document.close()
  }

  return (
    <>
      <button
        onClick={handlePrint}
        style={{
          display:'flex', alignItems:'center', gap:6,
          fontSize:11, fontWeight:600, padding:'6px 14px',
          borderRadius:'var(--radius-6)', border:'1px solid var(--border-light)',
          background:'rgba(var(--white-rgb),0.5)',
          color:'var(--accent-dark)', cursor:'pointer',
          transition:'all 0.15s', letterSpacing:'0.03em',
        }}
        onMouseEnter={e => { e.currentTarget.style.background='rgba(var(--accent-dark-rgb),0.1)'; e.currentTarget.style.borderColor='var(--accent-dark)' }}
        onMouseLeave={e => { e.currentTarget.style.background='rgba(var(--white-rgb),0.5)'; e.currentTarget.style.borderColor='var(--border-light)' }}
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
        Export PDF
      </button>
    </>
  )
}