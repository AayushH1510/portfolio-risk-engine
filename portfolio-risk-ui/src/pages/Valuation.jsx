import { useEffect, useState } from 'react'
import axios from 'axios'
import { errorMessage } from '../lib/errorMessage'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const fmtPct  = v => v != null ? `${(v * 100).toFixed(1)}%` : 'N/A'

const COLUMNS = [
  { label: '#',           tip: null },
  { label: 'Ticker',      tip: null },
  { label: 'P/S',         tip: 'Price-to-Sales ratio. How much investors pay per $1 of revenue. Lower = cheaper relative to sales. High-growth companies often have high P/S.' },
  { label: 'EV/EBITDA',   tip: 'Enterprise Value divided by earnings before interest, tax, depreciation and amortisation. A measure of overall company value vs operating profit. Lower = cheaper. Negative means the company has negative EBITDA (unprofitable at operating level).' },
  { label: 'Gross Margin',tip: 'Revenue kept after subtracting the direct cost of goods sold. Higher = stronger pricing power and more money left for R&D, sales, and profit. Software companies often exceed 70%. Manufacturers are typically 20–40%.' },
  { label: 'Rev Growth',  tip: 'Year-over-year revenue growth. How fast the company is growing its top line. Green = strong growth (15%+). Red = shrinking revenue - a serious warning sign.' },
  { label: 'Mkt Cap',     tip: 'Market capitalisation - total value of all shares outstanding. Gives context for scale: mega-cap ($1T+), large-cap ($10B+), mid-cap ($2B+), small-cap (under $2B).' },
  { label: 'V/G Score',   tip: 'Value/Growth Score = P/S ÷ Revenue Growth %. Lower is better - it means you are getting more growth per dollar of valuation. Think of it as a revenue-based PEG ratio. A score below 0.15 is excellent.' },
]

function ColHeader({ label, tip }) {
  return (
    <th style={{
      padding: '10px 14px',
      textAlign: label === '#' || label === 'Ticker' ? 'left' : 'right',
      fontSize: 10, fontWeight: 700, letterSpacing: '0.06em',
      textTransform: 'uppercase', color: 'var(--text-muted)',
      whiteSpace: 'nowrap', userSelect: 'none',
    }}>
      {tip ? (
        <span
          title={`${label}\n\n${tip}`}
          style={{
            display: 'inline-flex', alignItems: 'center', gap: 4,
            cursor: 'help', borderBottom: '1px dashed rgba(var(--signal-positive-rgb),0.3)',
            paddingBottom: 1,
          }}
        >
          {label}
          <svg width="10" height="10" viewBox="0 0 12 12" fill="none" style={{ opacity: 0.45, flexShrink: 0 }}>
            <circle cx="6" cy="6" r="5" stroke="var(--signal-positive)" strokeWidth="1.2"/>
            <text x="6" y="9.5" textAnchor="middle" fontSize="7" fill="var(--signal-positive)" fontWeight="700">?</text>
          </svg>
        </span>
      ) : label}
    </th>
  )
}
const fmtX    = v => v != null ? `${v.toFixed(1)}x` : 'N/A'
const fmtCap  = v => {
  if (!v) return 'N/A'
  if (v >= 1e12) return `$${(v/1e12).toFixed(1)}T`
  if (v >= 1e9)  return `$${(v/1e9).toFixed(1)}B`
  return `$${(v/1e6).toFixed(0)}M`
}

const SEV_COLOR = { high: 'var(--signal-negative)', medium: 'var(--signal-caution)', low: 'var(--signal-positive-soft)' }
const SEV_BG    = { high: 'rgba(var(--signal-negative-rgb),0.13)', medium: 'rgba(var(--signal-caution-rgb),0.13)', low: 'rgba(var(--signal-positive-soft-rgb),0.13)' }
const CAT_LABEL = { accounting: 'Accounting', concentration: 'Leverage / Ownership', competitive: 'Competitive' }

function ScoreBar({ value, max, color }) {
  if (value == null) return <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>N/A</span>
  const pct = Math.min((value / max) * 100, 100)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ flex: 1, height: 4, background: 'rgba(var(--text-primary-rgb),0.07)', overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: color }} />
      </div>
      <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', minWidth: 40, textAlign: 'right' }}>
        {fmtPct(value)}
      </span>
    </div>
  )
}

export default function Valuation({ tickers }) {
  const [data, setData]     = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState(null)
  const [selected, setSelected] = useState(null)

  useEffect(() => {
    if (!tickers?.length) return
    setLoading(true)
    setError(null)
    setSelected(null)
    axios.get(`${API}/api/fundamentals?tickers=${tickers.join(',')}`)
      .then(r => { setData(r.data.tickers); setSelected(r.data.tickers[0]?.ticker) })
      .catch(e => setError(errorMessage(e, 'Failed to load fundamentals')))
      .finally(() => setLoading(false))
  }, [tickers.join(',')])

  if (loading) return (
    <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:'100%', gap:12, color:'var(--text-muted)' }}>
      <svg className="spin" width="20" height="20" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="10" stroke="var(--signal-positive)" strokeWidth="3" strokeDasharray="40 20"/>
      </svg>
      Loading fundamental data...
    </div>
  )

  if (error) return (
    <div style={{ color:'var(--signal-negative)', fontSize:13, padding:16 }}>{error}</div>
  )

  if (!data) return (
    <div style={{ color:'var(--text-muted)', fontSize:13, padding:16 }}>
      Run an analysis first to see valuation data.
    </div>
  )

  const selectedStock = data.find(d => d.ticker === selected)

  // Ranking helpers
  const rank = (arr, key, lowerBetter = false) => {
    const sorted = [...arr].filter(d => d[key] != null).sort((a,b) => lowerBetter ? a[key]-b[key] : b[key]-a[key])
    return sorted.map((d,i) => ({ ticker: d.ticker, rank: i+1 }))
  }

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:14, height:'100%', overflowY:'auto' }}>

      {/* Header */}
      <div>
        <div style={{ fontSize:'var(--text-caption)', fontWeight:'var(--weight-medium)', textTransform:'uppercase', letterSpacing:'var(--tracking-caption)', color:'var(--text-muted)', fontFamily:'var(--font-primary)', marginBottom:2 }}>
          Relative Valuation
        </div>
        <div style={{ fontSize:11, color:'var(--text-muted)' }}>
          Ranked by Value/Growth score - lowest = most growth per dollar of valuation
        </div>
      </div>

      {/* Valuation table */}
      <div className="card" style={{ padding:0, overflow:'hidden', flexShrink:0 }}>
        <table style={{ width:'100%', borderCollapse:'collapse', fontSize:12 }}>
          <thead>
            <tr style={{ borderBottom:'1px solid rgba(var(--text-primary-rgb),0.06)' }}>
              {COLUMNS.map(col => (
                <ColHeader key={col.label} label={col.label} tip={col.tip} />
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((stock, i) => {
              const isSelected = stock.ticker === selected
              const hasError   = !!stock.error
              const flagCount  = stock.flags?.length || 0

              return (
                <tr
                  key={stock.ticker}
                  onClick={() => !hasError && setSelected(stock.ticker)}
                  style={{
                    borderBottom:'1px solid rgba(var(--text-primary-rgb),0.04)',
                    background: isSelected ? 'rgba(var(--signal-positive-rgb),0.07)' : 'transparent',
                    cursor: hasError ? 'default' : 'pointer',
                    transition:'background 0.12s',
                  }}
                  onMouseEnter={e => !isSelected && (e.currentTarget.style.background = 'rgba(var(--text-primary-rgb),0.02)')}
                  onMouseLeave={e => !isSelected && (e.currentTarget.style.background = 'transparent')}
                >
                  {/* Rank */}
                  <td style={{ padding:'12px 14px' }}>
                    <div style={{
                      width:22, height:22, fontSize:11, fontWeight:700,
                      display:'flex', alignItems:'center', justifyContent:'center',
                      background: i===0 ? 'var(--signal-positive)' : i===1 ? 'rgba(var(--signal-positive-rgb),0.3)' : 'rgba(var(--text-primary-rgb),0.07)',
                      color: i===0 ? 'var(--surface-canvas)' : 'var(--text-primary)',
                    }}>{i+1}</div>
                  </td>

                  {/* Ticker + name */}
                  <td style={{ padding:'12px 14px' }}>
                    <div style={{ fontWeight:700, fontFamily:'var(--font-mono)', color:'var(--text-primary)', fontSize:13 }}>
                      {stock.ticker}
                      {flagCount > 0 && (
                        <span style={{
                          marginLeft:6, fontSize:9, fontWeight:700, padding:'1px 5px',
                          background:'rgba(var(--signal-negative-rgb),0.2)', color:'var(--signal-negative)',
                        }}>{flagCount} flag{flagCount>1?'s':''}</span>
                      )}
                    </div>
                    <div style={{ fontSize:10, color:'var(--text-muted)', marginTop:1, maxWidth:140, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                      {stock.name || stock.industry || '-'}
                    </div>
                  </td>

                  {hasError ? (
                    <td colSpan={6} style={{ padding:'12px 14px', color:'var(--signal-negative)', fontSize:11 }}>
                      {stock.error || 'Could not load data'}
                    </td>
                  ) : (
                    <>
                      <td style={{ padding:'12px 14px', textAlign:'right', fontFamily:'var(--font-mono)', color:'var(--text-primary)' }}>{fmtX(stock.ps_ratio)}</td>
                      <td style={{ padding:'12px 14px', textAlign:'right', fontFamily:'var(--font-mono)', color:'var(--text-primary)' }}>{fmtX(stock.ev_ebitda)}</td>
                      <td style={{ padding:'12px 14px', textAlign:'right', fontFamily:'var(--font-mono)', color: stock.gross_margin > 0.5 ? 'var(--signal-positive)' : 'var(--text-primary)' }}>
                        {fmtPct(stock.gross_margin)}
                      </td>
                      <td style={{ padding:'12px 14px', textAlign:'right', fontFamily:'var(--font-mono)',
                        color: stock.rev_growth > 0.15 ? 'var(--signal-positive)' : stock.rev_growth < 0 ? 'var(--signal-negative)' : 'var(--text-primary)' }}>
                        {stock.rev_growth != null ? `${stock.rev_growth > 0 ? '+' : ''}${(stock.rev_growth*100).toFixed(1)}%` : 'N/A'}
                      </td>
                      <td style={{ padding:'12px 14px', textAlign:'right', fontFamily:'var(--font-mono)', color:'var(--text-muted)' }}>{fmtCap(stock.market_cap)}</td>
                      <td style={{ padding:'12px 14px', textAlign:'right' }}>
                        {stock.vg_score != null ? (
                          <span style={{
                            fontFamily:'var(--font-mono)', fontWeight:700, fontSize:13,
                            color: stock.vg_score < 0.15 ? 'var(--signal-positive)' : stock.vg_score < 0.4 ? 'var(--signal-caution)' : 'var(--signal-negative)',
                          }}>
                            {stock.vg_score.toFixed(2)}
                            <span style={{ fontSize:9, marginLeft:4, color:'var(--text-muted)', fontWeight:400 }}>
                              {i===0 ? '· Best' : i===data.length-1 ? '· Worst' : '· Mid'}
                            </span>
                          </span>
                        ) : <span style={{ color:'var(--text-muted)', fontSize:11 }}>N/A</span>}
                      </td>
                    </>
                  )}
                </tr>
              )
            })}
          </tbody>
        </table>

        {/* Score explanation */}
        <div style={{ padding:'10px 14px', borderTop:'1px solid rgba(var(--text-primary-rgb),0.04)', fontSize:10, color:'var(--text-muted)', lineHeight:1.6 }}>
          <strong style={{ color:'var(--text-secondary)' }}>Value/Growth Score</strong> = P/S ÷ Revenue Growth %.
          Lower = more growth per dollar of valuation. Click a row to see the full risk assessment.
        </div>
      </div>

      {/* Risk assessment panel */}
      {selectedStock && !selectedStock.error && (
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12, flexShrink:0 }}>

          {/* Metrics breakdown */}
          <div className="card" style={{ padding:'14px 16px' }}>
            <div style={{ fontSize:'var(--text-caption)', fontWeight:'var(--weight-medium)', textTransform:'uppercase', letterSpacing:'var(--tracking-caption)', color:'var(--text-muted)', fontFamily:'var(--font-primary)', marginBottom:12 }}>
              {selectedStock.ticker} - Key metrics
            </div>
            {[
              { label:'P/S Ratio',      val: fmtX(selectedStock.ps_ratio),       note:'Price per $1 of revenue' },
              { label:'EV/EBITDA',      val: fmtX(selectedStock.ev_ebitda),      note:'Enterprise value vs earnings' },
              { label:'Gross Margin',   val: fmtPct(selectedStock.gross_margin), note:'Revenue kept after COGS' },
              { label:'Net Margin',     val: fmtPct(selectedStock.profit_margin),note:'Revenue kept after all costs' },
              { label:'Revenue Growth', val: selectedStock.rev_growth != null ? `${selectedStock.rev_growth>0?'+':''}${(selectedStock.rev_growth*100).toFixed(1)}%` : 'N/A', note:'Year over year' },
              { label:'Debt/Equity',    val: selectedStock.debt_equity != null ? `${selectedStock.debt_equity.toFixed(0)}%` : 'N/A', note:'Leverage level' },
              { label:'Current Ratio',  val: selectedStock.current_ratio != null ? `${selectedStock.current_ratio.toFixed(2)}x` : 'N/A', note:'Short-term liquidity' },
              { label:'ROE',            val: fmtPct(selectedStock.roe),           note:'Return on equity' },
              { label:'Beta',           val: selectedStock.beta != null ? selectedStock.beta.toFixed(2) : 'N/A', note:'Market sensitivity' },
              { label:'Market Cap',     val: fmtCap(selectedStock.market_cap),   note:'Total company value' },
            ].map(m => (
              <div key={m.label} style={{ display:'flex', justifyContent:'space-between', alignItems:'center', padding:'5px 0', borderBottom:'1px solid rgba(var(--text-primary-rgb),0.03)' }}>
                <div>
                  <div style={{ fontSize:11, color:'var(--text-secondary)' }}>{m.label}</div>
                  <div style={{ fontSize:10, color:'var(--text-muted)' }}>{m.note}</div>
                </div>
                <div style={{ fontSize:12, fontFamily:'var(--font-mono)', fontWeight:600, color:'var(--text-primary)' }}>{m.val}</div>
              </div>
            ))}
          </div>

          {/* Risk flags + positives */}
          <div style={{ display:'flex', flexDirection:'column', gap:10 }}>

            {/* Risk flags */}
            <div className="card" style={{ padding:'14px 16px', flex: selectedStock.flags?.length > 0 ? 1 : 0 }}>
              <div style={{ fontSize:'var(--text-caption)', fontWeight:'var(--weight-medium)', textTransform:'uppercase', letterSpacing:'var(--tracking-caption)', color:'var(--text-muted)', fontFamily:'var(--font-primary)', marginBottom:10 }}>
                Risk flags ({selectedStock.flags?.length || 0})
              </div>
              {selectedStock.flags?.length === 0 && (
                <div style={{ fontSize:12, color:'var(--signal-positive)' }}>No significant risk flags detected.</div>
              )}
              {selectedStock.flags?.map((flag, i) => (
                <div key={i} style={{
                  marginBottom:8, padding:'8px 10px',
                  background:'rgba(var(--text-primary-rgb),0.03)',
                  borderLeft:`3px solid ${SEV_COLOR[flag.severity] || 'var(--grey-generic)'}`,
                }}>
                  <div style={{ display:'flex', justifyContent:'space-between', marginBottom:3 }}>
                    <span style={{ fontSize:11, fontWeight:700, color:'var(--text-primary)' }}>{flag.title}</span>
                    <span style={{
                      fontSize:9, fontWeight:700, padding:'1px 6px',
                      background: SEV_BG[flag.severity] || 'var(--surface-elevated)',
                      color: SEV_COLOR[flag.severity],
                      textTransform:'uppercase', letterSpacing:'0.05em',
                    }}>{flag.severity}</span>
                  </div>
                  <div style={{ fontSize:11, color:'var(--text-muted)', lineHeight:1.5 }}>{flag.detail}</div>
                  <div style={{ fontSize:10, color:'var(--text-muted)', marginTop:3, opacity:0.6 }}>{CAT_LABEL[flag.category]}</div>
                </div>
              ))}
            </div>

            {/* Positives */}
            {selectedStock.positives?.length > 0 && (
              <div className="card" style={{ padding:'14px 16px' }}>
                <div style={{ fontSize:'var(--text-caption)', fontWeight:'var(--weight-medium)', textTransform:'uppercase', letterSpacing:'var(--tracking-caption)', color:'var(--text-muted)', fontFamily:'var(--font-primary)', marginBottom:10 }}>
                  Strengths
                </div>
                {selectedStock.positives.map((p, i) => (
                  <div key={i} style={{
                    display:'flex', gap:8, alignItems:'flex-start',
                    fontSize:11, color:'var(--text-secondary)', marginBottom:7, lineHeight:1.5,
                  }}>
                    <div style={{ width:6, height:6, background:'var(--signal-positive)', flexShrink:0, marginTop:4 }}/>
                    {p}
                  </div>
                ))}
              </div>
            )}

            {/* Sector / industry */}
            <div className="card" style={{ padding:'12px 14px' }}>
              <div style={{ display:'flex', gap:12 }}>
                <div>
                  <div style={{ fontSize:10, color:'var(--text-muted)', marginBottom:2 }}>Sector</div>
                  <div style={{ fontSize:12, color:'var(--text-primary)', fontWeight:500 }}>{selectedStock.sector || '-'}</div>
                </div>
                <div style={{ borderLeft:'1px solid rgba(var(--text-primary-rgb),0.06)', paddingLeft:12 }}>
                  <div style={{ fontSize:10, color:'var(--text-muted)', marginBottom:2 }}>Industry</div>
                  <div style={{ fontSize:12, color:'var(--text-primary)', fontWeight:500 }}>{selectedStock.industry || '-'}</div>
                </div>
              </div>
            </div>

          </div>
        </div>
      )}

      {/* Disclaimer */}
      <div style={{ fontSize:10, color:'var(--text-muted)', paddingBottom:8, lineHeight:1.6 }}>
        Fundamental data sourced from Finnhub. Risk flags are rule-based and for educational purposes only - not financial advice. Always verify figures with primary sources.
      </div>

    </div>
  )
}