import { useEffect, useState } from 'react'
import axios from 'axios'
import { errorMessage } from '../lib/errorMessage'
import InsightBox from '../components/InsightBox'
import MetricTooltip from '../components/MetricTooltip'
import useCompactViewport from '../hooks/useCompactViewport'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const fmtPct  = v => v != null ? `${(v * 100).toFixed(1)}%` : 'N/A'

// metricKey looks up the real explanation in content/metricExplanations.js
// via the shared MetricTooltip component (same one Dashboard/RiskAnalysis
// use) — this table used to carry its own inline tip text through a native
// `title` attribute instead, which is invisible on touch devices and
// inconsistent enough elsewhere that it read as broken. stock_beta is a
// distinct key from the portfolio-level "beta" already in that dictionary —
// this one describes a single ticker's own market sensitivity.
const COLUMNS = [
  { label: '#',           metricKey: null },
  { label: 'Ticker',      metricKey: null },
  { label: 'P/S',         metricKey: 'ps_ratio' },
  { label: 'EV/EBITDA',   metricKey: 'ev_ebitda' },
  { label: 'Gross Margin',metricKey: 'gross_margin' },
  { label: 'Rev Growth',  metricKey: 'rev_growth' },
  { label: 'Mkt Cap',     metricKey: 'market_cap' },
  { label: 'V/G Score',   metricKey: 'vg_score' },
]

// Sticky Ticker column (compact viewport only — see the table wrapper's own
// comment below for why this is gated rather than unconditional). Shared by
// the header <th> and every body <td> so the solid-background/hairline
// treatment can't drift between them. `tint` layers a translucent overlay
// (the row's own selected/hover colour, or none for the header) UNDER an
// opaque base via a stacked background — the actual "solid" part of "solid
// background so scrolled cells don't show through": the header and resting
// rows pass no tint (plain var(--surface-card)), a selected or hovered row
// passes its own rgba so the sticky cell still reflects row state instead
// of reading as a dead strip while the rest of the row is tinted.
const STICKY_COL_STYLE = (tint) => ({
  position: 'sticky',
  left: 0,
  zIndex: 2,
  background: tint ? `linear-gradient(${tint}, ${tint}), var(--surface-card)` : 'var(--surface-card)',
  borderRight: '1px solid var(--line-hairline)',
})

function ColHeader({ label, metricKey, sticky }) {
  return (
    <th style={{
      padding: '10px 14px',
      textAlign: label === '#' || label === 'Ticker' ? 'left' : 'right',
      fontSize: 10, fontWeight: 700, letterSpacing: '0.06em',
      textTransform: 'uppercase', color: 'var(--text-muted)',
      whiteSpace: 'nowrap', userSelect: 'none',
      ...(sticky ? STICKY_COL_STYLE() : {}),
    }}>
      {metricKey ? <MetricTooltip metricKey={metricKey}>{label}</MetricTooltip> : label}
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

// Clickable ticker symbol — opens the Stock Drawer, distinct from clicking
// elsewhere in the row (which selects it for the Key Metrics panel below).
// e.stopPropagation() keeps those two actions independent: clicking the
// symbol never also re-selects the row. Styling mirrors Sidebar's
// TickerLabel (dotted underline, hover arrow) so the "this opens the
// drawer" affordance reads the same everywhere it appears.
function TickerLink({ ticker, onClick }) {
  const [hovered, setHovered] = useState(false)
  return (
    <span
      onClick={e => { e.stopPropagation(); onClick?.(ticker) }}
      title="Click to view stock details"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        fontWeight: 700, fontFamily: 'var(--font-mono)', fontSize: 13,
        color: hovered ? 'var(--signal-positive)' : 'var(--text-primary)',
        cursor: 'pointer',
        display: 'inline-flex', alignItems: 'center', gap: 3,
        paddingBottom: 1,
        borderBottom: hovered
          ? '1.5px solid var(--signal-positive)'
          : '1.5px dashed rgba(var(--signal-positive-rgb),0.4)',
        transition: 'color 0.15s, border-color 0.15s',
        userSelect: 'none',
      }}
    >
      {ticker}
      <svg width="8" height="8" viewBox="0 0 8 8" fill="none" style={{ opacity: hovered ? 1 : 0.5, transition: 'opacity 0.15s' }}>
        <path d="M1.5 4H6.5M4 1.5L6.5 4L4 6.5" stroke="var(--signal-positive)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    </span>
  )
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

export default function Valuation({ tickers, onTickerClick }) {
  const [data, setData]     = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState(null)
  const [selected, setSelected] = useState(null)
  // Same hooks-before-early-return placement as every other tab this pass
  // touched (Comparison.jsx/MonteCarlo.jsx/etc.) — the loading/error/no-data
  // returns below aren't hit on every render, so this has to run before them.
  const isCompactViewport = useCompactViewport()

  useEffect(() => {
    if (!tickers?.length) return
    setLoading(true)
    setError(null)
    setSelected(null)
    axios.get(`${API}/api/fundamentals?tickers=${tickers.join(',')}`)
      .then(r => {
        const list = r.data.tickers
        setData(list)
        // Prefer the first ticker with real fundamentals to select by
        // default — a fund/ETF's detail panel would just be a wall of
        // N/A, so don't lead with one if a real stock is also in the mix.
        const firstWithData = list.find(d => !d.error && !d.is_fund)
        setSelected((firstWithData || list[0])?.ticker)
      })
      // Same neutral fallback as useAnalysis / useComparison — errorMessage()
      // still surfaces the backend's real `detail` when there is one; this
      // line only shows for network-level failures that carry no response.
      .catch(e => setError(errorMessage(e, 'Something went wrong loading data. Please try again in a moment.')))
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

  // A fund/ETF has no P/S ratio, gross margin, etc. of its own - those are
  // company-level metrics that belong to its underlying holdings, not the
  // wrapper (see api.py: is_fund). A portfolio made entirely of funds gets
  // the same kind of honest explanation Sector Exposure already shows for
  // this exact case, instead of a table full of N/A.
  const allFunds = data.length > 0 && data.every(d => !d.error && d.is_fund)

  return (
    <div className="tab-scroll-root" style={{ display:'flex', flexDirection:'column', gap:14, height:'100%', overflowY:'auto' }}>

      {/* Header */}
      <div>
        <div style={{ fontSize:'var(--text-caption)', fontWeight:'var(--weight-medium)', textTransform:'uppercase', letterSpacing:'var(--tracking-caption)', color:'var(--text-muted)', fontFamily:'var(--font-primary)', marginBottom:2 }}>
          Relative Valuation
        </div>
        <div style={{ fontSize:11, color:'var(--text-muted)' }}>
          Ranked by Value/Growth score - lowest = most growth per dollar of valuation
        </div>
      </div>

      {allFunds ? (
        // priority="primary": the one thing on the tab when it renders —
        // mutually exclusive with the "Why fundamentals" box below, so
        // both get the identical always-open-but-toggleable treatment
        // rather than one having the affordance and the other not.
        <InsightBox
          tone="neutral"
          label="Not applicable"
          priority="primary"
          text="Your portfolio is made up of funds and ETFs, which each hold many companies rather than being one themselves - so per-company valuation ratios like P/S, EV/EBITDA, and gross margin don't apply to the fund wrapper itself. Valuation works best with individual stock holdings."
        />
      ) : (
      <>
      {/* Why fundamentals matter — only shown when there's a real table to introduce;
          the all-funds branch above already carries its own honest explanation. */}
      <InsightBox
        label="Why fundamentals, not just price"
        compact
        priority="primary"
        text="Price tells you what the market thinks. Fundamentals tell you why. This ranks each holding on what it earns, how fast it's growing, and how much debt it's carrying, then surfaces the risk flags and strengths automatically so you don't have to dig for them. See whether the numbers back up the price."
      />

      {/* Valuation table. At compact viewport: the '#' rank column drops
          (pure row index once the table can't show every column at once —
          the ticker still identifies the row, and the V/G Score column's
          own "· Best"/"· Worst" suffix already carries the one thing the
          rank badge said beyond plain order: which row is the top pick),
          and the Ticker column goes sticky so a reader who's scrolled the
          table horizontally can still tell which row they're looking at —
          see STICKY_COL_STYLE's own comment above for the background
          mechanism. Both are gated on isCompactViewport, not unconditional:
          at desktop the table fits without scrolling (§6 of this audit:
          8 columns need ≥600-650px, comfortably inside 1280+), so nothing
          here would ever visibly activate there — gating keeps desktop
          pixel-identical rather than relying on that inertness alone. */}
      <div className="card" style={{ padding:0, flexShrink:0 }}>
        <div style={{ overflowX: 'auto' }}>
        <table style={{ width:'100%', borderCollapse:'collapse', fontSize:12 }}>
          <thead>
            <tr style={{ borderBottom:'1px solid rgba(var(--text-primary-rgb),0.06)' }}>
              {COLUMNS.filter(col => !(isCompactViewport && col.label === '#')).map(col => (
                <ColHeader key={col.label} label={col.label} metricKey={col.metricKey} sticky={isCompactViewport && col.label === 'Ticker'} />
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((stock, i) => {
              const isSelected = stock.ticker === selected
              const hasError   = !!stock.error
              const isFund     = !hasError && stock.is_fund
              const flagCount  = stock.flags?.length || 0

              return (
                <tr
                  key={stock.ticker}
                  onClick={() => !hasError && !isFund && setSelected(stock.ticker)}
                  style={{
                    borderBottom:'1px solid rgba(var(--text-primary-rgb),0.04)',
                    background: isSelected ? 'rgba(var(--signal-positive-rgb),0.07)' : 'transparent',
                    cursor: hasError || isFund ? 'default' : 'pointer',
                    transition:'background 0.12s',
                  }}
                  // The sticky Ticker cell (when present) needs its own
                  // background kept in sync with the row's hover state too
                  // — a plain background on the <tr> doesn't paint through
                  // a sticky descendant's own explicit background, so the
                  // row's imperative hover handlers reach into the sticky
                  // cell by class and update it the same way.
                  onMouseEnter={e => {
                    if (isSelected) return
                    e.currentTarget.style.background = 'rgba(var(--text-primary-rgb),0.02)'
                    const sticky = e.currentTarget.querySelector('.valuation-sticky-col')
                    if (sticky) sticky.style.background = `linear-gradient(rgba(var(--text-primary-rgb),0.02), rgba(var(--text-primary-rgb),0.02)), var(--surface-card)`
                  }}
                  onMouseLeave={e => {
                    if (isSelected) return
                    e.currentTarget.style.background = 'transparent'
                    const sticky = e.currentTarget.querySelector('.valuation-sticky-col')
                    if (sticky) sticky.style.background = 'var(--surface-card)'
                  }}
                >
                  {/* Rank */}
                  {!isCompactViewport && (
                    <td style={{ padding:'12px 14px' }}>
                      <div style={{
                        width:22, height:22, fontSize:11, fontWeight:700,
                        display:'flex', alignItems:'center', justifyContent:'center',
                        background: i===0 ? 'var(--signal-positive)' : i===1 ? 'rgba(var(--signal-positive-rgb),0.3)' : 'rgba(var(--text-primary-rgb),0.07)',
                        color: i===0 ? 'var(--surface-canvas)' : 'var(--text-primary)',
                      }}>{i+1}</div>
                    </td>
                  )}

                  {/* Ticker + name */}
                  <td
                    className={isCompactViewport ? 'valuation-sticky-col' : undefined}
                    style={{
                      padding:'12px 14px',
                      ...(isCompactViewport ? STICKY_COL_STYLE(isSelected ? 'rgba(var(--signal-positive-rgb),0.07)' : null) : {}),
                    }}
                  >
                    <div>
                      <TickerLink ticker={stock.ticker} onClick={onTickerClick} />
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
                  ) : isFund ? (
                    <td colSpan={6} style={{ padding:'12px 14px', color:'var(--text-muted)', fontSize:11 }}>
                      Fund/ETF - per-company valuation ratios don't apply
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
        </div>

        {/* Score explanation */}
        <div style={{ padding:'10px 14px', borderTop:'1px solid rgba(var(--text-primary-rgb),0.04)', fontSize:10, color:'var(--text-muted)', lineHeight:1.6 }}>
          <strong style={{ color:'var(--text-secondary)' }}>Value/Growth Score</strong> = P/S ÷ Revenue Growth %.
          Lower = more growth per dollar of valuation. Click a row to see the full risk assessment.
        </div>
      </div>

      {/* Risk assessment panel. .valuation-risk-grid: gridTemplateColumns
          lives in that CSS class, not inline, so the compact-viewport
          override can reach it (the usual inline-always-wins reason every
          other grid fix in this audit has hit). This is the source of the
          tab's own 337px/17px-past-320 overflow — the metrics-breakdown
          card and the flags/positives column squeezed into a fixed 1fr 1fr
          row with no flex-wrap, confirmed by measurement before fixing. */}
      {selectedStock && !selectedStock.error && !selectedStock.is_fund && (
        <div className="valuation-risk-grid" style={{ display:'grid', gap:12, flexShrink:0 }}>

          {/* Metrics breakdown */}
          <div className="card" style={{ padding:'14px 16px' }}>
            <div style={{ fontSize:'var(--text-caption)', fontWeight:'var(--weight-medium)', textTransform:'uppercase', letterSpacing:'var(--tracking-caption)', color:'var(--text-muted)', fontFamily:'var(--font-primary)', marginBottom:12 }}>
              {selectedStock.ticker} - Key metrics
            </div>
            {[
              { label:'P/S Ratio',      metricKey:'ps_ratio',     val: fmtX(selectedStock.ps_ratio),       note:'Price per $1 of revenue' },
              { label:'EV/EBITDA',      metricKey:'ev_ebitda',    val: fmtX(selectedStock.ev_ebitda),      note:'Enterprise value vs earnings' },
              { label:'Gross Margin',   metricKey:'gross_margin', val: fmtPct(selectedStock.gross_margin), note:'Revenue kept after COGS' },
              { label:'Net Margin',     metricKey:'profit_margin',val: fmtPct(selectedStock.profit_margin),note:'Revenue kept after all costs' },
              { label:'Revenue Growth', metricKey:'rev_growth',   val: selectedStock.rev_growth != null ? `${selectedStock.rev_growth>0?'+':''}${(selectedStock.rev_growth*100).toFixed(1)}%` : 'N/A', note:'Year over year' },
              { label:'Debt/Equity',    metricKey:'debt_equity',  val: selectedStock.debt_equity != null ? `${selectedStock.debt_equity.toFixed(0)}%` : 'N/A', note:'Leverage level' },
              { label:'Current Ratio',  metricKey:'current_ratio',val: selectedStock.current_ratio != null ? `${selectedStock.current_ratio.toFixed(2)}x` : 'N/A', note:'Short-term liquidity' },
              { label:'ROE',            metricKey:'roe',          val: fmtPct(selectedStock.roe),           note:'Return on equity' },
              { label:'Beta',           metricKey:'stock_beta',   val: selectedStock.beta != null ? selectedStock.beta.toFixed(2) : 'N/A', note:'Market sensitivity' },
              { label:'Market Cap',     metricKey:'market_cap',   val: fmtCap(selectedStock.market_cap),   note:'Total company value' },
            ].map(m => (
              <div key={m.label} style={{ display:'flex', justifyContent:'space-between', alignItems:'center', padding:'5px 0', borderBottom:'1px solid rgba(var(--text-primary-rgb),0.03)' }}>
                <div>
                  <div style={{ fontSize:11, color:'var(--text-secondary)' }}><MetricTooltip metricKey={m.metricKey}>{m.label}</MetricTooltip></div>
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
      </>
      )}

    </div>
  )
}