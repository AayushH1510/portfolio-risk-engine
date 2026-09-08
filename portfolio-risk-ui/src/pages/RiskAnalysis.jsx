import {
  LineChart, Line, AreaChart, Area,
  XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import ReturnHistogram from '../components/ReturnHistogram'
import StressTest from '../components/StressTest'
import MetricTooltip from '../components/MetricTooltip'
import InsightBox from '../components/InsightBox'

const fmt  = v => `${(v * 100).toFixed(2)}%`
const fmtS = v => `${(v * 100).toFixed(1)}%`

const AXIS = { fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }
const TIP  = {
  background: 'var(--surface-elevated)', border: 'var(--border-emphasis)',
  fontSize: 11, padding: '6px 10px',
}

function MiniChart({ data, dataKey, color, filled, refVal }) {
  return (
    <ResponsiveContainer width="100%" height={72}>
      {filled ? (
        <AreaChart data={data} margin={{ top: 4, right: 2, bottom: 0, left: 2 }}>
          <XAxis dataKey="date" hide />
          <YAxis hide />
          {refVal != null && <ReferenceLine y={refVal} stroke="rgba(var(--text-primary-rgb),0.1)" strokeDasharray="3 3" />}
          <Tooltip formatter={v => fmtS(v)} contentStyle={TIP} itemStyle={{ color }} />
          <Area type="monotone" dataKey={dataKey} stroke={color} strokeWidth={1.5} fill={color} fillOpacity={0.08} dot={false} />
        </AreaChart>
      ) : (
        <LineChart data={data} margin={{ top: 4, right: 2, bottom: 0, left: 2 }}>
          <XAxis dataKey="date" hide />
          <YAxis hide />
          {refVal != null && <ReferenceLine y={refVal} stroke="rgba(var(--text-primary-rgb),0.1)" strokeDasharray="3 3" />}
          <Tooltip formatter={v => v?.toFixed(2)} contentStyle={TIP} itemStyle={{ color }} />
          <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={1.5} dot={false} />
        </LineChart>
      )}
    </ResponsiveContainer>
  )
}

function MetricPill({ label, value, color, sub }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', gap: 3,
      padding: '10px 14px',
      background: 'var(--surface-elevated)',
      border: 'var(--border-faint)',
    }}>
      <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)' }}>
        {label}
      </div>
      <div style={{ fontSize: 15, fontWeight: 700, fontFamily: 'var(--font-mono)', color: color || 'var(--text-primary)' }}>
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 9, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{sub}</div>
      )}
    </div>
  )
}

// Highest and lowest pairwise correlation in the matrix, and which two
// tickers each belongs to — only ever looks at the upper triangle (i < j)
// so the always-1.0 diagonal (a ticker against itself) and duplicate
// mirrored pairs never enter into it.
function pairwiseCorrStats({ tickers, values }) {
  let maxVal = -Infinity, maxPair = null
  let minVal = Infinity, minPair = null
  for (let i = 0; i < tickers.length; i++) {
    for (let j = i + 1; j < tickers.length; j++) {
      const v = values[i][j]
      if (v > maxVal) { maxVal = v; maxPair = [tickers[i], tickers[j]] }
      if (v < minVal) { minVal = v; minPair = [tickers[i], tickers[j]] }
    }
  }
  return { maxVal, maxPair, minVal, minPair }
}

function CorrLegend() {
  return (
    <div style={{ display: 'flex', gap: 10, fontSize: 10, color: 'var(--text-muted)' }}>
      {[
        ['≥ 0.8 High',  'rgba(var(--signal-negative-rgb),0.3)'],
        ['0.5–0.8 Med', 'rgba(var(--signal-caution-rgb),0.3)'],
        ['< 0.2 Low',   'rgba(var(--signal-positive-rgb),0.25)'],
      ].map(([l, c]) => (
        <span key={l} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <div style={{ width: 8, height: 8, background: c }}/>
          {l}
        </span>
      ))}
    </div>
  )
}

function CorrMatrix({ corr }) {
  const n = corr.tickers.length
  const cellPad  = n >= 5 ? '5px 3px' : n >= 4 ? '7px 4px' : '9px 6px'
  const cellFont = n >= 5 ? 9         : n >= 4 ? 10        : 11
  const headFont = n >= 5 ? 9         : 10
  const rowWidth = n >= 5 ? 32        : n >= 4 ? 38        : 44

  const cellColor = v => {
    if (v >= 0.8) return { bg: 'rgba(var(--signal-negative-rgb),0.25)',   text: 'var(--signal-negative)' }
    if (v >= 0.5) return { bg: 'rgba(var(--signal-caution-rgb),0.2)',   text: 'var(--signal-caution)' }
    if (v >= 0.2) return { bg: 'rgba(var(--text-primary-rgb),0.06)', text: 'rgba(var(--text-primary-rgb),0.7)' }
    return            { bg: 'rgba(var(--signal-positive-rgb),0.15)',  text: 'var(--signal-positive)' }
  }

  return (
    <div>
      <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: n >= 5 ? 2 : 4 }}>
        <thead>
          <tr>
            <th style={{ width: rowWidth }}/>
            {corr.tickers.map(t => (
              <th key={t} style={{
                fontSize: headFont, fontWeight: 700, color: 'var(--text-muted)',
                fontFamily: 'var(--font-mono)', padding: '0 2px 6px', textAlign: 'center',
              }}>{t}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {corr.tickers.map((row, ri) => (
            <tr key={row}>
              <td style={{
                fontSize: headFont, fontWeight: 700, color: 'var(--text-muted)',
                fontFamily: 'var(--font-mono)', paddingRight: 6, textAlign: 'right',
              }}>{row}</td>
              {corr.values[ri].map((val, ci) => {
                const { bg, text } = cellColor(Math.abs(val))
                return (
                  <td key={ci} style={{
                    textAlign: 'center', fontSize: cellFont, fontFamily: 'var(--font-mono)',
                    fontWeight: 700, padding: cellPad,
                    background: bg, color: text,
                  }}>
                    {val.toFixed(2)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function RiskAnalysis({ data, tickers, weights, portfolioValue, onTickerClick }) {
  if (!data) return null

  const {
    rolling_volatility: rv, rolling_sharpe: rs,
    correlation_matrix: corr, var_cvar, var_cvar_99,
    annualised_volatility: vol, sharpe_ratio: sharpe,
    portfolio_returns, treynor_ratio: treynor, information_ratio: infoRatio,
  } = data

  const rvData = rv.dates.map((d, i) => ({ date: d.slice(5), vol: rv.values[i] }))
  const rsData = rs.dates.map((d, i) => ({ date: d.slice(5), sharpe: rs.values[i] }))

  const recentVol      = rv.values[rv.values.length - 1]
  const recentSharpe   = rs.values[rs.values.length - 1]
  const riskRising     = recentVol > vol
  const sharpeImproving = recentSharpe > sharpe

  const toneColor = { good: 'var(--signal-positive)', warning: 'var(--signal-caution)', bad: 'var(--signal-negative)' }
  const treynorTone   = treynor   > 1   ? 'good' : treynor   > 0 ? 'warning' : 'bad'
  const infoRatioTone = infoRatio > 0.5 ? 'good' : infoRatio > 0 ? 'warning' : 'bad'

  // Corr matrix "why this matters" copy is derived entirely from the real
  // pairwise data below, not a generic verdict — see pairwiseCorrStats.
  // The "lowest pair" sentence only makes sense with 3+ tickers (a 2-ticker
  // portfolio has exactly one pair, nothing to contrast it against), and
  // only when that lowest pair is meaningfully below the highest one —
  // otherwise it'd just be restating the same pair/value a second time.
  const corrStats          = pairwiseCorrStats(corr)
  const hasContrastingPair = corr.tickers.length >= 3 && (corrStats.maxVal - corrStats.minVal) >= 0.15

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, height: '100%', overflowY: 'auto' }}>

      <InsightBox
        label="Why risk-adjusted metrics matter"
        text="Two portfolios can have the same return but very different risk. These metrics show whether you're being compensated fairly for the risk you're taking, not just how much you made."
      />

      {/* Row 1: Rolling charts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>

        <div className="card" style={{ padding: '14px 16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
            <div>
              <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)', marginBottom: 4 }}>
                <MetricTooltip metricKey="rolling_volatility">Rolling volatility</MetricTooltip>
              </div>
              <div style={{ fontSize: 20, fontWeight: 700, fontFamily: 'var(--font-mono)', color: riskRising ? 'var(--signal-caution)' : 'var(--signal-positive)' }}>
                {fmtS(recentVol)}
              </div>
            </div>
            <div style={{
              fontSize: 10, fontWeight: 600, padding: '3px 8px',
              background: riskRising ? 'rgba(var(--signal-caution-rgb),0.12)' : 'rgba(var(--signal-positive-rgb),0.12)',
              color: riskRising ? 'var(--signal-caution)' : 'var(--signal-positive)',
              border: `1px solid ${riskRising ? 'rgba(var(--signal-caution-rgb),0.3)' : 'rgba(var(--signal-positive-rgb),0.3)'}`,
              marginTop: 2,
            }}>
              {riskRising ? '↑ Rising' : '↓ Falling'}
            </div>
          </div>
          <MiniChart data={rvData} dataKey="vol" color="var(--signal-caution)" filled refVal={vol} />
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6 }}>
            Avg <span style={{ color: 'var(--text-secondary)', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>{fmtS(vol)}</span>
            <span style={{ marginLeft: 8, opacity: 0.5 }}>dashed line = long-run average</span>
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4, lineHeight: 1.5 }}>
            {riskRising
              ? 'Volatility is currently above its long-run average, meaning risk has been rising and returns have felt less stable recently.'
              : 'Volatility is currently below its long-run average, meaning risk has been easing and returns have felt more stable recently.'}
          </div>
        </div>

        <div className="card" style={{ padding: '14px 16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
            <div>
              <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)', marginBottom: 4 }}>
                <MetricTooltip metricKey="rolling_sharpe">Rolling Sharpe</MetricTooltip>
              </div>
              <div style={{ fontSize: 20, fontWeight: 700, fontFamily: 'var(--font-mono)', color: recentSharpe > 1 ? 'var(--signal-positive)' : recentSharpe > 0 ? 'var(--signal-caution)' : 'var(--signal-negative)' }}>
                {recentSharpe.toFixed(2)}
              </div>
            </div>
            <div style={{
              fontSize: 10, fontWeight: 600, padding: '3px 8px',
              background: sharpeImproving ? 'rgba(var(--signal-positive-rgb),0.12)' : 'rgba(var(--signal-caution-rgb),0.12)',
              color: sharpeImproving ? 'var(--signal-positive)' : 'var(--signal-caution)',
              border: `1px solid ${sharpeImproving ? 'rgba(var(--signal-positive-rgb),0.3)' : 'rgba(var(--signal-caution-rgb),0.3)'}`,
              marginTop: 2,
            }}>
              {sharpeImproving ? '↑ Improving' : '↓ Weakening'}
            </div>
          </div>
          <MiniChart data={rsData} dataKey="sharpe" color="var(--signal-positive)" filled={false} refVal={sharpe} />
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6 }}>
            Avg <span style={{ color: 'var(--text-secondary)', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>{sharpe.toFixed(2)}</span>
            <span style={{ marginLeft: 8, opacity: 0.5 }}>above 1.0 is strong</span>
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4, lineHeight: 1.5 }}>
            {sharpeImproving
              ? "Your risk-adjusted returns have been improving lately, better than this portfolio's own typical performance."
              : "Your risk-adjusted returns have been weakening lately, below this portfolio's own typical performance."}
          </div>
          {recentSharpe < 0 && (
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4, lineHeight: 1.5 }}>
              Recent returns haven't been enough to justify the risk taken, regardless of the trend.
            </div>
          )}
        </div>
      </div>

      {/* Row 1.5: Benchmark-relative ratios */}
      {(treynor != null || infoRatio != null) && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10 }}>
          {treynor != null && (
            <MetricPill
              label={<MetricTooltip metricKey="treynor_ratio">Treynor ratio</MetricTooltip>}
              value={treynor.toFixed(2)}
              color={toneColor[treynorTone]}
              sub="return per unit of market risk"
            />
          )}
          {infoRatio != null && (
            <MetricPill
              label={<MetricTooltip metricKey="information_ratio">Information ratio</MetricTooltip>}
              value={infoRatio.toFixed(2)}
              color={toneColor[infoRatioTone]}
              sub="active return per unit of tracking risk"
            />
          )}
        </div>
      )}

      {/* Row 2: VaR + Correlation */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>

        {/* Downside risk — 95% and 99% side by side */}
        <div className="card" style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column' }}>
          <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)', marginBottom: 12 }}>
            <MetricTooltip metricKey="downside_risk">Downside risk</MetricTooltip>
          </div>

          {/* Confidence level headers */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 6 }}>
            {[
              { label: '95% confidence', color: 'var(--signal-negative)', borderColor: 'rgba(var(--signal-negative-rgb),0.19)' },
              { label: '99% confidence', color: 'var(--signal-negative-strong)', borderColor: 'rgba(var(--signal-negative-strong-rgb),0.19)' },
            ].map(({ label, color, borderColor }) => (
              <div key={label} style={{
                fontSize: 9, fontWeight: 700, textTransform: 'uppercase',
                letterSpacing: '0.07em', color, textAlign: 'center',
                padding: '3px 0', borderBottom: `1px solid ${borderColor}`,
              }}>
                {label}
              </div>
            ))}
          </div>

          {/* VaR row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 6 }}>
            <MetricPill
              label={<MetricTooltip metricKey="var_95">VaR</MetricTooltip>}
              value={`−${fmt(Math.abs(var_cvar.var_pct))}`}
              color="var(--signal-negative)"
              sub={`$${Math.abs(var_cvar.var_dollar).toFixed(0)} per day`}
            />
            <MetricPill
              label={<MetricTooltip metricKey="var_99">VaR</MetricTooltip>}
              value={`−${fmt(Math.abs(var_cvar_99?.var_pct ?? 0))}`}
              color="var(--signal-negative-strong)"
              sub={`$${Math.abs(var_cvar_99?.var_dollar ?? 0).toFixed(0)} per day`}
            />
          </div>

          {/* CVaR row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 12 }}>
            <MetricPill
              label={<MetricTooltip metricKey="cvar_95">CVaR (avg tail)</MetricTooltip>}
              value={`−${fmt(Math.abs(var_cvar.cvar_pct))}`}
              color="var(--signal-negative)"
              sub={`$${Math.abs(var_cvar.cvar_dollar).toFixed(0)} avg`}
            />
            <MetricPill
              label={<MetricTooltip metricKey="cvar_99">CVaR (avg tail)</MetricTooltip>}
              value={`−${fmt(Math.abs(var_cvar_99?.cvar_pct ?? 0))}`}
              color="var(--signal-negative-strong)"
              sub={`$${Math.abs(var_cvar_99?.cvar_dollar ?? 0).toFixed(0)} avg`}
            />
          </div>

          <div style={{
            fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6,
            padding: '10px 12px', marginTop: 'auto',
            background: 'rgba(var(--signal-negative-rgb),0.05)',
            border: '1px solid rgba(var(--signal-negative-rgb),0.12)',
          }}>
            On a typical bad day, you could lose around{' '}
            <strong style={{ color: 'var(--signal-negative)' }}>${Math.abs(var_cvar.var_dollar).toFixed(0)}</strong>.
            {' '}On a truly rare, severe day, closer to{' '}
            <strong style={{ color: 'var(--signal-negative-strong)' }}>${Math.abs(var_cvar_99?.cvar_dollar ?? 0).toFixed(0)}</strong>.
          </div>
        </div>

        {/* display:flex + marginTop:'auto' on the summary box below (matching
            the Downside Risk card above) bottom-anchors both cards' summary
            text at the same height regardless of how tall their own content
            is — a 3-ticker correlation table is much shorter than the VaR/CVaR
            grid opposite it, so without this the two summary boxes drift to
            different vertical positions. */}
        <div className="card" style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 14 }}>
            <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)' }}>
              <MetricTooltip metricKey="correlation_matrix">Correlation matrix</MetricTooltip>
            </div>
            <CorrLegend />
          </div>
          <CorrMatrix corr={corr} />
          <div style={{
            fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6,
            padding: '10px 12px', marginTop: 'auto',
            background: 'rgba(var(--text-primary-rgb),0.02)',
            border: '1px solid rgba(var(--text-primary-rgb),0.05)',
          }}>
            {corrStats.maxVal >= 0.7 ? (
              <>{corrStats.maxPair[0]} and {corrStats.maxPair[1]} move together the most in your portfolio, correlated at <strong style={{ color: 'rgba(var(--text-primary-rgb),0.65)' }}>{corrStats.maxVal.toFixed(2)}</strong>. When you hold both, you're not getting much real diversification from having two positions instead of one.</>
            ) : corrStats.maxVal >= 0.4 ? (
              <>{corrStats.maxPair[0]} and {corrStats.maxPair[1]} are your most closely linked holdings, correlated at <strong style={{ color: 'rgba(var(--text-primary-rgb),0.65)' }}>{corrStats.maxVal.toFixed(2)}</strong>, a moderate relationship. They still add some diversification, just not as much as fully independent assets would.</>
            ) : (
              <>Even your most closely linked pair, {corrStats.maxPair[0]} and {corrStats.maxPair[1]}, only correlates at <strong style={{ color: 'rgba(var(--text-primary-rgb),0.65)' }}>{corrStats.maxVal.toFixed(2)}</strong>. Your holdings are behaving largely independently of each other.</>
            )}
            {hasContrastingPair && (
              <> {corrStats.minPair[0]} and {corrStats.minPair[1]} are your most independent pair, correlated at only <strong style={{ color: 'rgba(var(--text-primary-rgb),0.65)' }}>{corrStats.minVal.toFixed(2)}</strong>, they're doing the most real diversification work in this portfolio.</>
            )}
            {' '}The more your holdings move together, the more a single bad day for one can mean a bad day for all of them, real diversification comes from genuinely independent assets, not just from owning more tickers.
          </div>
        </div>
      </div>

      {/* Row 3: Returns distribution */}
      {portfolio_returns && (
        <>
          <InsightBox
            label="Why the daily returns distribution matters"
            text="This chart plots every daily return your portfolio produced over the selected period. The x-axis is the size of a single day's gain or loss, from a big loss on the left to a big gain on the right. The y-axis is how many days landed in that range - a tall bar means a lot of days moved by roughly that amount. A narrow, tight cluster near the middle means your day-to-day performance has been consistent and predictable. A wide, lumpy spread, especially with bars reaching far to either side, means bigger day-to-day swings and occasional sharp outlier days - a portfolio with these 'fat tails' can look calm most days while still carrying serious downside risk."
          />
          <div className="card" style={{ padding: '14px 16px' }}>
            <div style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', textTransform: 'uppercase', letterSpacing: 'var(--tracking-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)', marginBottom: 14 }}>
              <MetricTooltip metricKey="daily_returns_distribution">Daily returns distribution</MetricTooltip>
            </div>
            <ReturnHistogram
              portfolioReturns={portfolio_returns}
              varPct={var_cvar.var_pct}
              cvarPct={var_cvar.cvar_pct}
              varDollar={var_cvar.var_dollar}
              cvarDollar={var_cvar.cvar_dollar}
              confidence={var_cvar.confidence}
            />
          </div>
        </>
      )}

      {/* Row 4: Stress test */}
      <InsightBox
        label="Why stress testing matters"
        text="Backtested returns show how your portfolio performs in normal conditions. Stress tests show what happens when markets panic, the scenario most investors are least prepared for."
      />
      <StressTest tickers={tickers} weights={weights} portfolioValue={portfolioValue} />

    </div>
  )
}