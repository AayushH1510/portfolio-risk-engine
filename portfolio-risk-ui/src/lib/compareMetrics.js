// Single source of truth for "which side wins" a head-to-head metric on the
// Compare tab. Every metric's direction is declared explicitly here, once —
// no caller infers or re-derives whether higher or lower is better.
//
// Each entry's `score(v)` maps that metric's raw stored value to a number
// where HIGHER SCORE ALWAYS MEANS BETTER. Callers never compare raw values
// against each other directly (no `lowerBetter` flag applied to a possibly-
// signed value) — they compare scores with a plain `>`.
//
// This is what correctly handles the metrics stored as negative numbers
// representing a loss (max_drawdown, var_pct, cvar_pct): a -4.7% drawdown is
// a smaller loss than a -6.6% drawdown, and -0.047 > -0.066 is already true,
// so these metrics' score is just the raw value, unchanged — no abs() or
// sign flip needed. The bug this file fixes came from wrapping these in a
// naive "lowerBetter: true" flag applied to the raw signed value, which
// treated "more negative" (a bigger loss) as "lower" and therefore "better"
// — the opposite of correct.
export const METRIC_DIRECTIONS = {
  return:     { label: 'Return',     score: v => v },   // higher raw value = better
  volatility: { label: 'Volatility', score: v => -v },  // lower raw value = better (always >= 0)
  sharpe:     { label: 'Sharpe',     score: v => v },   // higher raw value = better
  sortino:    { label: 'Sortino',    score: v => v },   // higher raw value = better
  drawdown:   { label: 'Drawdown',   score: v => v },   // negative; less negative (smaller loss) = higher raw value = better
  var95:      { label: 'VaR 95%',    score: v => v },   // negative; smaller loss = higher raw value = better
  cvar95:     { label: 'CVaR 95%',   score: v => v },   // negative; smaller loss = higher raw value = better
  beta:       { label: 'Beta',       score: v => -v },  // lower raw value (less systematic risk) = better
  alpha:      { label: 'Alpha',      score: v => v },   // higher raw value = better
}

// Returns 'A', 'B', or null (missing data or an exact tie — neither side is
// declared a winner).
export function winner(aVal, bVal, metricKey) {
  const direction = METRIC_DIRECTIONS[metricKey]
  if (!direction) throw new Error(`compareMetrics: unknown metric key "${metricKey}"`)
  if (typeof aVal !== 'number' || typeof bVal !== 'number' || Number.isNaN(aVal) || Number.isNaN(bVal)) return null
  const aScore = direction.score(aVal)
  const bScore = direction.score(bVal)
  if (aScore === bScore) return null
  return aScore > bScore ? 'A' : 'B'
}
