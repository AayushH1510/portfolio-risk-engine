// Benchmark tickers offered in the Sidebar's benchmark dropdown, and the
// single source of truth for turning a benchmark ticker back into the
// human-readable name shown in chart legends, insight-box copy, and
// exports — every consumer reads data.benchmark (echoed back by
// /api/analyse*) rather than guessing from Sidebar state, so labels always
// match whatever ticker actually produced the numbers on screen.
//
// `article` is the leading "the " a sentence needs before the label to read
// naturally — named indices/funds take it ("the S&P 500"), commodities and
// bond baskets don't ("Gold", not "the Gold"). Set it per new entry (e.g.
// Oil: '') rather than assuming — it doesn't follow from the ticker itself.
export const BENCHMARKS = [
  { ticker: 'SPY',  label: 'S&P 500',                  article: 'the ' },
  { ticker: 'QQQ',  label: 'Nasdaq 100',                article: 'the ' },
  { ticker: 'VTI',  label: 'Total US Market',           article: 'the ' },
  { ticker: 'VXUS', label: 'Total International Market', article: 'the ' },
  { ticker: 'GLD',  label: 'Gold',                      article: '' },
  { ticker: 'SLV',  label: 'Silver',                    article: '' },
  { ticker: 'USO',  label: 'Oil',                        article: '' },
  { ticker: 'AGG',  label: 'US Bonds',                  article: '' },
]

export function getBenchmarkLabel(ticker) {
  return BENCHMARKS.find(b => b.ticker === ticker)?.label || ticker || 'Benchmark'
}

// For sentence construction — "Beat {phrase} by 8.5%" — returns the label
// prefixed with "the " only when the benchmark actually takes one.
export function getBenchmarkPhrase(ticker) {
  const b = BENCHMARKS.find(b => b.ticker === ticker)
  return b ? `${b.article}${b.label}` : getBenchmarkLabel(ticker)
}
