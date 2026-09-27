// Buckets a portfolio's tickers into real sector rows plus two named
// exception rows, so Sector Exposure's bars always sum to the portfolio's
// full weight instead of a failed ticker just vanishing from the
// aggregation with nothing visible left behind (the actual bug this fixes:
// a 33%-weighted stock — a transient Finnhub timeout, not a real data gap —
// disappeared with the remaining bars silently summing to 67%, and no way
// to tell that apart from a portfolio that's genuinely 33% ETFs).
//
//   - "Data unavailable — TICKER[, TICKER...]" (kind: 'unavailable') — a
//     transient failure (timeout, 429, or Finnhub's own auth error — see
//     api.py's `transient` field). Worth retrying once; see
//     hasTransientFailure below.
//   - "ETFs & unclassified — TICKER[, TICKER...]" (kind: 'unclassified') —
//     a fund/ETF with no sector of its own, or a ticker Finnhub has
//     positively confirmed doesn't exist (`transient: false`). Neither is
//     going to resolve on retry, so these are never retried.
//
// A ticker missing from the response entirely (shouldn't happen with the
// current backend, but not something this should assume away) is treated
// as recoverable — bucketed with the transient failures, not the permanent
// ones — since silently and permanently hiding it again on nothing more
// than an unexpected response shape would be the same class of bug this
// whole thing exists to fix.
export function buildSectorRows(fundamentalsList, tickers, weights) {
  const byTicker = {}
  fundamentalsList.forEach(t => { byTicker[t.ticker] = t })

  const weightBySector  = {}
  const tickersBySector = {}
  const unavailableTickers  = []
  const unclassifiedTickers = []
  let unavailableWeight  = 0
  let unclassifiedWeight = 0

  tickers.forEach((tk, i) => {
    const entry  = byTicker[tk.toUpperCase()]
    const weight = weights[i] ?? 0

    if (!entry) {
      unavailableTickers.push(tk)
      unavailableWeight += weight
      return
    }
    if (entry.error) {
      if (entry.transient === false) {
        unclassifiedTickers.push(tk)
        unclassifiedWeight += weight
      } else {
        unavailableTickers.push(tk)
        unavailableWeight += weight
      }
      return
    }
    if (!entry.sector || entry.sector === 'Unknown') {
      unclassifiedTickers.push(tk)
      unclassifiedWeight += weight
      return
    }
    weightBySector[entry.sector]  = (weightBySector[entry.sector] || 0) + weight
    tickersBySector[entry.sector] = [...(tickersBySector[entry.sector] || []), tk]
  })

  const rows = Object.entries(weightBySector).map(([sector, weight]) => ({
    sector, weight, tickers: tickersBySector[sector], kind: 'sector',
  }))

  if (unavailableTickers.length) {
    rows.push({
      sector: `Data unavailable — ${unavailableTickers.join(', ')}`,
      weight: unavailableWeight,
      tickers: unavailableTickers,
      kind: 'unavailable',
    })
  }
  if (unclassifiedTickers.length) {
    rows.push({
      sector: `ETFs & unclassified — ${unclassifiedTickers.join(', ')}`,
      weight: unclassifiedWeight,
      tickers: unclassifiedTickers,
      kind: 'unclassified',
    })
  }

  return rows.sort((a, b) => b.weight - a.weight)
}

// One automatic retry (App.jsx) is only worth scheduling when at least one
// ticker's failure was tagged transient — a permanent one (an ETF, a
// confirmed-invalid ticker) will look exactly the same 60 seconds later.
export function hasTransientFailure(fundamentalsList) {
  return fundamentalsList.some(t => t.error && t.transient === true)
}
