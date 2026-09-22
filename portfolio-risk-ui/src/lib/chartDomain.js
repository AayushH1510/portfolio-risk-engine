// Recharts' default YAxis domain rounds OUT to "nice" tick values (e.g.
// -10%/30% for data that actually runs -2% to 27%) — reasonable in
// isolation, but on a short chart this leaves the line occupying a thin
// band in the middle of a lot of dead vertical space. Computes the true
// min/max across every series actually plotted and pads by a fixed
// fraction of the range rather than rounding to a "nice" number, so the
// line reliably fills most of the available height without ever touching
// the plot edges. Falls back to Recharts' own 'auto' when there's no
// finite range to measure. Originally Dashboard.jsx-only (its growth/
// by-holding charts); Comparison.jsx's Monte Carlo chart reuses it rather
// than shipping a second copy of the same function.
export function computeYDomain(rows, keys, padFraction = 0.1) {
  let min = Infinity, max = -Infinity
  for (const row of rows) {
    for (const key of keys) {
      const v = row[key]
      if (typeof v !== 'number' || !isFinite(v)) continue
      if (v < min) min = v
      if (v > max) max = v
    }
  }
  if (!isFinite(min) || !isFinite(max)) return ['auto', 'auto']
  const pad = (max - min) * padFraction || Math.abs(max || 1) * padFraction
  return [min - pad, max + pad]
}
