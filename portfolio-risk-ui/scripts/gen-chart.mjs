// Generates a plausible-looking (not real) price-series path for
// ProductPlate's growth-chart replica — irregular volatility, one
// asymmetric drawdown, no repeating rhythm. Deterministic LCG (same
// technique as components/landing/visuals.jsx's seededPaths()) so re-running
// this script reproduces the same output; the result gets hand-pasted into
// ProductPlate.jsx as a literal path, not computed at runtime.

function makeRng(seed) {
  let s = seed
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff
    return s / 0x7fffffff
  }
}

// Portfolio line: x 0..800, y 192 (start, bottom) -> 14 (end, top), SVG y
// increases downward so "up" means decreasing y. Volatility clusters via a
// slowly-varying multiplier; a deliberate drawdown (sharp decline, slower
// choppier recovery) added mid-series.
function genSeries(seed, { n, startY, endY, baseStep, drawdown }) {
  const rng = makeRng(seed)
  const raw = [0]
  let volMult = 1
  for (let i = 1; i <= n; i++) {
    // Volatility regime drifts slowly, occasionally jumps (clustering).
    volMult += (rng() - 0.5) * 0.2
    if (rng() < 0.1) volMult *= 0.4 + rng() * 1.8 // regime shift
    volMult = Math.max(0.25, Math.min(2.6, volMult))
    const step = (rng() - 0.5) * 2 * baseStep * volMult
    raw.push(raw[i - 1] + step)
  }

  // Detrend the raw walk (endpoints -> 0), then lay the desired start/end
  // trend back on top — a Brownian-bridge construction so the noise shape
  // is preserved but the series is pinned to exact start/end values.
  const rawSpan = raw[n] - raw[0]
  const bridged = raw.map((v, i) => v - raw[0] - (i / n) * rawSpan)
  let series = bridged.map((v, i) => startY + ((endY - startY) * i) / n + v)

  // Deliberate drawdown: sharp rise (price drop) over a short window,
  // slower/uneven recovery over a longer one — not a symmetric V.
  if (drawdown) {
    const { peakIdx, depth, declineLen, recoverLen } = drawdown
    for (let i = 0; i <= n; i++) {
      let bump = 0
      if (i >= peakIdx - declineLen && i < peakIdx) {
        const t = (i - (peakIdx - declineLen)) / declineLen
        bump = depth * Math.pow(t, 1.6) // accelerating decline
      } else if (i >= peakIdx && i <= peakIdx + recoverLen) {
        const t = (i - peakIdx) / recoverLen
        // uneven recovery: fast-ish then stalls then finishes, via a
        // slightly wobbled ease-out rather than a clean curve
        const ease = 1 - Math.pow(1 - t, 1.7)
        const wobble = Math.sin(t * 7.3) * depth * 0.06 * (1 - t)
        bump = depth * (1 - ease) + wobble
      }
      series[i] += bump
    }
  }
  return series
}

function toPath(xs, ys, close, closeY) {
  let d = `M${xs[0]},${ys[0].toFixed(1)}`
  for (let i = 1; i < xs.length; i++) d += ` L${xs[i]},${ys[i].toFixed(1)}`
  if (close) d += ` L${xs[xs.length - 1]},${closeY} L${xs[0]},${closeY} Z`
  return d
}

const N = 44
const xs = Array.from({ length: N + 1 }, (_, i) => Math.round((800 / N) * i))

const portfolio = genSeries(20260919, {
  n: N, startY: 192, endY: 14, baseStep: 9,
  // depth is a gross bump; the underlying upward trend (~20 units of
  // natural y-decrease over a 5-step decline window) partially cancels it,
  // so depth needs real headroom above the visible dip size actually wanted.
  drawdown: { peakIdx: 24, depth: 70, declineLen: 5, recoverLen: 12 },
})

// Benchmark: correlated but distinct — its own noise draw (different seed),
// shallower overall trend (ends higher/less-improved than the portfolio,
// matching "beat the S&P 500"), and its own smaller dip roughly in the same
// era as the portfolio's (markets move together) but offset and shallower.
const benchmark = genSeries(20260920, {
  n: N, startY: 200, endY: 116, baseStep: 6.5,
  drawdown: { peakIdx: 27, depth: 34, declineLen: 6, recoverLen: 9 },
})

console.log('PORTFOLIO min/max:', Math.min(...portfolio).toFixed(1), Math.max(...portfolio).toFixed(1))
console.log('BENCHMARK min/max:', Math.min(...benchmark).toFixed(1), Math.max(...benchmark).toFixed(1))
console.log()
console.log('AREA (fill):')
console.log(toPath(xs, portfolio, true, 230))
console.log()
console.log('LINE (portfolio stroke):')
console.log(toPath(xs, portfolio, false))
console.log()
console.log('LINE (benchmark dashed):')
console.log(toPath(xs, benchmark, false))
