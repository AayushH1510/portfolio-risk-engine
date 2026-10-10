// Print-clean inline-SVG chart builders shared by both PDF reports. Every
// function returns a self-contained <svg>...</svg> string sized to a
// viewBox, not tied to any particular on-screen pixel size — callers set
// width/height via CSS, the viewBox keeps the drawing proportional, same
// convention the Summary report's own growth sparkline already used.
// These are deliberately simple — a polyline/area, a modest scatter, a
// bucketed bar chart — not a re-implementation of the interactive Recharts/
// Canvas versions elsewhere in the app: a printed report has no hover
// tooltips, no animation, and needs to render identically from a single
// static data snapshot, so there's nothing the interactive versions' extra
// complexity would buy here.

// Area + line, baseline-to-bottom — the Summary report's original growth
// sparkline, generalised so both reports (and more than one chart within
// the full report) can call the same function instead of each hand-rolling
// their own point-scaling math. Works for a series that crosses zero
// (growth — dashed zero reference line drawn where it actually falls) and
// for one that's one-sided (drawdown, always <= 0 — the zero line lands at
// the very top of the chart, which is exactly the correct boundary: "how
// far below zero" is the shaded area under the curve down to the bottom
// edge, the same visual the app's own Recharts drawdown AreaChart renders).
// `className`, not a wrapping <div>, is what a caller passes to make a print
// stylesheet rule (e.g. `.vr-chart-svg { height: 72px }`) resize this chart —
// CSS `height` on an <svg> overrides its own height="" attribute while
// viewBox stays fixed, so preserveAspectRatio="none" rescales the whole
// drawing to fit (confirmed the mechanism the Summary report's print
// compaction already relied on before this function existed). A wrapping
// element instead would leave the inner <svg>'s own fixed-pixel height
// attribute unaffected by the wrapper's own shrunken height, clipping the
// chart instead of rescaling it.
export function buildAreaSparkline(values, {
  width = 700, height = 120, color = 'var(--color-accent-mint)',
  gradientId = 'sg1', showZero = true, className = '',
} = {}) {
  const min = Math.min(...values), max = Math.max(...values)
  const range = (max - min) || 0.01
  const points = values.map((v, i) =>
    `${(i / (values.length - 1)) * width},${height - ((v - min) / range) * height}`
  ).join(' ')
  const zeroY = height - ((0 - min) / range) * height
  const zeroLine = (showZero && zeroY > 0 && zeroY < height)
    ? `<line x1="0" y1="${zeroY.toFixed(1)}" x2="${width}" y2="${zeroY.toFixed(1)}" stroke="var(--doc-hairline)" stroke-dasharray="6,4" stroke-width="1"/>`
    : ''
  return `
    <svg ${className ? `class="${className}" ` : ''}width="100%" height="${height}" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <defs>
        <linearGradient id="${gradientId}" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="${color}" stop-opacity="0.2"/>
          <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
        </linearGradient>
      </defs>
      ${zeroLine}
      <polygon points="${points} ${width},${height} 0,${height}" fill="url(#${gradientId})"/>
      <polyline points="${points}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
    </svg>
  `
}

// Several same-length series on one shared scale — Backtest's cumulative
// return lines (your portfolio / equal weight / benchmark) and Compare's
// two-portfolio growth overlay both need this shape: N lines, independent
// colours/dash, one y-domain spanning all of them.
export function buildMultiLineSVG({ series, width = 700, height = 160 }) {
  const all = series.flatMap(s => s.values)
  const min = Math.min(...all), max = Math.max(...all)
  const range = (max - min) || 0.01
  const n = series[0]?.values.length || 1
  const x = i => (i / (n - 1)) * width
  const y = v => height - ((v - min) / range) * height
  const zeroY = y(0)
  const zeroLine = (zeroY > 0 && zeroY < height)
    ? `<line x1="0" y1="${zeroY.toFixed(1)}" x2="${width}" y2="${zeroY.toFixed(1)}" stroke="var(--doc-hairline)" stroke-dasharray="6,4" stroke-width="1"/>`
    : ''
  const lines = series.map(s => {
    const pts = s.values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
    return `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="${s.strokeWidth || 1.8}" ${s.dash ? 'stroke-dasharray="5,4"' : ''}/>`
  }).join('')
  return `<svg width="100%" height="${height}" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">${zeroLine}${lines}</svg>`
}

// Monte Carlo's bad/median/good fan, collapsed to its three percentile
// lines plus the band between the 5th and 95th — the interactive version's
// ~150 faint sampled paths are exactly the "replicating the interactive
// version" complexity the brief says to skip; the band communicates the
// same spread without them.
export function buildMonteCarloBandSVG({ p5, p50, p95, width = 700, height = 170 }) {
  const all = [...p5, ...p50, ...p95]
  const min = Math.min(...all), max = Math.max(...all)
  const range = (max - min) || 1
  const n = p50.length
  const x = i => (i / (n - 1)) * width
  const y = v => height - ((v - min) / range) * height
  const line = arr => arr.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
  const bandPoints = `${line(p95)} ${line([...p5].reverse())}`
  return `
    <svg width="100%" height="${height}" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <polygon points="${bandPoints}" fill="var(--color-accent-mint)" fill-opacity="0.08"/>
      <polyline points="${line(p5)}" fill="none" stroke="var(--color-signal-negative)" stroke-width="1.3" stroke-dasharray="5,4"/>
      <polyline points="${line(p50)}" fill="none" stroke="var(--color-accent-mint)" stroke-width="2.2"/>
      <polyline points="${line(p95)}" fill="none" stroke="var(--doc-ink-secondary)" stroke-width="1.3" stroke-dasharray="5,4"/>
    </svg>
  `
}

// The frontier cloud, downsampled to maxPoints (the real simulation runs
// 5,000 — an inline SVG with 5,000 <circle>s is both unnecessary for what
// a static print page communicates and a real file-size/render-time cost,
// unlike the live Canvas version which redraws from scratch every frame
// anyway). Sharpe still drives each dot's colour band low/mid/high, same
// three-way split the correlation heat table elsewhere in both reports
// already uses, so a reader doesn't need the full 5-stop gradient legend to
// read "greener is better" at a glance.
export function buildFrontierScatterSVG({
  vols, returns, sharpes, yourVol, yourReturn, optVol, optReturn,
  width = 700, height = 260, maxPoints = 450,
}) {
  const n = vols.length
  const step = Math.max(1, Math.floor(n / maxPoints))
  const sampled = []
  for (let i = 0; i < n; i += step) sampled.push({ v: vols[i], r: returns[i], s: sharpes[i] })

  const allV = [...vols, yourVol, optVol].filter(v => v != null)
  const allR = [...returns, yourReturn, optReturn].filter(v => v != null)
  const rawMinV = Math.min(...allV), rawMaxV = Math.max(...allV)
  const rawMinR = Math.min(...allR), rawMaxR = Math.max(...allR)
  const padV = (rawMaxV - rawMinV) * 0.08 || 0.01
  const padR = (rawMaxR - rawMinR) * 0.1  || 0.01
  const minV = rawMinV - padV, maxV = rawMaxV + padV
  const minR = rawMinR - padR, maxR = rawMaxR + padR
  const x = v => ((v - minV) / (maxV - minV)) * width
  const y = r => height - ((r - minR) / (maxR - minR)) * height

  const minS = Math.min(...sharpes), maxS = Math.max(...sharpes)
  const dots = sampled.map(p => {
    const t = maxS !== minS ? (p.s - minS) / (maxS - minS) : 0.5
    const color = t > 0.6 ? 'var(--color-accent-mint)' : t > 0.3 ? 'var(--color-signal-warning)' : 'var(--color-signal-negative)'
    return `<circle cx="${x(p.v).toFixed(1)}" cy="${y(p.r).toFixed(1)}" r="1.6" fill="${color}" fill-opacity="0.5"/>`
  }).join('')

  const optMarker = optVol != null
    ? `<circle cx="${x(optVol).toFixed(1)}" cy="${y(optReturn).toFixed(1)}" r="5.5" fill="var(--color-signal-warning)" stroke="var(--doc-bg)" stroke-width="1.5"/>`
    : ''
  const yoursMarker = yourVol != null
    ? `<circle cx="${x(yourVol).toFixed(1)}" cy="${y(yourReturn).toFixed(1)}" r="5.5" fill="var(--color-accent-mint)" stroke="var(--doc-bg)" stroke-width="1.5"/>`
    : ''

  return `<svg width="100%" height="${height}" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">${dots}${optMarker}${yoursMarker}</svg>`
}

// Daily returns bucketed into `bins` histogram bars, with a dashed VaR
// reference line — the one chart type neither report had a precedent for
// (the other five all generalise an existing sparkline/line shape), built
// fresh here rather than inside either export module so it stays next to
// its siblings and is easy to find.
export function buildHistogramSVG({ returns, varPct, width = 700, height = 160, bins = 22 }) {
  const min = Math.min(...returns), max = Math.max(...returns)
  const range = (max - min) || 0.01
  const binW = range / bins
  const counts = new Array(bins).fill(0)
  returns.forEach(v => {
    let idx = Math.floor((v - min) / binW)
    if (idx >= bins) idx = bins - 1
    if (idx < 0) idx = 0
    counts[idx]++
  })
  const maxCount = Math.max(...counts) || 1
  const barW = width / bins
  const bars = counts.map((c, i) => {
    const h = (c / maxCount) * height
    const binMid = min + (i + 0.5) * binW
    const color = binMid < 0 ? 'var(--color-signal-negative)' : 'var(--color-accent-mint)'
    return `<rect x="${(i * barW + 0.5).toFixed(1)}" y="${(height - h).toFixed(1)}" width="${Math.max(barW - 1, 0.5).toFixed(1)}" height="${h.toFixed(1)}" fill="${color}" fill-opacity="0.55"/>`
  }).join('')
  const varX = ((varPct - min) / range) * width
  const varLine = (varPct != null && varX >= 0 && varX <= width)
    ? `<line x1="${varX.toFixed(1)}" y1="0" x2="${varX.toFixed(1)}" y2="${height}" stroke="var(--color-signal-negative)" stroke-width="1.5" stroke-dasharray="4,3"/>`
    : ''
  return `<svg width="100%" height="${height}" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">${bars}${varLine}</svg>`
}
