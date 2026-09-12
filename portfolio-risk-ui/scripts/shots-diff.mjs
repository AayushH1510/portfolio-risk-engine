// Compares two screenshot trees produced by scripts/shots.mjs (e.g. a
// "before" and "after" baseline) and reports per-view pixel-difference
// counts and percentages. Exits non-zero if any view's differing-pixel
// percentage exceeds --threshold, or if a view is missing from either side,
// or if two same-named views have different dimensions.
//
// Usage:
//   npm run shots:diff -- screenshots-before screenshots-after
//   npm run shots:diff -- screenshots-before screenshots-after --threshold=0.05
import fs from 'node:fs'
import path from 'node:path'
import pixelmatch from 'pixelmatch'
import { PNG } from 'pngjs'

function parseArgs(argv) {
  const positional = argv.filter(a => !a.startsWith('--'))
  const opts = {}
  for (const a of argv) {
    const m = /^--([a-z]+)=(.*)$/.exec(a)
    if (m) opts[m[1]] = m[2]
  }
  return {
    dirA: positional[0],
    dirB: positional[1],
    threshold: opts.threshold != null ? parseFloat(opts.threshold) : 0.1,   // % of pixels, not pixelmatch's own per-pixel sensitivity
  }
}

function walkPngs(dir, base = dir, out = []) {
  if (!fs.existsSync(dir)) return out
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) walkPngs(full, base, out)
    else if (entry.name.toLowerCase().endsWith('.png')) out.push(path.relative(base, full).split(path.sep).join('/'))
  }
  return out
}

function main() {
  const { dirA, dirB, threshold } = parseArgs(process.argv.slice(2))
  if (!dirA || !dirB) {
    console.error('Usage: npm run shots:diff -- <dirA> <dirB> [--threshold=0.1]')
    process.exit(2)
  }

  const filesA = new Set(walkPngs(dirA))
  const filesB = new Set(walkPngs(dirB))
  const allFiles = [...new Set([...filesA, ...filesB])].sort()

  if (allFiles.length === 0) {
    console.error(`No .png files found under ${dirA} or ${dirB}`)
    process.exit(2)
  }

  let worstOffenders = false
  const rows = []

  for (const rel of allFiles) {
    if (!filesA.has(rel)) { rows.push({ rel, status: 'missing-in-A' }); worstOffenders = true; continue }
    if (!filesB.has(rel)) { rows.push({ rel, status: 'missing-in-B' }); worstOffenders = true; continue }

    const imgA = PNG.sync.read(fs.readFileSync(path.join(dirA, rel)))
    const imgB = PNG.sync.read(fs.readFileSync(path.join(dirB, rel)))

    if (imgA.width !== imgB.width || imgA.height !== imgB.height) {
      rows.push({ rel, status: 'size-mismatch', sizeA: `${imgA.width}x${imgA.height}`, sizeB: `${imgB.width}x${imgB.height}` })
      worstOffenders = true
      continue
    }

    const { width, height } = imgA
    const diff = new PNG({ width, height })
    const diffPixels = pixelmatch(imgA.data, imgB.data, diff.data, width, height, { threshold: 0.1 })
    const totalPixels = width * height
    const pct = totalPixels ? (diffPixels / totalPixels) * 100 : 0
    const over = pct > threshold
    if (over) worstOffenders = true
    rows.push({ rel, status: 'ok', diffPixels, totalPixels, pct, over })
  }

  console.log(`Comparing ${dirA}  vs  ${dirB}   (threshold: ${threshold}% of pixels)\n`)
  for (const r of rows) {
    if (r.status === 'missing-in-A')  { console.log(`  MISSING-A   ${r.rel}`); continue }
    if (r.status === 'missing-in-B')  { console.log(`  MISSING-B   ${r.rel}`); continue }
    if (r.status === 'size-mismatch') { console.log(`  SIZE-DIFF   ${r.rel}   A=${r.sizeA}  B=${r.sizeB}`); continue }
    const marker = r.over ? 'DIFF   ' : 'ok     '
    console.log(`  ${marker}${r.rel.padEnd(40)} ${r.diffPixels}/${r.totalPixels} px  (${r.pct.toFixed(4)}%)`)
  }

  const okRows = rows.filter(r => r.status === 'ok')
  const overRows = okRows.filter(r => r.over)
  console.log(`\n${okRows.length - overRows.length}/${okRows.length} views within threshold.`)
  if (overRows.length) console.log(`${overRows.length} view(s) exceed ${threshold}%.`)

  process.exitCode = worstOffenders ? 1 : 0
}

main()
