import { useEffect } from 'react'

// The actual report-building logic (a genuinely large function, though not
// a heavy dependency in itself — see lib/exportPdfReport.js's own comment)
// is dynamically imported on click, not statically imported here, so the
// /app bundle doesn't pay for it on every load of a feature only some
// users ever click. ExportMenu.jsx's own two PDF menu items do the
// identical dynamic imports rather than sharing static ones, for the same
// reason.
//
// Two buttons, not one: the brief's "Summary PDF" / "Full report PDF" pair,
// styled to match each other and the existing Export CSV button below them
// (Sidebar.jsx), each with an explicit 44px tap target — the full report's
// own button is new, so it gets one from the start rather than inheriting
// whatever implicit height the original single "Export PDF" button happened
// to have.
export default function ExportPDF({ data, tickers, weights, portfolioValue, sectorData, comparison, stressTestData, valuationData }) {
  // Preloaded once a run's results are on screen — this component only ever
  // renders once hasRun is true (see Sidebar.jsx's hasRun-gated block, the
  // single mount point on phone and desktop alike), so by the time someone
  // actually taps either button, the matching module is already cached and
  // the pop-up opens almost immediately instead of waiting on a fresh
  // import() first.
  useEffect(() => {
    import('../lib/exportPdfReport')
    import('../lib/exportFullReportPdf')
  }, [])

  if (!data) return null

  const handleSummary = async () => {
    const { runExportPDF } = await import('../lib/exportPdfReport')
    runExportPDF({ data, tickers, weights })
  }

  const handleFullReport = async () => {
    const { runExportFullReportPDF } = await import('../lib/exportFullReportPdf')
    runExportFullReportPDF({ data, tickers, weights, portfolioValue, sectorData, comparison, stressTestData, valuationData })
  }

  const btnStyle = {
    display: 'flex', alignItems: 'center', gap: 6, minHeight: 44,
    fontSize: 'var(--text-body-sm)', fontWeight: 'var(--weight-medium)', padding: '6px 14px',
    border: 'var(--border-default)', fontFamily: 'var(--font-primary)',
    background: 'transparent',
    color: 'var(--text-primary)', cursor: 'pointer',
    transition: 'all var(--duration-fast) var(--ease-standard)', letterSpacing: '0.02em',
  }
  const onEnter = e => { e.currentTarget.style.background = 'var(--surface-elevated)'; e.currentTarget.style.borderColor = 'var(--line-emphasis)' }
  const onLeave = e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.borderColor = 'var(--line-hairline)' }

  const icon = (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="7 10 12 15 17 10"/>
      <line x1="12" y1="15" x2="12" y2="3"/>
    </svg>
  )

  return (
    <>
      <button onClick={handleSummary} style={btnStyle} onMouseEnter={onEnter} onMouseLeave={onLeave}>
        {icon}
        Summary PDF
      </button>
      <button onClick={handleFullReport} style={btnStyle} onMouseEnter={onEnter} onMouseLeave={onLeave}>
        {icon}
        Full report PDF
      </button>
    </>
  )
}
