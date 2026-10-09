import { useEffect } from 'react'

// The actual report-building logic (a genuinely large function, though not
// a heavy dependency in itself — see lib/exportPdfReport.js's own comment)
// is dynamically imported on click, not statically imported here, so the
// /app bundle doesn't pay for it on every load of a feature only some
// users ever click. ExportMenu.jsx's own "PDF Report" menu item does the
// identical dynamic import rather than sharing a static one, for the same
// reason.
export default function ExportPDF({ data, tickers, weights }) {
  // Preloaded once a run's results are on screen — this component only ever
  // renders once hasRun is true (see Sidebar.jsx's hasRun-gated block, the
  // single mount point on phone and desktop alike), so by the time someone
  // actually taps Export, the module below is already cached and the pop-up
  // opens almost immediately instead of waiting on a fresh import() first.
  useEffect(() => { import('../lib/exportPdfReport') }, [])

  if (!data) return null

  const handleClick = async () => {
    const { runExportPDF } = await import('../lib/exportPdfReport')
    runExportPDF({ data, tickers, weights })
  }

  return (
    <button
      onClick={handleClick}
      style={{
        display:'flex', alignItems:'center', gap:6,
        fontSize:'var(--text-body-sm)', fontWeight:'var(--weight-medium)', padding:'6px 14px',
        border:'var(--border-default)', fontFamily:'var(--font-primary)',
        background:'transparent',
        color:'var(--text-primary)', cursor:'pointer',
        transition:'all var(--duration-fast) var(--ease-standard)', letterSpacing:'0.02em',
      }}
      onMouseEnter={e => { e.currentTarget.style.background='var(--surface-elevated)'; e.currentTarget.style.borderColor='var(--line-emphasis)' }}
      onMouseLeave={e => { e.currentTarget.style.background='transparent'; e.currentTarget.style.borderColor='var(--line-hairline)' }}
    >
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="7 10 12 15 17 10"/>
        <line x1="12" y1="15" x2="12" y2="3"/>
      </svg>
      Export PDF
    </button>
  )
}
