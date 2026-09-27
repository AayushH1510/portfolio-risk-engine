import { useEffect, useRef, useState } from 'react'
import { runExportPDF } from './ExportPDF'
import { runExportCSV } from './ExportCSV'

// Combined "Export" trigger for the desktop header — replaces two separate
// Export PDF / Export CSV buttons there (RESPONSIVE_AUDIT.md's 1280px
// nav-fit fix: the header didn't have room for both plus eight tabs plus
// Sign in). The sidebar drawer keeps the original two full-width buttons
// (ExportPDF.jsx / ExportCSV.jsx's own default exports) — phone/tablet were
// never the width-constrained case, so there's no reason to collapse them
// there too. This menu calls the same runExportPDF/runExportCSV functions
// those two files export, not a reimplementation of either flow.
export default function ExportMenu({ data, tickers, weights }) {
  const [open, setOpen] = useState(false)
  const [coords, setCoords] = useState(null)
  const wrapperRef = useRef(null)
  const triggerRef = useRef(null)
  const pdfItemRef = useRef(null)
  const csvItemRef = useRef(null)

  const positionPanel = () => {
    const rect = triggerRef.current.getBoundingClientRect()
    setCoords({ top: rect.bottom + 6, right: window.innerWidth - rect.right })
  }

  useEffect(() => {
    if (!open) return
    pdfItemRef.current?.focus()

    // Recomputed on resize, not just at open — a menu left open through a
    // window resize (or the 125% browser-zoom case this fix was verified
    // against) would otherwise anchor to a stale trigger position.
    const handleResize = () => positionPanel()

    // Checked against the whole wrapper (trigger + panel), not just the
    // trigger — a real bug caught by testing, not spotted by inspection:
    // checking triggerRef alone treated a click on a menu item itself as
    // "outside", closing the menu on mousedown before the item's own click
    // handler ever ran (mousedown fires, and fires first, on every click).
    // The PDF/CSV items looked like they worked — the menu did close — but
    // runExportPDF/runExportCSV never actually executed.
    const handlePointerDown = (e) => {
      if (wrapperRef.current?.contains(e.target)) return
      setOpen(false)
    }

    const handleKeydown = (e) => {
      if (e.key === 'Escape') {
        setOpen(false)
        triggerRef.current?.focus()
        return
      }
      if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return
      e.preventDefault()
      const items = [pdfItemRef.current, csvItemRef.current]
      const currentIndex = items.indexOf(document.activeElement)
      const nextIndex = e.key === 'ArrowDown'
        ? (currentIndex + 1) % items.length
        : (currentIndex - 1 + items.length) % items.length
      items[nextIndex]?.focus()
    }

    window.addEventListener('resize', handleResize)
    document.addEventListener('mousedown', handlePointerDown)
    document.addEventListener('keydown', handleKeydown)
    return () => {
      window.removeEventListener('resize', handleResize)
      document.removeEventListener('mousedown', handlePointerDown)
      document.removeEventListener('keydown', handleKeydown)
    }
  }, [open])

  if (!data) return null

  const toggleOpen = () => {
    if (open) {
      setOpen(false)
      return
    }
    positionPanel()
    setOpen(true)
  }

  const runAndClose = (fn) => {
    fn({ data, tickers, weights })
    setOpen(false)
    triggerRef.current?.focus()
  }

  return (
    <div className="export-menu" ref={wrapperRef}>
      <button
        ref={triggerRef}
        className="export-menu-trigger"
        onClick={toggleOpen}
        aria-haspopup="true"
        aria-expanded={open}
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
        Export
        <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" style={{ marginLeft: 2 }}>
          <polyline points="6 9 12 15 18 9"/>
        </svg>
      </button>

      {open && coords && (
        <div
          className="export-menu-panel"
          role="menu"
          style={{ position: 'fixed', top: coords.top, right: coords.right }}
        >
          <button
            ref={pdfItemRef}
            className="export-menu-item"
            role="menuitem"
            onClick={() => runAndClose(runExportPDF)}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            PDF Report
          </button>
          <button
            ref={csvItemRef}
            className="export-menu-item"
            role="menuitem"
            onClick={() => runAndClose(runExportCSV)}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            CSV Data
          </button>
        </div>
      )}
    </div>
  )
}
