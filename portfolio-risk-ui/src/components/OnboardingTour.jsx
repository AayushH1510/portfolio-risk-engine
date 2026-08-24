import { useState, useEffect, useLayoutEffect } from 'react'

const STORAGE_KEY = 'varense_has_visited'

const STEPS = [
  { anchor: '[data-tour="stocks"]',       text: 'Add your stocks here' },
  { anchor: '[data-tour="weights"]',      text: 'Set how much of each you own' },
  { anchor: '[data-tour="period"]',       text: 'Pick your time period' },
  { anchor: '[data-tour="run-analysis"]', text: 'Click here to see your risk' },
]

export default function OnboardingTour() {
  const [visible, setVisible] = useState(false)
  const [step, setStep]       = useState(0)
  const [rect, setRect]       = useState(null)

  // Only ever run for a first-time visitor.
  useEffect(() => {
    if (!localStorage.getItem(STORAGE_KEY)) setVisible(true)
  }, [])

  // Track the current step's anchor position, and keep it in sync on resize.
  useLayoutEffect(() => {
    if (!visible) return
    const measure = () => {
      const el = document.querySelector(STEPS[step].anchor)
      setRect(el ? el.getBoundingClientRect() : null)
    }
    measure()
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [visible, step])

  const finish = () => {
    localStorage.setItem(STORAGE_KEY, 'true')
    setVisible(false)
  }

  const next = () => {
    if (step === STEPS.length - 1) finish()
    else setStep(s => s + 1)
  }

  if (!visible || !rect) return null

  const isLast = step === STEPS.length - 1
  const top    = Math.min(rect.top + rect.height / 2, window.innerHeight - 90)
  const left   = rect.right + 16

  return (
    <div style={{
      position: 'fixed', top, left, transform: 'translateY(-50%)',
      zIndex: 999, width: 236,
      background: 'var(--card)', border: '1px solid var(--border)',
      borderRadius: 10, padding: '14px 16px',
      boxShadow: '0 12px 32px rgba(0,0,0,0.45)',
    }}>
      {/* Arrow pointing back at the anchor */}
      <div style={{
        position: 'absolute', left: -7, top: '50%', transform: 'translateY(-50%)',
        width: 0, height: 0,
        borderTop: '7px solid transparent',
        borderBottom: '7px solid transparent',
        borderRight: '7px solid var(--card)',
      }} />
      <div style={{
        position: 'absolute', left: -8, top: '50%', transform: 'translateY(-50%)',
        width: 0, height: 0,
        borderTop: '8px solid transparent',
        borderBottom: '8px solid transparent',
        borderRight: '8px solid var(--border)',
        zIndex: -1,
      }} />

      <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.08em', color: 'var(--accent)', marginBottom: 6 }}>
        STEP {step + 1} OF {STEPS.length}
      </div>
      <div style={{ fontSize: 13, color: 'var(--text-primary)', lineHeight: 1.5, marginBottom: 14 }}>
        {STEPS[step].text}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <button
          onClick={finish}
          style={{ background: 'transparent', color: 'var(--text-muted)', fontSize: 11, fontWeight: 600, padding: 0, textTransform: 'none', letterSpacing: 0 }}
        >
          Skip
        </button>
        <button
          onClick={next}
          className="btn-primary"
          style={{ width: 'auto', padding: '6px 16px', fontSize: 11 }}
        >
          {isLast ? 'Got it' : 'Next'}
        </button>
      </div>
    </div>
  )
}
