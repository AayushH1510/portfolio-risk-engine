// Requires the `feedback` table — run supabase-feedback-table.sql (project
// root) once in the Supabase SQL editor before this will work.

import { useState } from 'react'
import { supabase } from '../lib/supabase'

const TYPES = ['Bug', 'Confusing', 'Suggestion', 'Other']

export default function FeedbackButton({ user }) {
  const [open, setOpen]         = useState(false)
  const [type, setType]         = useState('Suggestion')
  const [message, setMessage]   = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [submitted, setSubmitted]   = useState(false)
  const [error, setError]       = useState(null)

  const reset = () => {
    setOpen(false)
    setType('Suggestion')
    setMessage('')
    setSubmitted(false)
    setError(null)
  }

  const handleSubmit = async () => {
    if (!message.trim()) return
    setSubmitting(true)
    setError(null)

    const { error: insertError } = await supabase.from('feedback').insert({
      user_id: user?.id ?? null,
      type,
      message: message.trim(),
      page: window.location.pathname,
    })

    setSubmitting(false)

    if (insertError) {
      setError('Could not send feedback — please try again.')
      return
    }

    setSubmitted(true)
    setTimeout(reset, 1800)
  }

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        aria-label="Send feedback"
        style={{
          position: 'fixed', bottom: 20, right: 20, zIndex: 500,
          width: 48, height: 48, borderRadius: '50%',
          background: 'var(--accent)', border: 'none',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 6px 20px rgba(0,0,0,0.35)',
          cursor: 'pointer', transition: 'transform 0.15s',
        }}
        onMouseEnter={e => { e.currentTarget.style.transform = 'scale(1.06)' }}
        onMouseLeave={e => { e.currentTarget.style.transform = 'scale(1)' }}
      >
        <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="#0d1a10" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>
        </svg>
      </button>

      {open && (
        <div
          style={{
            position: 'fixed', inset: 0, zIndex: 1000,
            background: 'rgba(0,0,0,0.6)',
            display: 'flex', alignItems: 'flex-end', justifyContent: 'flex-end',
            padding: 20,
          }}
          onClick={reset}
        >
          <div
            onClick={e => e.stopPropagation()}
            style={{
              width: 320, background: 'var(--card)',
              border: '1px solid var(--border)',
              borderRadius: 14, padding: '20px 20px 18px',
              boxShadow: '0 24px 64px rgba(0,0,0,0.5)',
            }}
          >
            {submitted ? (
              <div style={{ textAlign: 'center', padding: '18px 0' }}>
                <div style={{ fontSize: 28, marginBottom: 8 }}>✓</div>
                <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>
                  Thanks, got it!
                </div>
              </div>
            ) : (
              <>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>
                    Send feedback
                  </div>
                  <button
                    onClick={reset}
                    aria-label="Close"
                    style={{ background: 'transparent', color: 'var(--text-muted)', fontSize: 16, padding: 0, lineHeight: 1 }}
                  >
                    ×
                  </button>
                </div>

                <textarea
                  value={message}
                  onChange={e => setMessage(e.target.value)}
                  placeholder="What's working, what's confusing, what's broken?"
                  rows={4}
                  style={{
                    width: '100%', resize: 'vertical', fontFamily: 'inherit',
                    fontSize: 12, marginBottom: 10,
                  }}
                />

                <select
                  value={type}
                  onChange={e => setType(e.target.value)}
                  style={{ marginBottom: 12, fontSize: 12 }}
                >
                  {TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                </select>

                {error && (
                  <div style={{ fontSize: 11, color: 'var(--negative)', marginBottom: 10 }}>
                    {error}
                  </div>
                )}

                <button
                  className="btn-primary"
                  onClick={handleSubmit}
                  disabled={submitting || !message.trim()}
                  style={{ fontSize: 12 }}
                >
                  {submitting ? 'Sending...' : 'Submit'}
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </>
  )
}
