import { useState } from 'react'

export default function AuthModal({ onSignInGoogle, onSignInEmail, onSignUpEmail, onClose }) {
  const [mode, setMode]       = useState('signin') // signin | signup
  const [email, setEmail]     = useState('')
  const [password, setPassword] = useState('')
  const [error, setError]     = useState(null)
  const [loading, setLoading] = useState(false)
  const [sent, setSent]       = useState(false)

  const handleSubmit = async () => {
    setLoading(true)
    setError(null)
    const err = mode === 'signin'
      ? await onSignInEmail(email, password)
      : await onSignUpEmail(email, password)
    setLoading(false)
    if (err) setError(err.message)
    else if (mode === 'signup') setSent(true)
    else onClose()
  }

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 1000,
      background: 'rgba(0,0,0,0.6)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}
      onClick={onClose}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          width: 360, background: 'var(--card)',
          border: '1px solid var(--border)',
          borderRadius: 14, padding: '28px 28px 24px',
          boxShadow: '0 24px 64px rgba(0,0,0,0.5)',
        }}
      >
        {/* Logo + title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 22 }}>
          <div style={{ width: 32, height: 32, background: 'var(--accent)', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M8 2L14 6V10L8 14L2 10V6L8 2Z" fill="#1e2420"/>
            </svg>
          </div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>Varense</div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{mode === 'signin' ? 'Sign in to your account' : 'Create your account'}</div>
          </div>
        </div>

        {sent ? (
          <div style={{ textAlign: 'center', padding: '12px 0' }}>
            <div style={{ fontSize: 28, marginBottom: 10 }}>📧</div>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 6 }}>Check your email</div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.6 }}>
              We sent a confirmation link to <strong style={{ color: 'var(--text-secondary)' }}>{email}</strong>. Click it to activate your account.
            </div>
          </div>
        ) : (
          <>
            {/* Google */}
            <button
              onClick={onSignInGoogle}
              style={{
                width: '100%', padding: '10px 0', borderRadius: 8,
                fontSize: 13, fontWeight: 600, cursor: 'pointer',
                background: 'var(--card-2)', border: '1px solid var(--border)',
                color: 'var(--text-primary)', display: 'flex', alignItems: 'center',
                justifyContent: 'center', gap: 10, marginBottom: 16, transition: 'all 0.15s',
              }}
              onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--accent)'}
              onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
            >
              <svg width="18" height="18" viewBox="0 0 18 18">
                <path fill="#4285F4" d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908c1.702-1.567 2.684-3.875 2.684-6.615z"/>
                <path fill="#34A853" d="M9 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z"/>
                <path fill="#FBBC05" d="M3.964 10.71A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.71V4.958H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.042l3.007-2.332z"/>
                <path fill="#EA4335" d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.958L3.964 7.29C4.672 5.163 6.656 3.58 9 3.58z"/>
              </svg>
              Continue with Google
            </button>

            {/* Divider */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
              <div style={{ flex: 1, height: 1, background: 'var(--border)' }}/>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>or</span>
              <div style={{ flex: 1, height: 1, background: 'var(--border)' }}/>
            </div>

            {/* Email + password */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 14 }}>
              <input
                type="email" placeholder="Email address"
                value={email} onChange={e => setEmail(e.target.value)}
                style={{ fontSize: 13 }}
              />
              <input
                type="password" placeholder="Password"
                value={password} onChange={e => setPassword(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSubmit()}
                style={{ fontSize: 13 }}
              />
            </div>

            {error && (
              <div style={{ fontSize: 12, color: 'var(--negative)', marginBottom: 10, padding: '6px 10px', background: 'rgba(224,92,92,0.1)', borderRadius: 6 }}>
                {error}
              </div>
            )}

            <button
              onClick={handleSubmit}
              disabled={loading || !email || !password}
              className="btn-primary"
              style={{ marginBottom: 14 }}
            >
              {loading ? 'Please wait...' : mode === 'signin' ? 'Sign in' : 'Create account'}
            </button>

            <div style={{ textAlign: 'center', fontSize: 12, color: 'var(--text-muted)' }}>
              {mode === 'signin' ? "Don't have an account? " : 'Already have an account? '}
              <span
                onClick={() => { setMode(mode === 'signin' ? 'signup' : 'signin'); setError(null) }}
                style={{ color: 'var(--accent)', cursor: 'pointer', fontWeight: 600 }}
              >
                {mode === 'signin' ? 'Sign up' : 'Sign in'}
              </span>
            </div>
          </>
        )}
      </div>
    </div>
  )
}