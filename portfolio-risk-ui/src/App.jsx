import { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import axios from 'axios'
import Sidebar from './components/Sidebar'
import Logo from './components/Logo'
import Dashboard from './pages/Dashboard'
import RiskAnalysis from './pages/RiskAnalysis'
import MonteCarlo from './pages/MonteCarlo'
import Frontier from './pages/Frontier'
import Valuation from './pages/Valuation'
import Backtest from './pages/Backtest'
import Learn from './pages/Learn'
import AuthModal from './components/AuthModal'
import ExportMenu from './components/ExportMenu'
import CompareWrapper from './components/CompareWrapper'
import BusyNotice from './components/BusyNotice'
import StockDrawer from './components/StockDrawer'
import OnboardingTour from './components/OnboardingTour'
import FeedbackButton from './components/FeedbackButton'
import { useComparison } from './hooks/useComparison'
import { useAnalysis } from './hooks/useAnalysis'
import { usePortfolios } from './hooks/usePortfolios'
import { useAuth } from './hooks/useAuth'
import useCompactViewport from './hooks/useCompactViewport'
import { supportsFinePointer } from './lib/pointer'
import { buildSectorRows, hasTransientFailure } from './lib/sectorExposure'

// Matches api.py's FUNDAMENTALS_FAILURE_TTL exactly — that's how long the
// backend caches a transient failure before it'll actually attempt Finnhub
// again, so retrying any sooner from here would just re-read the same
// cached failure. One retry only (see the effect below) — not a poll loop.
const SECTOR_RETRY_DELAY_MS = 60_000

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const TABS = [
  { id: 'dashboard',  label: 'Dashboard' },
  { id: 'risk',       label: 'Risk Analysis' },
  { id: 'montecarlo', label: 'Monte Carlo' },
  { id: 'frontier',   label: 'Efficient Frontier' },
  { id: 'valuation',  label: 'Valuation' },
  { id: 'compare',    label: 'Compare' },
  { id: 'backtest',   label: 'Backtest' },
  { id: 'learn',      label: 'Learn' },
]

export default function App() {
  const [activeTab, setActiveTab]       = useState('dashboard')
  const [showAuth, setShowAuth]         = useState(false)
  const [drawerTicker, setDrawerTicker] = useState(null)
  const [drawerWeight, setDrawerWeight] = useState(null)
  // null = not fetched yet, or the whole request failed outright (a
  // network error, not a per-ticker one) — render nothing. Once fetched,
  // every portfolio ticker lands in some row (a real sector, or one of the
  // two named exception rows — see lib/sectorExposure.js's buildSectorRows)
  // so this is effectively never `[]` for a non-empty portfolio anymore;
  // SectorChart.jsx's own "this portfolio is all funds/ETFs" explanatory
  // state now checks the rows' own `kind` instead of an empty array.
  const [sectorData, setSectorData]     = useState(null)
  const [sectorLoading, setSectorLoading] = useState(false)
  // Sector composition only depends on which tickers (and at what weights)
  // are in the portfolio — not on the analysis result itself. `data`
  // updates twice per run now (the fast summary, then the background full
  // response replacing it once it lands), and without this the effect
  // below re-fires — a second /api/fundamentals round-trip and a "Loading
  // sector exposure…" flicker — for the exact same portfolio composition.
  const sectorFetchKeyRef = useRef(null)
  // Both refs, not effect-local state — the effect below intentionally
  // no-ops (via the fetchKey guard above) when `data`'s heavy-tier update
  // lands for the *same* composition, and React still tears down and
  // re-runs the effect's own cleanup on every one of those no-ops. A
  // pending retry timeout stored in an effect-local variable would get
  // silently cancelled by that teardown before it ever fired; a ref
  // survives it, the same reason sectorFetchKeyRef itself is a ref.
  const sectorRetryTimeoutRef = useRef(null)
  // Has this specific composition already used its one retry? Reset only
  // when the composition actually changes (below), not on the harmless
  // same-composition re-runs above.
  const sectorRetriedRef = useRef(false)
  const tabBarRef = useRef(null)
  const tabRefs   = useRef({})
  // Below --breakpoint-tablet the sidebar is a closed-by-default off-canvas
  // drawer (see .sidebar-shell, index.css) — this is the single source of
  // truth for open/closed, shared by the hamburger, the backdrop, and
  // Sidebar's own Escape/focus-trap handling. Meaningless at 1024px+ (CSS
  // resets the sidebar to a normal column regardless of this value).
  const [sidebarOpen, setSidebarOpen] = useState(false)
  // OnboardingTour's anchors all live inside the sidebar — while it's
  // running, the drawer needs to stay open and Run Analysis (step 4's own
  // anchor) must not auto-close it out from under the tour.
  const [tourActive, setTourActive]   = useState(false)

  const { user, loading: authLoading, signInWithGoogle, signInWithEmail, signUpWithEmail, signOut } = useAuth()
  const analysis = useAnalysis()
  const { portfolios, savePortfolio, deletePortfolio } = usePortfolios(user)
  const compB = useComparison()
  const { data, loading, error, hasRun, heavyLoading, heavyError, runAnalysis, tickers, weights } = analysis
  const { rateLimited, heavyRateLimited, retryRun, retryHeavy } = analysis
  const isCompactViewport = useCompactViewport()

  useEffect(() => {
    if (!data || !tickers.length) {
      setSectorData(null)
      setSectorLoading(false)
      sectorFetchKeyRef.current = null
      if (sectorRetryTimeoutRef.current) {
        clearTimeout(sectorRetryTimeoutRef.current)
        sectorRetryTimeoutRef.current = null
      }
      sectorRetriedRef.current = false
      return
    }
    const fetchKey = `${tickers.join(',')}|${weights.join(',')}`
    if (fetchKey === sectorFetchKeyRef.current) return   // heavy tier just replaced `data` for the same run — nothing to refetch

    // A genuinely new composition (not just the heavy tier landing) — any
    // retry still pending for the *previous* composition is now chasing a
    // portfolio that no longer exists, and this new one gets its own fresh
    // chance at the one retry below.
    if (sectorRetryTimeoutRef.current) {
      clearTimeout(sectorRetryTimeoutRef.current)
      sectorRetryTimeoutRef.current = null
    }
    sectorRetriedRef.current = false
    sectorFetchKeyRef.current = fetchKey

    // Staleness is judged against the ref at resolution time, not an
    // effect-cleanup flag — React tears down this effect's closure (and
    // would flip a local `cancelled` flag) the moment `data` changes again,
    // which now happens for reasons unrelated to this fetch (the heavy tier
    // landing). A closure-scoped flag would falsely mark this in-flight
    // request "cancelled" and its .finally() would then skip clearing
    // sectorLoading, leaving the spinner stuck forever even though nothing
    // is actually wrong. Comparing against the ref only treats a request as
    // stale when a *different* portfolio has genuinely superseded it.
    const fetchSectorData = (isRetry) => {
      if (!isRetry) setSectorLoading(true)
      axios.get(`${API}/api/fundamentals?tickers=${tickers.join(',')}`)
        .then(res => {
          if (sectorFetchKeyRef.current !== fetchKey) return
          setSectorData(buildSectorRows(res.data.tickers, tickers, weights))

          // One retry only: schedule it the first time a response for this
          // composition comes back with a transient failure, never again
          // after that (whether the retry itself succeeds or not) — see
          // sectorRetriedRef's own comment.
          if (!sectorRetriedRef.current && hasTransientFailure(res.data.tickers)) {
            sectorRetriedRef.current = true
            sectorRetryTimeoutRef.current = setTimeout(() => {
              sectorRetryTimeoutRef.current = null
              if (sectorFetchKeyRef.current !== fetchKey) return
              fetchSectorData(true)
            }, SECTOR_RETRY_DELAY_MS)
          }
        })
        .catch(() => { if (sectorFetchKeyRef.current === fetchKey) setSectorData(null) })
        .finally(() => { if (sectorFetchKeyRef.current === fetchKey && !isRetry) setSectorLoading(false) })
    }

    fetchSectorData(false)
  }, [data])

  // Keep the active tab in view when the header bar overflows (narrow
  // widths). scrollLeft = offsetLeft rather than scrollIntoView(): the
  // latter walks up and scrolls ancestors too, and the app shell above
  // this is overflow:hidden, which turns that into visible surprises.
  useEffect(() => {
    const container = tabBarRef.current
    const activeEl  = tabRefs.current[activeTab]
    if (container && activeEl) container.scrollLeft = activeEl.offsetLeft
  }, [activeTab])

  // Mouse-wheel reachability for the tab bar (RESPONSIVE_AUDIT.md's 1280px
  // nav-fit fix, item 2). `overflowX:'auto'` already scrolls by touch-drag,
  // but a plain vertical wheel gesture over it scrolls the *page*, not the
  // nav — a mouse/trackpad user had no gesture at all that reached a tab
  // past the visible edge. (pointer: fine) only, computed once at module
  // load (lib/pointer.js — same convention as supportsHover): touch
  // behaviour is completely untouched, this only adds a second input path
  // for a pointer type that previously had none.
  useEffect(() => {
    if (!supportsFinePointer) return
    const container = tabBarRef.current
    if (!container) return

    const handleWheel = (e) => {
      if (container.scrollWidth <= container.clientWidth) return
      // Only take over a predominantly-vertical gesture — a trackpad's own
      // native horizontal scroll (deltaX already the larger component) is
      // left alone, so this never fights an input that can already move
      // the row by itself.
      if (Math.abs(e.deltaY) <= Math.abs(e.deltaX)) return
      e.preventDefault()
      container.scrollLeft += e.deltaY
    }

    container.addEventListener('wheel', handleWheel, { passive: false })
    return () => container.removeEventListener('wheel', handleWheel)
  }, [])

  // Open the drawer the moment the tour starts. A no-op visually at
  // 1024px+ (CSS ignores sidebarOpen there), so this doesn't need its own
  // width check.
  useEffect(() => {
    if (tourActive) setSidebarOpen(true)
  }, [tourActive])

  // A failed run otherwise leaves the drawer open on a compact viewport —
  // today's behaviour, kept for every other error. The rate-limit busy
  // state is the one exception: its message and Try again button render
  // behind the open drawer there (see lib/rateLimit.js), so this closes it
  // the same way a successful run does. Sidebar stays mounted either way —
  // only its open/closed CSS state changes, so its inputs are untouched.
  // Same tourActive guard as the open-on-tour-start effect above, for the
  // same reason: Run Analysis is itself one of the tour's anchors.
  useEffect(() => {
    if (rateLimited && isCompactViewport && !tourActive) setSidebarOpen(false)
  }, [rateLimited, isCompactViewport, tourActive])

  const closeSidebarDrawer = () => setSidebarOpen(false)

  const handleLoadPortfolio = (p) => {
    analysis.setTickers(p.tickers)
    analysis.setWeightsAll(p.weights)
    analysis.setPeriod(p.period)
    analysis.setPortfolioValue(p.portfolio_value)
  }

  const openDrawer = (ticker, weight) => {
    setDrawerTicker(ticker)
    setDrawerWeight(weight ?? null)
  }

  const closeDrawer = () => {
    setDrawerTicker(null)
    setDrawerWeight(null)
  }

  // No fallback string here on purpose — this only ever renders inside
  // `{user && (...)}` below now, so there's nothing for a fallback to
  // paper over. The old `|| 'PR'` was the actual bug this pass fixed: the
  // avatar rendered unconditionally (a sibling after the signed-in/
  // signed-out ternary, not inside it), so a signed-out visitor saw both
  // "Sign in" and a "PR" initials-avatar look-alike implying someone was
  // already signed in.
  const initials = user?.email?.slice(0, 2).toUpperCase()

  return (
    <div className="grain-canvas app-shell" style={{ overflow: 'hidden', background: 'var(--surface-canvas)', position: 'relative' }}>
      <div className="vignette-layer" />

      <div style={{ display: 'flex', height: '100%', position: 'relative', zIndex: 1 }}>

      {showAuth && (
        <AuthModal
          onSignInGoogle={signInWithGoogle}
          onSignInEmail={signInWithEmail}
          onSignUpEmail={signUpWithEmail}
          onClose={() => setShowAuth(false)}
        />
      )}

      <StockDrawer
        ticker={drawerTicker}
        weight={drawerWeight}
        onClose={closeDrawer}
      />

      <OnboardingTour onActiveChange={setTourActive} />

      {/* Backdrop — only visible/interactive below 1024px (.sidebar-backdrop,
          index.css); mounting it unconditionally on sidebarOpen is safe
          since it's display:none at 1024px+ regardless of this state. */}
      {sidebarOpen && (
        <div className="sidebar-backdrop" onClick={closeSidebarDrawer} />
      )}

      <Sidebar
        {...analysis}
        setWeightsAll={analysis.setWeightsAll}
        onRun={(customDates) => runAnalysis(customDates, () => { if (!tourActive) closeSidebarDrawer() })}
        loading={loading}
        portfolios={portfolios}
        onSavePortfolio={savePortfolio}
        onLoadPortfolio={handleLoadPortfolio}
        onDeletePortfolio={deletePortfolio}
        user={user}
        authLoading={authLoading}
        onSignOut={signOut}
        onShowAuth={() => setShowAuth(true)}
        onTickerClick={openDrawer}
        drawerOpen={sidebarOpen}
        onCloseDrawer={closeSidebarDrawer}
      />

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>

        <header style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '0 12px', height: 48,
          background: 'transparent',
          borderBottom: 'var(--border-default)',
          flexShrink: 0, minWidth: 0, overflow: 'hidden',
        }}>
          <button
            className="sidebar-hamburger"
            onClick={() => setSidebarOpen(o => !o)}
            aria-label={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
            aria-expanded={sidebarOpen}
            style={{
              width: 44, height: 44, flexShrink: 0, marginRight: 2,
              alignItems: 'center', justifyContent: 'center',
              background: 'transparent', border: 'none', cursor: 'pointer', padding: 0,
            }}
          >
            <svg width="18" height="14" viewBox="0 0 18 14" fill="none">
              <path d="M0 1H18M0 7H18M0 13H18" stroke="var(--text-primary)" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
          </button>

          <nav
            ref={tabBarRef}
            className="tab-bar-scroll"
            style={{
              display: 'flex', gap: 0, height: '100%', alignItems: 'stretch',
              overflowX: 'auto', flex: 1, minWidth: 0, position: 'relative',
              WebkitOverflowScrolling: 'touch', scrollSnapType: 'x proximity',
            }}
          >
            {TABS.map(tab => {
              const active   = activeTab === tab.id
              const unlocked = tab.id === 'learn' || tab.id === 'valuation' || tab.id === 'dashboard' || tab.id === 'compare' || hasRun
              return (
                <button
                  key={tab.id}
                  ref={el => { tabRefs.current[tab.id] = el }}
                  onClick={() => unlocked && setActiveTab(tab.id)}
                  aria-current={active ? 'page' : undefined}
                  className="app-tab-btn"
                  style={{
                    fontSize: 'var(--text-body-sm)', fontWeight: 'var(--weight-medium)',
                    letterSpacing: 'var(--tracking-tab)', textTransform: 'uppercase',
                    fontFamily: 'var(--font-primary)',
                    background: 'transparent', border: 'none',
                    borderBottom: active ? '2px solid var(--signal-positive)' : '2px solid transparent',
                    color: active ? 'var(--text-primary)' : !unlocked ? 'var(--text-faint)' : 'var(--text-muted)',
                    cursor: !unlocked ? 'not-allowed' : 'pointer',
                    transition: 'all var(--duration-fast) var(--ease-standard)', whiteSpace: 'nowrap', flexShrink: 0,
                    scrollSnapAlign: 'start',
                  }}
                >
                  {tab.label}
                </button>
              )
            })}
          </nav>

          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0, marginLeft: 8 }}>
            <FeedbackButton user={user} />

            {hasRun && (
              <div className="header-export-actions">
                <ExportMenu data={data} tickers={tickers} weights={weights} />
              </div>
            )}

            {/* Sign-in/account — hidden below the compact-viewport
                threshold (.header-account-actions, index.css): moved into
                the sidebar drawer there instead (Sidebar.jsx's own
                sidebar-account-actions, alongside Export — §3 already
                established the drawer as the action surface at phone
                width, not the header). display/gap live in that class, not
                here, for the usual inline-always-wins-over-a-media-query
                reason every other responsive fix in this file relies on. */}
            <div className="header-account-actions">
              {authLoading ? null : user ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{ fontSize: 'var(--text-body-sm)', color: 'var(--text-secondary)', maxWidth: 90, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {user.email?.split('@')[0]}
                  </div>
                  <button
                    onClick={signOut}
                    style={{
                      fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', padding: '4px 10px',
                      border: 'var(--border-default)', fontFamily: 'var(--font-primary)',
                      background: 'transparent', color: 'var(--text-muted)', cursor: 'pointer',
                    }}
                  >
                    Sign out
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setShowAuth(true)}
                  style={{
                    fontSize: 'var(--text-body-sm)', fontWeight: 'var(--weight-medium)', padding: '10px 20px',
                    border: 'var(--border-default)', fontFamily: 'var(--font-primary)',
                    background: 'transparent', color: 'var(--text-primary)',
                    cursor: 'pointer', letterSpacing: '0.02em', textTransform: 'uppercase',
                    transition: 'all var(--duration-fast) var(--ease-standard)',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.background = 'var(--surface-elevated)'; e.currentTarget.style.borderColor = 'var(--line-emphasis)' }}
                  onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.borderColor = 'var(--line-hairline)' }}
                >
                  Sign in
                </button>
              )}

              {/* Only ever rendered for a signed-in user now — see this
                  section's own comment above `initials` for the bug this
                  fixes. */}
              {user && (
                <div style={{
                  width: 32, height: 32,
                  background: 'var(--signal-positive)',
                  border: 'var(--border-default)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 'var(--text-body-sm)', fontWeight: 'var(--weight-semibold)',
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--surface-canvas)',
                }}>
                  {initials}
                </div>
              )}
            </div>

            {/* Compact-viewport-only counterpart to the block above: an
                icon, not text, signalling signed-in/signed-out at a glance
                without spending header width on it — the actual sign-in/
                sign-out controls live in the drawer this opens. Hidden by
                default, shown only inside the compact-viewport block
                (.header-account-indicator, index.css) — the exact inverse
                of .header-account-actions, so exactly one of the two is
                ever visible at a given width. */}
            <button
              className="header-account-indicator"
              onClick={() => setSidebarOpen(true)}
              aria-label={user ? 'Signed in — open account menu' : 'Signed out — open account menu'}
              title={user ? 'Signed in' : 'Signed out'}
            >
              <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                <circle cx="9" cy="6.2" r="3.2" stroke={user ? 'var(--signal-positive)' : 'var(--text-muted)'} strokeWidth="1.6" />
                <path d="M2.4 16c0-3.6 2.9-6.2 6.6-6.2s6.6 2.6 6.6 6.2" stroke={user ? 'var(--signal-positive)' : 'var(--text-muted)'} strokeWidth="1.6" strokeLinecap="round" />
              </svg>
            </button>
          </div>
        </header>

        <main style={{ flex: 1, overflow: 'hidden', padding: 16, minHeight: 0 }}>

          {error && rateLimited && (
            <div style={{ marginBottom: 12 }}>
              <BusyNotice onRetry={retryRun} />
            </div>
          )}

          {error && !rateLimited && (
            <div style={{
              background: 'var(--signal-negative-wash)', border: '1px solid var(--signal-negative)',
              padding: '10px 14px', marginBottom: 12,
              fontSize: 'var(--text-body-sm)', color: 'var(--signal-negative)', fontFamily: 'var(--font-primary)',
            }}>
              {error}
            </div>
          )}

          {loading && (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: 16 }}>
              <svg className="spin" width="32" height="32" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="var(--signal-positive)" strokeWidth="3" strokeDasharray="40 20"/>
              </svg>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
                <div style={{ fontSize: 'var(--text-body)', color: 'var(--text-muted)' }}>Fetching data and running analysis...</div>
                <div style={{ fontSize: 'var(--text-body-sm)', color: 'var(--text-muted)' }}>Taking longer than usual? A quick refresh usually does the trick.</div>
              </div>
            </div>
          )}

          {!loading && !hasRun && activeTab !== 'learn' && activeTab !== 'valuation' && (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: 16, textAlign: 'center' }}>
              <Logo variant="mark" size={48} ink="var(--signal-positive)" />
              <div>
                <div style={{ fontFamily: 'var(--font-primary)', fontSize: 'var(--text-heading-sm)', fontWeight: 'var(--weight-medium)', letterSpacing: 'var(--tracking-heading-sm)', color: 'var(--text-primary)', marginBottom: 6 }}>Varense</div>
                <div style={{ fontSize: 'var(--text-body)', color: 'var(--text-secondary)', maxWidth: 380, lineHeight: 1.6 }}>
                  Your portfolio's already loaded with three tech stocks. Hit Run Analysis to see what your risk actually looks like.
                  {!user && <span> <span style={{ color: 'var(--signal-positive)', cursor: 'pointer', fontWeight: 'var(--weight-semibold)' }} onClick={() => setShowAuth(true)}>Sign in</span> to save your portfolios.</span>}
                </div>
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
                {['VaR & CVaR', 'Sharpe & Sortino', 'Monte Carlo', 'Efficient Frontier', 'Beta & Alpha'].map(f => (
                  <div key={f} style={{ fontSize: 'var(--text-caption)', fontWeight: 'var(--weight-medium)', letterSpacing: 'var(--tracking-caption)', textTransform: 'uppercase', padding: '4px 10px', background: 'var(--signal-positive-wash)', border: '1px solid var(--signal-positive)', color: 'var(--signal-positive)', fontFamily: 'var(--font-primary)' }}>
                    {f}
                  </div>
                ))}
              </div>
            </div>
          )}

          {!loading && (hasRun || activeTab === 'learn' || activeTab === 'valuation') && (
            <div style={{ height: '100%' }} className="fade-up">
              {activeTab === 'dashboard'  && <Dashboard  data={data} tickers={tickers} weights={weights} portfolioValue={analysis.portfolioValue} onTickerClick={openDrawer} sectorData={sectorData} sectorLoading={sectorLoading} />}
              {activeTab === 'risk'       && <RiskAnalysis data={data} tickers={tickers} weights={weights} portfolioValue={analysis.portfolioValue} onTickerClick={openDrawer} />}
              {activeTab === 'montecarlo' && <MonteCarlo  data={data} heavyLoading={heavyLoading} heavyError={heavyError} heavyRateLimited={heavyRateLimited} onRetryHeavy={retryHeavy} />}
              {activeTab === 'frontier'   && <Frontier    data={data} tickers={tickers} weights={weights} heavyLoading={heavyLoading} heavyError={heavyError} heavyRateLimited={heavyRateLimited} onRetryHeavy={retryHeavy} />}
              {activeTab === 'valuation'  && <Valuation   tickers={tickers} onTickerClick={openDrawer} />}
              {activeTab === 'compare'    && (
                <CompareWrapper
                  dataA={data} tickersA={tickers} nameA="Portfolio A"
                  compB={compB} portfolios={portfolios}
                />
              )}
              {activeTab === 'backtest'   && <Backtest data={data} tickers={tickers} weights={weights} heavyLoading={heavyLoading} heavyError={heavyError} heavyRateLimited={heavyRateLimited} onRetryHeavy={retryHeavy} />}
              {activeTab === 'learn' && <Learn />}
            </div>
          )}
        </main>

        {/* .app-footer: padding lives in that class (index.css), not here,
            for the usual inline-always-wins-over-a-media-query reason every
            other responsive fix in this codebase has had to route around. */}
        <div className="app-footer" style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap',
          fontSize: 'var(--text-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)',
          borderTop: 'var(--border-default)', background: 'transparent', flexShrink: 0,
        }}>
          <span>Varense - educational tool only. Not financial advice. Past performance does not guarantee future results.</span>
          <span style={{ display: 'flex', gap: 12, marginLeft: 12 }}>
            {/* Feedback's own trigger portals in here on desktop — see
                .footer-feedback-link (index.css) and FeedbackButton.jsx's
                own comment for why this is its desktop home while phone
                keeps the header icon. */}
            <span id="footer-feedback-slot" />
            <Link to="/privacy" style={{ color: 'var(--text-muted)', textDecoration: 'none' }}>Privacy</Link>
            <Link to="/terms" style={{ color: 'var(--text-muted)', textDecoration: 'none' }}>Terms</Link>
          </span>
        </div>
      </div>
      </div>
    </div>
  )
}