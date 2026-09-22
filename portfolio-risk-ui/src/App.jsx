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
import ExportPDF from './components/ExportPDF'
import ExportCSV from './components/ExportCSV'
import CompareWrapper from './components/CompareWrapper'
import StockDrawer from './components/StockDrawer'
import OnboardingTour from './components/OnboardingTour'
import FeedbackButton from './components/FeedbackButton'
import { useComparison } from './hooks/useComparison'
import { useAnalysis } from './hooks/useAnalysis'
import { usePortfolios } from './hooks/usePortfolios'
import { useAuth } from './hooks/useAuth'

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
  // null = not fetched yet (or the fetch itself failed) — render nothing.
  // [] = fetched successfully but no ticker resolved to a usable sector
  // (e.g. an all-ETF portfolio) — SectorChart shows an explanatory empty
  // state for that case, distinct from "haven't checked".
  const [sectorData, setSectorData]     = useState(null)
  const [sectorLoading, setSectorLoading] = useState(false)
  // Sector composition only depends on which tickers (and at what weights)
  // are in the portfolio — not on the analysis result itself. `data`
  // updates twice per run now (the fast summary, then the background full
  // response replacing it once it lands), and without this the effect
  // below re-fires — a second /api/fundamentals round-trip and a "Loading
  // sector exposure…" flicker — for the exact same portfolio composition.
  const sectorFetchKeyRef = useRef(null)
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

  useEffect(() => {
    if (!data || !tickers.length) {
      setSectorData(null)
      setSectorLoading(false)
      sectorFetchKeyRef.current = null
      return
    }
    const fetchKey = `${tickers.join(',')}|${weights.join(',')}`
    if (fetchKey === sectorFetchKeyRef.current) return   // heavy tier just replaced `data` for the same run — nothing to refetch
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
    setSectorLoading(true)
    axios.get(`${API}/api/fundamentals?tickers=${tickers.join(',')}`)
      .then(res => {
        if (sectorFetchKeyRef.current !== fetchKey) return
        const sectorByTicker = {}
        res.data.tickers.forEach(t => { sectorByTicker[t.ticker] = t.sector })

        const weightBySector  = {}
        const tickersBySector = {}
        tickers.forEach((tk, i) => {
          const sector = sectorByTicker[tk.toUpperCase()]
          if (!sector || sector === 'Unknown') return
          weightBySector[sector]  = (weightBySector[sector] || 0) + (weights[i] ?? 0)
          tickersBySector[sector] = [...(tickersBySector[sector] || []), tk]
        })

        setSectorData(
          Object.entries(weightBySector)
            .map(([sector, weight]) => ({ sector, weight, tickers: tickersBySector[sector] }))
            .sort((a, b) => b.weight - a.weight)
        )
      })
      .catch(() => { if (sectorFetchKeyRef.current === fetchKey) setSectorData(null) })
      .finally(() => { if (sectorFetchKeyRef.current === fetchKey) setSectorLoading(false) })
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

  // Open the drawer the moment the tour starts. A no-op visually at
  // 1024px+ (CSS ignores sidebarOpen there), so this doesn't need its own
  // width check.
  useEffect(() => {
    if (tourActive) setSidebarOpen(true)
  }, [tourActive])

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
      <FeedbackButton user={user} />

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
              scrollbarWidth: 'none', WebkitOverflowScrolling: 'touch', scrollSnapType: 'x proximity',
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
                  style={{
                    padding: '0 10px', fontSize: 'var(--text-body-sm)', fontWeight: 'var(--weight-medium)',
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
            {hasRun && (
              <div className="header-export-actions">
                <ExportPDF data={data} tickers={tickers} weights={weights} portfolioValue={analysis.portfolioValue} />
                <ExportCSV data={data} tickers={tickers} weights={weights} />
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

          {error && (
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
              {activeTab === 'montecarlo' && <MonteCarlo  data={data} heavyLoading={heavyLoading} heavyError={heavyError} />}
              {activeTab === 'frontier'   && <Frontier    data={data} tickers={tickers} weights={weights} heavyLoading={heavyLoading} heavyError={heavyError} />}
              {activeTab === 'valuation'  && <Valuation   tickers={tickers} onTickerClick={openDrawer} />}
              {activeTab === 'compare'    && (
                <CompareWrapper
                  dataA={data} tickersA={tickers} nameA="Portfolio A"
                  compB={compB} portfolios={portfolios}
                />
              )}
              {activeTab === 'backtest'   && <Backtest data={data} tickers={tickers} weights={weights} heavyLoading={heavyLoading} heavyError={heavyError} />}
              {activeTab === 'learn' && <Learn />}
            </div>
          )}
        </main>

        {/* .app-footer: padding lives in that class (index.css), not here —
            the compact-viewport block adds extra padding-right there to
            keep this row's own text clear of the fixed feedback bubble
            (FeedbackButton.jsx, bottom-right); an inline padding shorthand
            here would permanently shadow that override regardless of
            media-query match, the same trap every other responsive fix in
            this codebase has had to route around. */}
        <div className="app-footer" style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap',
          fontSize: 'var(--text-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)',
          borderTop: 'var(--border-default)', background: 'transparent', flexShrink: 0,
        }}>
          <span>Varense - educational tool only. Not financial advice. Past performance does not guarantee future results.</span>
          <span style={{ display: 'flex', gap: 12, marginLeft: 12 }}>
            <Link to="/privacy" style={{ color: 'var(--text-muted)', textDecoration: 'none' }}>Privacy</Link>
            <Link to="/terms" style={{ color: 'var(--text-muted)', textDecoration: 'none' }}>Terms</Link>
          </span>
        </div>
      </div>
      </div>
    </div>
  )
}