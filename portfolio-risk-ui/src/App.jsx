import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import axios from 'axios'
import Sidebar from './components/Sidebar'
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

  const { user, loading: authLoading, signInWithGoogle, signInWithEmail, signUpWithEmail, signOut } = useAuth()
  const analysis = useAnalysis()
  const { portfolios, savePortfolio, deletePortfolio } = usePortfolios(user)
  const compB = useComparison()
  const { data, loading, error, hasRun, runAnalysis, tickers, weights } = analysis

  useEffect(() => {
    if (!data || !tickers.length) {
      setSectorData(null)
      setSectorLoading(false)
      return
    }
    let cancelled = false
    setSectorLoading(true)
    axios.get(`${API}/api/fundamentals?tickers=${tickers.join(',')}`)
      .then(res => {
        if (cancelled) return
        const sectorByTicker = {}
        res.data.tickers.forEach(t => { sectorByTicker[t.ticker] = t.sector })

        const weightBySector = {}
        tickers.forEach((tk, i) => {
          const sector = sectorByTicker[tk.toUpperCase()]
          if (!sector || sector === 'Unknown') return
          weightBySector[sector] = (weightBySector[sector] || 0) + (weights[i] ?? 0)
        })

        setSectorData(
          Object.entries(weightBySector)
            .map(([sector, weight]) => ({ sector, weight }))
            .sort((a, b) => b.weight - a.weight)
        )
      })
      .catch(() => { if (!cancelled) setSectorData(null) })
      .finally(() => { if (!cancelled) setSectorLoading(false) })
    return () => { cancelled = true }
  }, [data])

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

  const initials = user?.email?.slice(0, 2).toUpperCase() || 'PR'

  return (
    <div className="grain-canvas" style={{ height: '100vh', overflow: 'hidden', background: 'var(--surface-canvas)', position: 'relative' }}>
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

      <OnboardingTour />
      <FeedbackButton user={user} />

      <Sidebar
        {...analysis}
        setWeightsAll={analysis.setWeightsAll}
        onRun={(customDates) => runAnalysis(customDates)}
        loading={loading}
        portfolios={portfolios}
        onSavePortfolio={savePortfolio}
        onLoadPortfolio={handleLoadPortfolio}
        onDeletePortfolio={deletePortfolio}
        user={user}
        onTickerClick={openDrawer}
      />

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>

        <header style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '0 12px', height: 48,
          background: 'transparent',
          borderBottom: 'var(--border-default)',
          flexShrink: 0, minWidth: 0, overflow: 'hidden',
        }}>
          <nav style={{ display: 'flex', gap: 0, height: '100%', alignItems: 'stretch', overflowX: 'auto', flexShrink: 1, minWidth: 0 }}>
            {TABS.map(tab => {
              const active   = activeTab === tab.id
              const unlocked = tab.id === 'learn' || tab.id === 'valuation' || tab.id === 'dashboard' || tab.id === 'compare' || hasRun
              return (
                <button
                  key={tab.id}
                  onClick={() => unlocked && setActiveTab(tab.id)}
                  style={{
                    padding: '0 10px', fontSize: 'var(--text-body-sm)', fontWeight: 'var(--weight-medium)',
                    letterSpacing: 'var(--tracking-tab)', textTransform: 'uppercase',
                    fontFamily: 'var(--font-primary)',
                    background: 'transparent', border: 'none',
                    borderBottom: active ? '2px solid var(--signal-positive)' : '2px solid transparent',
                    color: active ? 'var(--text-primary)' : !unlocked ? 'var(--text-faint)' : 'var(--text-muted)',
                    cursor: !unlocked ? 'not-allowed' : 'pointer',
                    transition: 'all var(--duration-fast) var(--ease-standard)', whiteSpace: 'nowrap', flexShrink: 0,
                  }}
                >
                  {tab.label}
                </button>
              )
            })}
          </nav>

          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0, marginLeft: 8 }}>
            {hasRun && (
              <>
                <ExportPDF data={data} tickers={tickers} weights={weights} portfolioValue={analysis.portfolioValue} />
                <ExportCSV data={data} tickers={tickers} weights={weights} />
              </>
            )}

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

            <div style={{
              width: 32, height: 32,
              background: user ? 'var(--signal-positive)' : 'var(--surface-elevated)',
              border: 'var(--border-default)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 'var(--text-body-sm)', fontWeight: 'var(--weight-semibold)',
              fontFamily: 'var(--font-mono)',
              color: user ? 'var(--surface-canvas)' : 'var(--text-muted)',
            }}>
              {initials}
            </div>
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
                <div style={{ fontSize: 'var(--text-body-sm)', color: 'var(--text-muted)' }}>Taking longer than expected? Refresh the page to retry.</div>
              </div>
            </div>
          )}

          {!loading && !hasRun && activeTab !== 'learn' && activeTab !== 'valuation' && (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: 16, textAlign: 'center' }}>
              <div style={{ width: 56, height: 56, background: 'var(--signal-positive-wash)', border: '1px solid var(--signal-positive)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <path d="M3 17L9 11L13 15L21 7" stroke="var(--signal-positive)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
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
              {activeTab === 'montecarlo' && <MonteCarlo  data={data} />}
              {activeTab === 'frontier'   && <Frontier    data={data} tickers={tickers} weights={weights} />}
              {activeTab === 'valuation'  && <Valuation   tickers={tickers} onTickerClick={openDrawer} />}
              {activeTab === 'compare'    && (
                <CompareWrapper
                  dataA={data} tickersA={tickers} nameA="Portfolio A"
                  compB={compB} portfolios={portfolios}
                />
              )}
              {activeTab === 'backtest'   && <Backtest data={data} tickers={tickers} weights={weights} />}
              {activeTab === 'learn' && <Learn />}
            </div>
          )}
        </main>

        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '6px 20px', fontSize: 'var(--text-caption)', color: 'var(--text-muted)', fontFamily: 'var(--font-primary)',
          borderTop: 'var(--border-default)', background: 'transparent', flexShrink: 0,
        }}>
          <span>Varense — educational tool only. Not financial advice. Past performance does not guarantee future results.</span>
          <span style={{ display: 'flex', gap: 12, flexShrink: 0, marginLeft: 12 }}>
            <Link to="/privacy" style={{ color: 'var(--text-muted)', textDecoration: 'none' }}>Privacy</Link>
            <Link to="/terms" style={{ color: 'var(--text-muted)', textDecoration: 'none' }}>Terms</Link>
          </span>
        </div>
      </div>
      </div>
    </div>
  )
}