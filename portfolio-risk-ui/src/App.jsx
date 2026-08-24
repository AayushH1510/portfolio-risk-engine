import { useState, useEffect } from 'react'
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
  const [sectorData, setSectorData]     = useState([])

  const { user, loading: authLoading, signInWithGoogle, signInWithEmail, signUpWithEmail, signOut } = useAuth()
  const analysis = useAnalysis()
  const { portfolios, savePortfolio, deletePortfolio } = usePortfolios(user)
  const compB = useComparison()
  const { data, loading, error, hasRun, runAnalysis, tickers, weights } = analysis

  useEffect(() => {
    if (!data || !tickers.length) {
      setSectorData([])
      return
    }
    let cancelled = false
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
      .catch(() => { if (!cancelled) setSectorData([]) })
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
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: 'var(--bg)' }}>

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
          background: 'rgba(255,255,255,0.4)', backdropFilter: 'blur(8px)',
          borderBottom: '1px solid var(--border-light)',
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
                    padding: '0 10px', fontSize: 11, fontWeight: 600,
                    letterSpacing: '0.03em', textTransform: 'uppercase',
                    background: 'transparent', border: 'none',
                    borderBottom: active ? '2px solid var(--accent-dark)' : '2px solid transparent',
                    color: active ? 'var(--accent-dark)' : !unlocked ? '#aab8aa' : '#3a4a3a',
                    cursor: !unlocked ? 'not-allowed' : 'pointer',
                    transition: 'all 0.15s', whiteSpace: 'nowrap', flexShrink: 0,
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
                <div style={{ fontSize: 11, color: '#3a4a3a', maxWidth: 90, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {user.email?.split('@')[0]}
                </div>
                <button
                  onClick={signOut}
                  style={{
                    fontSize: 10, fontWeight: 600, padding: '4px 10px',
                    borderRadius: 6, border: '1px solid var(--border-light)',
                    background: 'transparent', color: '#5a7a5a', cursor: 'pointer',
                  }}
                >
                  Sign out
                </button>
              </div>
            ) : (
              <button
                onClick={() => setShowAuth(true)}
                style={{
                  fontSize: 11, fontWeight: 700, padding: '6px 14px',
                  borderRadius: 6, border: '1px solid var(--accent-dark)',
                  background: 'transparent', color: 'var(--accent-dark)',
                  cursor: 'pointer', letterSpacing: '0.04em', transition: 'all 0.15s',
                }}
                onMouseEnter={e => { e.currentTarget.style.background = 'var(--accent-dark)'; e.currentTarget.style.color = '#fff' }}
                onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--accent-dark)' }}
              >
                Sign in
              </button>
            )}

            <div style={{
              width: 32, height: 32, borderRadius: '50%',
              background: user ? 'var(--accent-dark)' : 'rgba(255,255,255,0.4)',
              border: '1px solid var(--border-light)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 12, fontWeight: 700,
              color: user ? '#fff' : '#5a7a5a',
            }}>
              {initials}
            </div>
          </div>
        </header>

        <main style={{ flex: 1, overflow: 'hidden', padding: 16, minHeight: 0 }}>

          {error && (
            <div style={{
              background: 'rgba(224,92,92,0.12)', border: '1px solid var(--negative)',
              borderRadius: 8, padding: '10px 14px', marginBottom: 12,
              fontSize: 12, color: 'var(--negative)',
            }}>
              {error}
            </div>
          )}

          {loading && (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: 16 }}>
              <svg className="spin" width="32" height="32" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="var(--accent)" strokeWidth="3" strokeDasharray="40 20"/>
              </svg>
              <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>Fetching data and running analysis...</div>
            </div>
          )}

          {!loading && !hasRun && activeTab !== 'learn' && activeTab !== 'valuation' && (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: 16, textAlign: 'center' }}>
              <div style={{ width: 56, height: 56, background: 'rgba(82,183,136,0.12)', border: '1px solid var(--accent)', borderRadius: 14, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <path d="M3 17L9 11L13 15L21 7" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <div>
                <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-dark)', marginBottom: 6 }}>Varense</div>
                <div style={{ fontSize: 13, color: '#5a7a5a', maxWidth: 380, lineHeight: 1.6 }}>
                  Your portfolio's already loaded with three tech stocks. Hit Run Analysis to see what your risk actually looks like.
                  {!user && <span> <span style={{ color: 'var(--accent-dark)', cursor: 'pointer', fontWeight: 600 }} onClick={() => setShowAuth(true)}>Sign in</span> to save your portfolios.</span>}
                </div>
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
                {['VaR & CVaR', 'Sharpe & Sortino', 'Monte Carlo', 'Efficient Frontier', 'Beta & Alpha'].map(f => (
                  <div key={f} style={{ fontSize: 11, fontWeight: 600, padding: '4px 10px', background: 'rgba(82,183,136,0.1)', border: '1px solid rgba(82,183,136,0.3)', borderRadius: 4, color: 'var(--accent-dark)' }}>
                    {f}
                  </div>
                ))}
              </div>
            </div>
          )}

          {!loading && (hasRun || activeTab === 'learn' || activeTab === 'valuation') && (
            <div style={{ height: '100%' }} className="fade-up">
              {activeTab === 'dashboard'  && <Dashboard  data={data} tickers={tickers} weights={weights} portfolioValue={analysis.portfolioValue} onTickerClick={openDrawer} sectorData={sectorData} />}
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

        <div style={{ padding: '6px 20px', fontSize: 10, color: '#8aaa8a', borderTop: '1px solid var(--border-light)', background: 'rgba(255,255,255,0.3)', flexShrink: 0 }}>
          Varense — educational tool only. Not financial advice. Past performance does not guarantee future results.
        </div>
      </div>
    </div>
  )
}