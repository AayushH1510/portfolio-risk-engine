import {
  LandingPage, LandingHeader, LandingHero, LandingFeatureGrid, LandingHowItWorks, LandingFooter,
} from '../components/LandingLayout'

const FEATURES = [
  {
    title: 'Cholesky-decomposed Monte Carlo',
    body: '1,000 correlated simulation paths, not naive univariate sampling.',
  },
  {
    title: 'Dual-confidence VaR/CVaR',
    body: '95% and 99% expected shortfall, computed directly from realized return distribution.',
  },
  {
    title: 'Historical stress testing',
    body: 'Realized portfolio returns during 2008 GFC, COVID crash, and 2022 rate shock, with reweighting for missing data.',
  },
  {
    title: 'Treynor and Information ratio',
    body: 'Full risk-adjusted return suite alongside Sharpe and Sortino.',
  },
  {
    title: 'Backtesting with attribution',
    body: 'Year-by-year performance vs equal-weight and S&P 500, buy-and-hold basis.',
  },
  {
    title: 'Efficient frontier optimization',
    body: '5,000-portfolio Monte Carlo simulation, CAGR-consistent with realized returns.',
  },
]

const STEPS = [
  'Input tickers and weights',
  'Full risk decomposition in seconds',
  'Export for reporting',
]

export default function LandingPro() {
  return (
    <LandingPage>
      <LandingHeader crossLinkLabel="For individual investors →" crossLinkTo="/" />
      <LandingHero
        headline="Institutional-grade portfolio risk analytics, without the Bloomberg terminal price tag"
        subhead="Cholesky-correlated Monte Carlo, dual-confidence VaR/CVaR, CAGR-consistent efficient frontier optimization, and real historical stress scenarios, for any portfolio, in seconds."
      />
      <LandingFeatureGrid features={FEATURES} />
      <LandingHowItWorks steps={STEPS} />
      <LandingFooter />
    </LandingPage>
  )
}
