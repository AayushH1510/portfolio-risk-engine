import { TrustPage, TrustSection } from '../components/LandingLayout'

export default function Privacy() {
  return (
    <TrustPage title="Privacy Policy" updated="August 26, 2026">
      <TrustSection heading="What we collect">
        <p style={{ marginBottom: 12 }}>
          If you sign in, we collect your <strong>email address</strong> through Supabase Authentication —
          we never see or store your password directly.
        </p>
        <p style={{ marginBottom: 12 }}>
          If you save a portfolio, we store the <strong>tickers and weights</strong> you chose to save,
          along with the portfolio value and time period you configured. That's it — we don't collect
          your name, address, phone number, or anything else beyond what's needed to make "saved portfolios"
          work.
        </p>
        <p>
          We do not collect any payment information, because Varense does not currently process payments.
        </p>
      </TrustSection>

      <TrustSection heading="Market data">
        <p style={{ marginBottom: 12 }}>
          Varense uses two third-party market data providers, each for a different part of the app.{' '}
          <strong>yfinance</strong> (Yahoo Finance) powers portfolio analysis, backtesting, and Monte Carlo
          simulation — the historical price data your risk metrics are calculated from.{' '}
          <strong>Finnhub</strong> powers the individual stock drawer and the Valuation tab — real-time
          quotes and company fundamentals for a ticker you click into.
        </p>
        <p>
          Both are public market information, not personal data about you, and neither is linked to your
          account beyond the tickers you choose to analyse.
        </p>
      </TrustSection>

      <TrustSection heading="How we use your data">
        <p>
          Your email is used solely to authenticate you and let you save/load portfolios across sessions.
          Your saved portfolio data is used solely to show it back to you when you sign in. We don't use
          your data for advertising, profiling, or any purpose beyond running the app.
        </p>
      </TrustSection>

      <TrustSection heading="We don't sell your data">
        <p>
          We do not sell, rent, or trade your personal data to third parties, ever. Full stop.
        </p>
      </TrustSection>

      <TrustSection heading="Your control over your data">
        <p>
          You can delete any saved portfolio at any time from within the app. If you'd like your account
          and all associated data fully removed, contact us and we'll take care of it.
        </p>
      </TrustSection>

      <TrustSection heading="Changes to this policy">
        <p>
          Varense is an early-stage, actively developed product. We may update this policy as the product
          evolves — significant changes will be reflected by updating the date at the top of this page.
        </p>
      </TrustSection>

      <TrustSection heading="Contact">
        <p>
          Questions about this policy or your data? Reach out at{' '}
          <a href="mailto:aayushdxb1510@gmail.com" style={{ color: 'var(--accent)' }}>aayushdxb1510@gmail.com</a>.
        </p>
      </TrustSection>
    </TrustPage>
  )
}
