import { TrustPage, TrustSection } from '../components/LandingLayout'

export default function Terms() {
  return (
    <TrustPage title="Terms of Service" updated="August 24, 2026">
      <TrustSection heading="Educational tool only - not financial advice">
        <p style={{ marginBottom: 12 }}>
          Varense is an educational portfolio analysis tool. Everything it shows you - risk metrics,
          Monte Carlo simulations, backtests, stress tests, the efficient frontier, valuation flags - is
          computed from historical and simulated data for informational purposes only.
        </p>
        <p>
          Nothing in Varense constitutes financial, investment, tax, or legal advice, and nothing here
          should be treated as a recommendation to buy, sell, or hold any security. Past performance does
          not guarantee future results. Always do your own research and consult a licensed financial
          advisor before making investment decisions.
        </p>
      </TrustSection>

      <TrustSection heading="Beta / testing phase">
        <p>
          Varense is currently in beta. Features, calculations, and data sources may change, be added, or
          be removed without notice as we continue building and testing the product. You may encounter
          bugs, inaccuracies, or temporary downtime - if you spot something off, we'd genuinely appreciate
          you reporting it via the feedback button in the app.
        </p>
      </TrustSection>

      <TrustSection heading="No warranty">
        <p>
          Varense is provided "as is" and "as available," without warranty of any kind, express or
          implied - including, without limitation, warranties of accuracy, completeness, or fitness for a
          particular purpose. We don't guarantee the data or calculations are error-free.
        </p>
      </TrustSection>

      <TrustSection heading="You're responsible for your decisions">
        <p>
          Any investment decisions you make, whether or not informed by Varense, are entirely your own
          responsibility. We are not liable for any losses, damages, or other consequences arising from
          your use of the app or reliance on any information it presents.
        </p>
      </TrustSection>

      <TrustSection heading="Changes to the service">
        <p>
          Because Varense is in active beta development, the service - including its features, pricing
          (currently free), and availability - may change at any time. We'll try to communicate major
          changes, but we don't guarantee advance notice during this phase.
        </p>
      </TrustSection>

      <TrustSection heading="Contact">
        <p>
          Questions about these terms? Reach out at{' '}
          <a href="mailto:aayushdxb1510@gmail.com" style={{ color: 'var(--signal-positive)' }}>aayushdxb1510@gmail.com</a>.
        </p>
      </TrustSection>
    </TrustPage>
  )
}
