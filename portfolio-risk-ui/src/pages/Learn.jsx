import { useState } from 'react'
import { learnContent } from '../content/learnContent'

const TIERS = [
  { key: 'baby',       label: 'Baby Language' },
  { key: 'individual', label: 'Individual Investor' },
  { key: 'expert',     label: 'Expert' },
]

const SECTIONS = [
  {
    category: 'Return & risk basics',
    items: [
      { key: 'annual_return', label: 'Annual Return' },
      { key: 'volatility',    label: 'Volatility' },
      { key: 'max_drawdown',  label: 'Max Drawdown' },
    ],
  },
  {
    category: 'Risk-adjusted performance',
    items: [
      { key: 'sharpe_ratio',       label: 'Sharpe Ratio' },
      { key: 'sortino_ratio',      label: 'Sortino Ratio' },
      { key: 'treynor_ratio',      label: 'Treynor Ratio' },
      { key: 'information_ratio',  label: 'Information Ratio' },
    ],
  },
  {
    category: 'Downside risk',
    items: [
      { key: 'var_95',  label: 'VaR 95%' },
      { key: 'cvar_95', label: 'CVaR 95%' },
      { key: 'var_99',  label: 'VaR 99%' },
      { key: 'cvar_99', label: 'CVaR 99%' },
    ],
  },
  {
    category: 'Market comparison',
    items: [
      { key: 'beta',  label: 'Beta' },
      { key: 'alpha', label: 'Alpha' },
    ],
  },
  {
    category: 'Diversification',
    items: [
      { key: 'diversification_score', label: 'Diversification Score' },
      { key: 'correlation_matrix',    label: 'Correlation Matrix' },
      { key: 'sector_exposure',       label: 'Sector Exposure' },
    ],
  },
  {
    category: 'Advanced analysis',
    items: [
      { key: 'rolling_metrics',    label: 'Rolling Volatility & Sharpe' },
      { key: 'monte_carlo',        label: 'Monte Carlo Simulation' },
      { key: 'scenario_analysis',  label: 'Bear / Base / Bull Scenarios' },
      { key: 'efficient_frontier', label: 'Efficient Frontier' },
      { key: 'stress_testing',     label: 'Historical Stress Testing' },
      { key: 'backtesting',        label: 'Backtesting' },
    ],
  },
]

function TierToggle({ tier, onChange }) {
  return (
    <div style={{
      display: 'inline-flex', gap: 4, padding: 4,
      background: 'var(--surface-elevated)', border: 'var(--border-default)',
    }}>
      {TIERS.map(t => (
        <button
          key={t.key}
          onClick={() => onChange(t.key)}
          style={{
            padding: '8px 18px', fontSize: 12, fontWeight: 700,
            letterSpacing: '0.02em', border: 'none', cursor: 'pointer',
            background: tier === t.key ? 'var(--signal-positive)' : 'transparent',
            color: tier === t.key ? 'var(--surface-canvas)' : 'var(--text-muted)',
            transition: 'background 0.15s, color 0.15s',
          }}
        >
          {t.label}
        </button>
      ))}
    </div>
  )
}

export default function Learn() {
  const [tier, setTier] = useState('individual')

  return (
    <div style={{ height: '100%', overflowY: 'auto', paddingRight: 8 }}>
      <div style={{ maxWidth: 820 }}>

        <div style={{
          marginBottom: 16,
          padding: '12px 16px',
          background: 'rgba(var(--signal-positive-rgb),0.08)',
          border: '1px solid rgba(var(--signal-positive-rgb),0.25)',
          fontSize: 12,
          color: 'var(--text-secondary)',
          lineHeight: 1.6,
        }}>
          Every concept used in this tool, explained at three levels. Pick the one that matches how
          deep you want to go, you can switch anytime.
        </div>

        <div style={{
          marginBottom: 24, position: 'sticky', top: 0, zIndex: 5,
          background: 'var(--surface-canvas)', paddingBottom: 10, paddingTop: 2,
        }}>
          <TierToggle tier={tier} onChange={setTier} />
        </div>

        {SECTIONS.map(section => (
          <div key={section.category} style={{ marginBottom: 28 }}>

            <div style={{
              fontSize: 10,
              fontWeight: 700,
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              color: 'var(--signal-positive)',
              marginBottom: 12,
              paddingBottom: 8,
              borderBottom: 'var(--border-default)',
            }}>
              {section.category}
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {section.items.map(item => {
                const content = learnContent[item.key]
                if (!content) return null
                return (
                  <div key={item.key} className="card" style={{ padding: '14px 16px' }}>
                    <div style={{
                      fontSize: 13,
                      fontWeight: 700,
                      color: 'var(--text-primary)',
                      marginBottom: 6,
                    }}>
                      {item.label}
                    </div>

                    <div style={{
                      fontSize: 12,
                      color: 'var(--text-secondary)',
                      lineHeight: 1.65,
                    }}>
                      {content[tier]}
                    </div>
                  </div>
                )
              })}
            </div>

          </div>
        ))}

        <div style={{
          fontSize: 10,
          color: 'var(--text-muted)',
          textAlign: 'center',
          padding: '16px 0 8px',
          borderTop: 'var(--border-default)',
        }}>
          This tool is for educational purposes only and does not constitute financial advice.
          Past performance does not guarantee future results.
        </div>

      </div>
    </div>
  )
}
