const concepts = [
  {
    category: 'Basics',
    items: [
      {
        term: 'Stock',
        def: 'A tiny slice of ownership in a company. When the company grows, your slice is worth more. When it shrinks, so does your slice.',
        ex: 'Buy AAPL at $150, it rises to $180 - you made $30, a 20% return.',
      },
      {
        term: 'Portfolio',
        def: 'Your collection of stocks. Spreading money across several stocks is called diversification - if one company has a bad year, the others might be fine.',
        ex: '40% AAPL, 40% MSFT, 20% GOOGL. If AAPL drops 20% but MSFT rises 10%, your total loss is only 4%.',
      },
      {
        term: 'Return',
        def: 'How much your investment grew or shrank as a percentage. An annual return of 15% means $10,000 became $11,500 in a year.',
        ex: 'The S&P 500 has historically returned about 10% per year on average.',
      },
    ],
  },
  {
    category: 'Risk metrics',
    items: [
      {
        term: 'Volatility',
        def: 'How much your portfolio jumps around day to day. High volatility means big swings - some days up 3%, others down 3%. Low volatility means a smoother ride.',
        ex: 'Tesla: high volatility. Gold: low volatility. A savings account: almost zero.',
      },
      {
        term: 'Drawdown',
        def: 'The biggest drop from a peak before recovery. If your portfolio hit $15,000 then fell to $10,500, that is a 30% drawdown. Most people panic-sell at the bottom, which locks in the loss permanently.',
        ex: 'S&P 500 dropped 34% during COVID in 2020. People who held recovered fully within 5 months.',
      },
      {
        term: 'VaR - Value at Risk',
        def: 'On a typical bad day (the worst 5% of all historical days), the most you would expect to lose. Think of it as your bad day budget - not a guarantee, but a realistic estimate.',
        ex: 'VaR of $200 at 95% means: on 95% of days you will not lose more than $200.',
      },
      {
        term: 'CVaR - Expected Shortfall',
        def: 'Goes one step further than VaR. On the very worst days beyond VaR, what do you lose on average? Always a bigger number than VaR. Banks and regulators use CVaR because it captures how bad the bad days truly get.',
        ex: 'If VaR is $200, CVaR might be $320 - on the worst 5% of days, your average loss is $320, not just $200.',
      },
    ],
  },
  {
    category: 'Performance ratios',
    items: [
      {
        term: 'Sharpe Ratio',
        def: 'Are you being paid enough for the risk you are taking? Compares your return to what you would earn with zero risk, like a government bond. Above 1.0 is good. Above 2.0 is excellent. Below 0 means a savings account would have done better.',
        ex: 'Portfolio A: 15% return, low volatility, Sharpe 1.8. Portfolio B: 20% return, high volatility, Sharpe 0.9. Portfolio A is actually the better choice.',
      },
      {
        term: 'Sortino Ratio',
        def: 'Like Sharpe but fairer - only penalises the bad days (losses), not the good ones (gains). If your portfolio is volatile mainly because it keeps surging upward, Sortino scores it higher than Sharpe. Usually the higher of the two numbers.',
        ex: 'A portfolio that often surges upward looks better on Sortino than Sharpe.',
      },
    ],
  },
  {
    category: 'Market comparison',
    items: [
      {
        term: 'Beta',
        def: 'How sensitive your portfolio is to the overall stock market. Beta of 1.3 means when the market drops 10%, you tend to drop 13%. Beta of 0.7 means you would only drop 7% - more defensive.',
        ex: 'Utility companies: low beta (0.3 to 0.6). Tech stocks: high beta (1.2 to 1.8).',
      },
      {
        term: 'Alpha',
        def: 'The return you earned above and beyond what the market predicted you should earn based on your risk level. Positive alpha means your specific stock picks added real value.',
        ex: 'If the market predicted 18% return but you returned 25%, your alpha is +7%. You genuinely outperformed.',
      },
    ],
  },
  {
    category: 'Advanced tools',
    items: [
      {
        term: 'Efficient Frontier',
        def: 'A chart of every possible way to split your money across your chosen stocks. The curved edge along the top-left of all those dots shows portfolios giving the best possible return for each level of risk. Anything below the curve means you are taking unnecessary risk.',
        ex: 'The star marks the mathematically optimal split - the one with the best risk-adjusted return based on historical data.',
      },
      {
        term: 'Monte Carlo simulation',
        def: 'Runs 1,000 simulations of your portfolio\'s possible future over the next year, based on its historical returns and volatility. Not a prediction - an honest range of outcomes showing best case, worst case, and everything in between.',
        ex: '75% chance of profit and median outcome of $16,000 means: in most simulated futures, you end the year ahead.',
      },
      {
        term: 'Correlation',
        def: 'Measures how much two stocks move together, from -1 to +1. Near +1 means they rise and fall in sync - owning both gives little protection. Near 0 means they are independent. Near -1 means they move opposite - the best natural protection.',
        ex: 'Apple and Microsoft correlate around 0.7. Adding gold (GLD) gives near-zero correlation with tech.',
      },
    ],
  },
]

export default function Learn() {
  return (
    <div style={{ height: '100%', overflowY: 'auto', paddingRight: 8 }}>
      <div style={{ maxWidth: 800 }}>

        <div style={{
          marginBottom: 20,
          padding: '12px 16px',
          background: 'rgba(82,183,136,0.08)',
          border: '1px solid rgba(82,183,136,0.25)',
          borderRadius: 8,
          fontSize: 12,
          color: 'var(--text-secondary)',
          lineHeight: 1.6,
        }}>
          Every concept used in this tool explained in plain English - no assumed knowledge, no jargon.
          If you are new to investing, start here before running your first analysis.
        </div>

        {concepts.map((section) => (
          <div key={section.category} style={{ marginBottom: 28 }}>

            <div style={{
              fontSize: 10,
              fontWeight: 700,
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              color: 'var(--accent)',
              marginBottom: 12,
              paddingBottom: 8,
              borderBottom: '1px solid var(--border)',
            }}>
              {section.category}
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {section.items.map((item) => (
                <div
                  key={item.term}
                  className="card"
                  style={{ padding: '14px 16px' }}
                >
                  <div style={{
                    fontSize: 13,
                    fontWeight: 700,
                    color: 'var(--text-primary)',
                    marginBottom: 6,
                  }}>
                    {item.term}
                  </div>

                  <div style={{
                    fontSize: 12,
                    color: 'var(--text-secondary)',
                    lineHeight: 1.65,
                    marginBottom: 10,
                  }}>
                    {item.def}
                  </div>

                  <div style={{
                    fontSize: 11,
                    color: 'var(--accent)',
                    background: 'rgba(82,183,136,0.08)',
                    borderLeft: '2px solid var(--accent)',
                    padding: '7px 11px',
                    borderRadius: '0 4px 4px 0',
                    fontFamily: 'monospace',
                    lineHeight: 1.5,
                  }}>
                    <span style={{ color: 'var(--text-muted)', fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', display: 'block', marginBottom: 3 }}>
                      Example
                    </span>
                    {item.ex}
                  </div>
                </div>
              ))}
            </div>

          </div>
        ))}

        <div style={{
          fontSize: 10,
          color: 'var(--text-muted)',
          textAlign: 'center',
          padding: '16px 0 8px',
          borderTop: '1px solid var(--border)',
        }}>
          This tool is for educational purposes only and does not constitute financial advice.
          Past performance does not guarantee future results.
        </div>

      </div>
    </div>
  )
}