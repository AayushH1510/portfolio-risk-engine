// Plain-language explanations for metric labels, shown via MetricTooltip.
// Keyed by metric name — add a new entry here and wrap the label with
// <MetricTooltip metricKey="..."> to give any future metric the same
// dotted-underline hover/tap treatment.
export const metricExplanations = {
  annual_return: "How much your portfolio grew per year, on average, over the selected period - compounded, not a simple average of yearly returns.",
  volatility: "How much your portfolio's value swings up and down over time. Higher volatility means a bumpier ride, even if the destination is the same.",
  sharpe_ratio: "Return earned per unit of overall risk taken. Above 1.0 is generally considered good.",
  sortino_ratio: "Like the Sharpe ratio, but only counts downside volatility - swings upward aren't treated as \"risk\".",
  beta: "How much your portfolio moves relative to the overall market. A beta of 1.5 means it tends to swing about 50% more than the market, in either direction.",
  alpha: "The extra return (or shortfall) your portfolio produced beyond what its risk level alone would predict. Positive alpha means you beat the market after adjusting for risk.",
  diversification_score: "How spread out your risk is across your holdings, based on the average pairwise correlation between your tickers (shown below as \"avg corr\"). This score is window-dependent - it reflects how your stocks moved together over the selected period, not a fixed trait of the stocks themselves.",
  var_95: "The most you'd expect to lose in a single day, 95% of the time. On the remaining 5% of days, losses can be worse than this.",
  cvar_95: "The average loss on your worst 5% of days - how bad things tend to get once the VaR 95% threshold is breached, not just where that threshold sits.",
  max_drawdown: "The biggest drop your portfolio experienced from a peak to a low point over the selected period - a look at the worst pain you'd have felt holding it.",
  var_99: "The most you'd expect to lose in a single day, 99% of the time - a stricter threshold than VaR 95%, capturing rarer and more severe days.",
  cvar_99: "The average loss on your worst 1% of days - the typical severity of your most extreme downside scenarios.",
  treynor_ratio: "Return earned per unit of market risk (beta), rather than total volatility. Useful for comparing already-diversified portfolios.",
  information_ratio: "How much extra return your portfolio generates versus a benchmark, per unit of risk taken to deviate from it. Higher means more consistent outperformance.",

  // Valuation tab — per-ticker fundamentals, distinct from the portfolio-level
  // metrics above (e.g. stock_beta is one holding's own market sensitivity,
  // not "beta" further up which describes the whole portfolio).
  ps_ratio: "Price-to-Sales ratio - how much investors pay per $1 of revenue. Lower is cheaper relative to sales, though high-growth companies often carry a high P/S anyway.",
  ev_ebitda: "Enterprise Value divided by EBITDA - a measure of overall company value against operating profit. Lower is cheaper; negative means the company's EBITDA itself is negative (unprofitable at the operating level).",
  gross_margin: "Revenue kept after the direct cost of goods sold. Higher means stronger pricing power and more left over for R&D, sales, and profit - software companies often exceed 70%, manufacturers typically sit at 20-40%.",
  rev_growth: "Year-over-year revenue growth - how fast the company is growing its top line. Strong growth is generally 15%+; shrinking revenue is a serious warning sign.",
  profit_margin: "Revenue kept after every cost, not just cost of goods sold - the bottom-line percentage of revenue that becomes actual profit.",
  debt_equity: "Total debt as a percentage of shareholder equity. Above 100% means the company owes more than its equity is worth; above 200% is heavy leverage, vulnerable in a rising-rate environment.",
  current_ratio: "Short-term assets divided by short-term liabilities. Below 1.0 means liabilities coming due exceed the cash and assets on hand to cover them.",
  roe: "Return on Equity - net income as a percentage of shareholder equity. Higher means the company generates more profit from the capital shareholders have put in.",
  stock_beta: "How much this ticker moves relative to the overall market. A beta of 1.5 means it tends to swing about 50% more than the market, in either direction - the same idea as portfolio beta above, just for one holding.",
  market_cap: "Total value of all shares outstanding - share price x shares outstanding. Gives a sense of scale: mega-cap ($1T+), large-cap ($10B+), mid-cap ($2B+), small-cap (under $2B).",
  vg_score: "Value/Growth Score = P/S ÷ Revenue Growth %. Lower is better - it means more growth per dollar of valuation, similar in spirit to a revenue-based PEG ratio. Below 0.15 is excellent.",
}
