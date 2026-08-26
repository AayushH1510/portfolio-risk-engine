// Plain-language explanations for metric labels, shown via MetricTooltip.
// Keyed by metric name — add a new entry here and wrap the label with
// <MetricTooltip metricKey="..."> to give any future metric the same
// dotted-underline hover/tap treatment.
export const metricExplanations = {
  annual_return: "How much your portfolio grew per year, on average, over the selected period — compounded, not a simple average of yearly returns.",
  volatility: "How much your portfolio's value swings up and down over time. Higher volatility means a bumpier ride, even if the destination is the same.",
  sharpe_ratio: "Return earned per unit of overall risk taken. Above 1.0 is generally considered good.",
  sortino_ratio: "Like the Sharpe ratio, but only counts downside volatility — swings upward aren't treated as \"risk\".",
  beta: "How much your portfolio moves relative to the overall market. A beta of 1.5 means it tends to swing about 50% more than the market, in either direction.",
  alpha: "The extra return (or shortfall) your portfolio produced beyond what its risk level alone would predict. Positive alpha means you beat the market after adjusting for risk.",
  diversification_score: "How spread out your risk is across your holdings, based on the average pairwise correlation between your tickers (shown below as \"avg corr\"). This score is window-dependent — it reflects how your stocks moved together over the selected period, not a fixed trait of the stocks themselves.",
  var_95: "The most you'd expect to lose in a single day, 95% of the time. On the remaining 5% of days, losses can be worse than this.",
  cvar_95: "The average loss on your worst 5% of days — how bad things tend to get once the VaR 95% threshold is breached, not just where that threshold sits.",
  max_drawdown: "The biggest drop your portfolio experienced from a peak to a low point over the selected period — a look at the worst pain you'd have felt holding it.",
  var_99: "The most you'd expect to lose in a single day, 99% of the time — a stricter threshold than VaR 95%, capturing rarer and more severe days.",
  cvar_99: "The average loss on your worst 1% of days — the typical severity of your most extreme downside scenarios.",
  treynor_ratio: "Return earned per unit of market risk (beta), rather than total volatility. Useful for comparing already-diversified portfolios.",
  information_ratio: "How much extra return your portfolio generates versus a benchmark, per unit of risk taken to deviate from it. Higher means more consistent outperformance.",
}
