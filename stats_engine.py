"""
stats_engine.py
---------------
The maths layer. Takes clean price data from data_fetcher.py
and computes all the risk metrics.

Metrics implemented:
  Core       — daily returns, portfolio returns, annualised return
  Risk       — volatility, VaR, CVaR, maximum drawdown
  Ratios     — Sharpe ratio, Sortino ratio
  Rolling    — rolling volatility, rolling Sharpe        ← Tier 2
  Market     — Beta, Alpha (Jensen's)                    ← Tier 2
  Frontier   — Efficient Frontier (Markowitz)            ← Tier 3
  Matrix     — covariance matrix, correlation matrix
"""

import numpy as np
import pandas as pd


# ─── Constants ────────────────────────────────────────────────────────────────

TRADING_DAYS   = 252
RISK_FREE_RATE = 0.045


# ─── Tooltips ─────────────────────────────────────────────────────────────────

METRIC_TOOLTIPS = {
    "annualised_return": (
        "How much your portfolio grew per year, on average. "
        "A 20% return means $10,000 became $12,000 in a year. Higher is better."
    ),
    "annualised_volatility": (
        "How much your portfolio value jumps around day to day, scaled to a full year. "
        "Low means a smooth ride. High means big swings — up AND down."
    ),
    "sharpe_ratio": (
        "Are you being paid enough for the risk you're taking? "
        "It compares your return to a 'safe' investment like a Treasury bond. "
        "Above 1.0 is good. Above 2.0 is excellent."
    ),
    "sortino_ratio": (
        "Like the Sharpe ratio, but fairer — it only penalises the bad days (losses), "
        "not the good ones (gains). Higher is better. Usually higher than Sharpe."
    ),
    "var": (
        "Value at Risk: on a typical bad day (bottom 5% of days historically), "
        "this is the most you'd expect to lose. Think of it as your 'bad day budget'."
    ),
    "cvar": (
        "Expected Shortfall: on the very worst days (beyond VaR), "
        "what do you lose on average? Always worse than VaR. "
        "Used by banks and regulators because it captures how bad the bad days really are."
    ),
    "max_drawdown": (
        "The biggest drop from a peak before the portfolio recovered. "
        "If your portfolio hit $14,000 then fell to $10,000, that's a -28.6% drawdown. "
        "Tells you the worst you would have felt holding this portfolio."
    ),
    "correlation_matrix": (
        "Shows how much each stock moves together. "
        "Values near +1 mean they move in sync (less diversification). "
        "Near 0 means they're independent — what you want for a balanced portfolio."
    ),
    "period": (
        "The date range of historical data used to calculate all metrics. "
        "Longer periods give more reliable numbers. Shorter periods may reflect "
        "unusual market conditions."
    ),
    "rolling_volatility": (
        "Volatility recalculated every day using only the past 30 days of data. "
        "Shows how risk changed over time — spiking during market crashes, "
        "calming during bull runs. Much more informative than a single average number."
    ),
    "rolling_sharpe": (
        "The Sharpe ratio recalculated on a rolling 30-day basis. "
        "Shows whether your risk-adjusted performance improved or deteriorated over time. "
        "Dips below zero mean you were better off in a Treasury bond during that period."
    ),
    "beta": (
        "How much your portfolio moves when the overall market (S&P 500) moves. "
        "Beta of 1.2 means if the market drops 10%, your portfolio tends to drop 12%. "
        "Below 1.0 means you're less sensitive than the market — a smoother ride."
    ),
    "alpha": (
        "The return you earned above and beyond what your Beta predicts you should have earned. "
        "Positive alpha means you genuinely outperformed the market on a risk-adjusted basis. "
        "This is what every fund manager is trying to achieve."
    ),
    "monte_carlo": (
        "Each line is one possible future for your portfolio over the next year. "
        "The shaded band shows where 90% of outcomes land. "
        "The higher the band, the more upside potential. The wider it is, the more uncertain the future."
    ),
    "efficient_frontier": (
        "Each dot is a possible portfolio — a different combination of your stocks. "
        "The curve along the top-left edge shows the best possible return for each level of risk. "
        "The star marks the single optimal portfolio with the highest risk-adjusted return."
    ),
}


# ─── Step 1: Prices → Returns ────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert closing prices into daily % returns. Drops the first NaN row."""
    return prices.pct_change().dropna()


# ─── Step 2: Portfolio return series ─────────────────────────────────────────

def compute_portfolio_returns(
    returns: pd.DataFrame,
    weights: list[float],
) -> pd.Series:
    """Weighted sum of individual stock returns → single portfolio return series."""
    weights = np.array(weights)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Weights must sum to 1.0, got {weights.sum():.4f}")
    return returns.dot(weights)


# ─── Step 3: Annualised return ────────────────────────────────────────────────

def compute_annualised_return(portfolio_returns: pd.Series) -> float:
    """CAGR — compounds daily returns and annualises. Accounts for compounding."""
    n_days     = len(portfolio_returns)
    cumulative = (1 + portfolio_returns).prod()
    return cumulative ** (TRADING_DAYS / n_days) - 1


# ─── Step 4: Volatility ───────────────────────────────────────────────────────

def compute_volatility_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised covariance matrix — foundation of portfolio variance."""
    return returns.cov() * TRADING_DAYS


def compute_volatility_from_cov(weights: list[float], cov_matrix: pd.DataFrame) -> float:
    """Portfolio volatility via σ = √(wᵀ Σ w). Accounts for asset correlations."""
    w = np.array(weights)
    return np.sqrt(w.T @ cov_matrix.values @ w)


def compute_volatility(portfolio_returns: pd.Series) -> float:
    """Simple annualised std — used internally by ratio functions."""
    return portfolio_returns.std() * np.sqrt(TRADING_DAYS)


# ─── Step 5: Sharpe ratio ─────────────────────────────────────────────────────

def compute_sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """Sharpe = (return − risk_free) / volatility."""
    ann_return = compute_annualised_return(portfolio_returns)
    ann_vol    = compute_volatility(portfolio_returns)
    if ann_vol == 0:
        return 0.0
    return (ann_return - risk_free_rate) / ann_vol


# ─── Step 6: Sortino ratio ────────────────────────────────────────────────────

def compute_sortino_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """Sortino = (return − risk_free) / downside_std. Only penalises bad days."""
    ann_return       = compute_annualised_return(portfolio_returns)
    negative_returns = portfolio_returns[portfolio_returns < 0]
    if len(negative_returns) == 0:
        return 0.0
    downside_std = negative_returns.std() * np.sqrt(TRADING_DAYS)
    return (ann_return - risk_free_rate) / downside_std


# ─── Step 7: Maximum drawdown ─────────────────────────────────────────────────

def compute_max_drawdown(portfolio_returns: pd.Series) -> dict:
    """Largest peak-to-trough decline. Returns value and full time series."""
    wealth_index    = (1 + portfolio_returns).cumprod()
    rolling_peak    = wealth_index.cummax()
    drawdown_series = (wealth_index - rolling_peak) / rolling_peak
    return {
        "max_drawdown":    drawdown_series.min(),
        "drawdown_series": drawdown_series,
    }


# ─── Step 8: CVaR / Expected Shortfall ───────────────────────────────────────

def compute_cvar(
    portfolio_returns: pd.Series,
    confidence: float = 0.95,
    portfolio_value: float = 10_000,
) -> dict:
    """VaR = tail threshold. CVaR = average loss beyond that threshold."""
    percentile    = (1 - confidence) * 100
    var_threshold = np.percentile(portfolio_returns, percentile)
    tail_returns  = portfolio_returns[portfolio_returns <= var_threshold]
    return {
        "var_pct":     var_threshold,
        "var_dollar":  var_threshold * portfolio_value,
        "cvar_pct":    tail_returns.mean(),
        "cvar_dollar": tail_returns.mean() * portfolio_value,
        "confidence":  confidence,
        "n_tail_days": len(tail_returns),
    }


# ─── Step 9: Correlation matrix ───────────────────────────────────────────────

def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation -1 to +1. Clears duplicate 'Ticker' axis labels."""
    corr = returns.corr()
    corr.index.name   = None
    corr.columns.name = None
    return corr


# ─── Step 10: Period info ─────────────────────────────────────────────────────

def compute_period(prices: pd.DataFrame) -> dict:
    """Actual date range from the price data — reflects what yfinance returned."""
    return {
        "start":   prices.index[0].strftime("%Y-%m-%d"),
        "end":     prices.index[-1].strftime("%Y-%m-%d"),
        "n_days":  len(prices),
        "n_years": round(len(prices) / TRADING_DAYS, 1),
    }


# ─── Step 11: Rolling metrics ─────────────────────────────────────────────────

def compute_rolling_metrics(
    portfolio_returns: pd.Series,
    window: int = 30,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """Rolling volatility and Sharpe on a sliding window."""
    rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(TRADING_DAYS)

    daily_rf       = risk_free_rate / TRADING_DAYS
    excess_returns = portfolio_returns - daily_rf
    rolling_sharpe = (
        excess_returns.rolling(window).mean()
        .div(excess_returns.rolling(window).std())
        * np.sqrt(TRADING_DAYS)
    )

    return {
        "rolling_volatility": rolling_vol,
        "rolling_sharpe":     rolling_sharpe,
        "window":             window,
    }


# ─── Step 12: Beta & Alpha ────────────────────────────────────────────────────

def compute_beta_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """
    Beta  = slope of regression line (portfolio vs benchmark returns).
    Alpha = annualised return above what Beta predicts via CAPM.
    """
    aligned = pd.concat(
        [portfolio_returns, benchmark_returns], axis=1, join="inner"
    ).dropna()

    port_r  = aligned.iloc[:, 0]
    bench_r = aligned.iloc[:, 1]

    beta, daily_alpha = np.polyfit(bench_r, port_r, deg=1)
    alpha             = daily_alpha * TRADING_DAYS

    bench_ann_return = compute_annualised_return(bench_r)
    capm_expected    = risk_free_rate + beta * (bench_ann_return - risk_free_rate)

    return {
        "beta":             round(beta, 4),
        "alpha":            round(alpha, 4),
        "capm_expected":    round(capm_expected, 4),
        "benchmark_return": round(bench_ann_return, 4),
        "n_days":           len(aligned),
    }


# ─── NEW Tier 3: Efficient Frontier ───────────────────────────────────────────

def compute_efficient_frontier(
    returns: pd.DataFrame,
    n_portfolios: int = 5000,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """
    Markowitz Efficient Frontier via Monte Carlo simulation.

    Generates n_portfolios random weight combinations and computes
    the return and volatility for each. The cloud of dots forms the
    opportunity set — the frontier is the top-left edge of that cloud.

    How random weights are generated:
      1. Draw n random numbers from a uniform distribution.
      2. Divide each by the sum so they add up to exactly 1.0.
      This is called a Dirichlet-like normalisation — it ensures weights
      are always positive and always sum to 1 without any loops.

    What we return:
      - vols, returns, sharpes : arrays of length n_portfolios (one per dot)
      - max_sharpe_weights     : the weights that maximise Sharpe ratio
      - max_sharpe_idx         : index into the arrays for the best portfolio
      - min_vol_idx            : index for the minimum volatility portfolio

    The Streamlit chart uses these arrays directly with Plotly scatter.

    Why 5,000?
      Enough to trace a smooth frontier curve visually.
      Runs in under a second with numpy vectorisation.
      You could go to 10,000 without noticeable slowdown.
    """
    n_assets = returns.shape[1]

    # Annualised covariance matrix — computed once, reused for all portfolios
    cov_matrix   = returns.cov() * TRADING_DAYS
    mean_returns = returns.mean() * TRADING_DAYS   # annualised mean return per asset

    # Pre-allocate result arrays — faster than appending in a loop
    all_returns = np.zeros(n_portfolios)
    all_vols    = np.zeros(n_portfolios)
    all_sharpes = np.zeros(n_portfolios)
    all_weights = np.zeros((n_portfolios, n_assets))

    for i in range(n_portfolios):
        # Step 1: generate random weights that sum to 1
        raw     = np.random.random(n_assets)
        weights = raw / raw.sum()

        # Step 2: portfolio annualised return = weighted sum of asset returns
        port_return = np.dot(weights, mean_returns)

        # Step 3: portfolio volatility via covariance matrix
        port_vol = np.sqrt(weights.T @ cov_matrix.values @ weights)

        # Step 4: Sharpe ratio
        port_sharpe = (port_return - risk_free_rate) / port_vol

        # Store results
        all_returns[i] = port_return
        all_vols[i]    = port_vol
        all_sharpes[i] = port_sharpe
        all_weights[i] = weights

    # Find the special portfolios
    max_sharpe_idx = np.argmax(all_sharpes)    # highest Sharpe — optimal portfolio
    min_vol_idx    = np.argmin(all_vols)       # lowest volatility — safest portfolio

    max_sharpe_weights = all_weights[max_sharpe_idx]
    min_vol_weights    = all_weights[min_vol_idx]

    # Build a readable weights dict for display
    tickers = returns.columns.tolist()

    def weights_to_dict(w):
        return {ticker: round(float(w[i]), 4) for i, ticker in enumerate(tickers)}

    return {
        # Arrays for the scatter plot (one entry per simulated portfolio)
        "vols":    all_vols,
        "returns": all_returns,
        "sharpes": all_sharpes,

        # Optimal portfolio (max Sharpe)
        "max_sharpe_idx":     max_sharpe_idx,
        "max_sharpe_return":  all_returns[max_sharpe_idx],
        "max_sharpe_vol":     all_vols[max_sharpe_idx],
        "max_sharpe_sharpe":  all_sharpes[max_sharpe_idx],
        "max_sharpe_weights": weights_to_dict(max_sharpe_weights),

        # Minimum volatility portfolio
        "min_vol_idx":        min_vol_idx,
        "min_vol_return":     all_returns[min_vol_idx],
        "min_vol_vol":        all_vols[min_vol_idx],
        "min_vol_weights":    weights_to_dict(min_vol_weights),

        "n_portfolios":       n_portfolios,
        "tickers":            tickers,
    }



# ─── Monte Carlo simulation ───────────────────────────────────────────────────

def compute_monte_carlo(
    portfolio_returns: pd.Series,
    portfolio_value: float = 10_000,
    n_simulations: int = 1_000,
    n_days: int = 252,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """
    Simulate n_simulations possible futures for the portfolio over n_days trading days.

    How it works:
      1. Compute the portfolio's historical daily mean return and daily std.
      2. For each simulation, draw n_days random returns from a normal distribution
         using those two parameters (this assumes returns are roughly normally distributed).
      3. Compound those returns into a final portfolio value.
      4. Repeat n_simulations times.

    The result is a distribution of possible outcomes — not a prediction,
    but an honest range of where the portfolio could end up.

    Key outputs:
      - all_paths        : (n_days+1) × n_simulations array — every daily value for every sim
      - final_values     : array of n_simulations end-of-year portfolio values
      - percentile_5     : 5th percentile path  (bad scenario)
      - percentile_50    : median path           (middle scenario)
      - percentile_95    : 95th percentile path  (good scenario)
      - prob_profit      : probability of ending above starting value
      - prob_loss_10pct  : probability of losing more than 10%
    """
    daily_mean = portfolio_returns.mean()
    daily_std  = portfolio_returns.std()

    # Matrix of random daily returns: shape (n_days, n_simulations)
    # Each column is one simulation's year of daily returns
    random_returns = np.random.normal(
        loc=daily_mean,
        scale=daily_std,
        size=(n_days, n_simulations),
    )

    # Build price paths: start every simulation at portfolio_value
    # Shape: (n_days+1, n_simulations) — row 0 is the starting value
    price_paths      = np.zeros((n_days + 1, n_simulations))
    price_paths[0]   = portfolio_value
    for day in range(1, n_days + 1):
        price_paths[day] = price_paths[day - 1] * (1 + random_returns[day - 1])

    final_values = price_paths[-1]   # end-of-year values for all simulations

    # Percentile paths — take the path whose final value is at each percentile
    p5_idx  = np.argsort(final_values)[int(0.05 * n_simulations)]
    p50_idx = np.argsort(final_values)[int(0.50 * n_simulations)]
    p95_idx = np.argsort(final_values)[int(0.95 * n_simulations)]

    return {
        "all_paths":       price_paths,
        "final_values":    final_values,
        "percentile_5":    price_paths[:, p5_idx],
        "percentile_50":   price_paths[:, p50_idx],
        "percentile_95":   price_paths[:, p95_idx],
        "p5_final":        final_values[p5_idx],
        "p50_final":       final_values[p50_idx],
        "p95_final":       final_values[p95_idx],
        "prob_profit":     float(np.mean(final_values > portfolio_value)),
        "prob_loss_10pct": float(np.mean(final_values < portfolio_value * 0.9)),
        "n_simulations":   n_simulations,
        "n_days":          n_days,
        "portfolio_value": portfolio_value,
        "daily_mean":      daily_mean,
        "daily_std":       daily_std,
    }


# ─── Master function ──────────────────────────────────────────────────────────

def compute_all_metrics(
    prices: pd.DataFrame,
    weights: list[float],
    portfolio_value: float = 10_000,
    benchmark_prices: pd.DataFrame | None = None,
    rolling_window: int = 30,
    n_frontier_portfolios: int = 5000,
    n_mc_simulations: int = 1_000,
) -> dict:
    """
    Full pipeline: prices → returns → all metrics.
    This is the single function the Streamlit app will call.
    """
    returns           = compute_returns(prices)
    portfolio_returns = compute_portfolio_returns(returns, weights)
    cov_matrix        = compute_volatility_matrix(returns)
    risk_result       = compute_cvar(portfolio_returns, portfolio_value=portfolio_value)
    drawdown_result   = compute_max_drawdown(portfolio_returns)
    rolling           = compute_rolling_metrics(portfolio_returns, window=rolling_window)
    frontier          = compute_efficient_frontier(returns, n_portfolios=n_frontier_portfolios)
    monte_carlo       = compute_monte_carlo(portfolio_returns, portfolio_value=portfolio_value, n_simulations=n_mc_simulations)

    result = {
        "period":                compute_period(prices),
        "annualised_return":     compute_annualised_return(portfolio_returns),
        "annualised_volatility": compute_volatility_from_cov(weights, cov_matrix),
        "sharpe_ratio":          compute_sharpe_ratio(portfolio_returns),
        "sortino_ratio":         compute_sortino_ratio(portfolio_returns),
        "var_cvar":              risk_result,
        "max_drawdown":          drawdown_result,
        "rolling":               rolling,
        "efficient_frontier":    frontier,
        "monte_carlo":           monte_carlo,
        "correlation_matrix":    compute_correlation_matrix(returns),
        "cov_matrix":            cov_matrix,
        "returns":               returns,
        "portfolio_returns":     portfolio_returns,
        "drawdown_series":       drawdown_result["drawdown_series"],
        "tooltips":              METRIC_TOOLTIPS,
    }

    if benchmark_prices is not None:
        benchmark_returns    = compute_returns(benchmark_prices).iloc[:, 0]
        result["beta_alpha"] = compute_beta_alpha(portfolio_returns, benchmark_returns)
    else:
        result["beta_alpha"] = None

    return result


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_fetcher import fetch_with_benchmark

    print("Fetching prices + benchmark...")
    portfolio_prices, benchmark_prices = fetch_with_benchmark(
        tickers=["AAPL", "MSFT", "GOOGL"],
        start_date="2023-01-01",
    )

    weights = [1/3, 1/3, 1/3]

    print("Computing metrics...\n")
    m = compute_all_metrics(
        prices=portfolio_prices,
        weights=weights,
        portfolio_value=10_000,
        benchmark_prices=benchmark_prices,
    )

    t  = m["tooltips"]
    p  = m["period"]
    f  = m["efficient_frontier"]
    ba = m["beta_alpha"]
    r  = m["rolling"]
    vc = m["var_cvar"]

    print("─" * 55)
    print("  PERIOD")
    print("─" * 55)
    print(f"\n  {p['start']} → {p['end']}")
    print(f"  {p['n_days']} trading days  ({p['n_years']} years)")

    print()
    print("─" * 55)
    print("  PERFORMANCE")
    print("─" * 55)
    print(f"\n  Annualised return:      {m['annualised_return']:.1%}")
    print(f"  Annualised volatility:  {m['annualised_volatility']:.1%}")

    print()
    print("─" * 55)
    print("  RISK-ADJUSTED RATIOS")
    print("─" * 55)
    print(f"\n  Sharpe ratio:   {m['sharpe_ratio']:.2f}")
    print(f"  Sortino ratio:  {m['sortino_ratio']:.2f}")

    print()
    print("─" * 55)
    print("  DOWNSIDE RISK  (on a $10,000 portfolio)")
    print("─" * 55)
    print(f"\n  VaR  95%:  {vc['var_pct']:.2%} per day  →  -${abs(vc['var_dollar']):,.0f}")
    print(f"  CVaR 95%:  {vc['cvar_pct']:.2%} per day  →  -${abs(vc['cvar_dollar']):,.0f}")
    print(f"  (averaging the worst {vc['n_tail_days']} days)")

    print()
    print("─" * 55)
    print("  DRAWDOWN")
    print("─" * 55)
    print(f"\n  Max drawdown:  {m['max_drawdown']['max_drawdown']:.1%}")

    print()
    print("─" * 55)
    print("  ROLLING METRICS  (30-day window)")
    print("─" * 55)
    print(f"\n  Rolling volatility:  {r['rolling_volatility'].dropna().iloc[-1]:.1%}")
    print(f"  Rolling Sharpe:      {r['rolling_sharpe'].dropna().iloc[-1]:.2f}")

    print()
    print("─" * 55)
    print("  BETA & ALPHA  (vs S&P 500)")
    print("─" * 55)
    print(f"\n  Beta:   {ba['beta']:.2f}")
    print(f"  Alpha:  {ba['alpha']:.2%}  (annualised)")
    print(f"\n  S&P 500 return:       {ba['benchmark_return']:.1%}")
    print(f"  CAPM expected:        {ba['capm_expected']:.1%}")
    print(f"  Your actual return:   {m['annualised_return']:.1%}")
    print(f"  → Alpha gap:          {m['annualised_return'] - ba['capm_expected']:.1%}")

    print()
    print("─" * 55)
    print("  EFFICIENT FRONTIER  (5,000 simulated portfolios)")
    print("─" * 55)
    print(f"\n  Simulated {f['n_portfolios']:,} random weight combinations")

    print(f"\n  ★  Max Sharpe portfolio (Sharpe: {f['max_sharpe_sharpe']:.2f})")
    print(f"     Return: {f['max_sharpe_return']:.1%}  |  Volatility: {f['max_sharpe_vol']:.1%}")
    print(f"     Weights: ", end="")
    for ticker, w in f["max_sharpe_weights"].items():
        print(f"{ticker} {w:.0%}", end="  ")

    print(f"\n\n  ◆  Min Volatility portfolio (Vol: {f['min_vol_vol']:.1%})")
    print(f"     Return: {f['min_vol_return']:.1%}  |  Sharpe: {f['min_vol_vol']:.2f}")
    print(f"     Weights: ", end="")
    for ticker, w in f["min_vol_weights"].items():
        print(f"{ticker} {w:.0%}", end="  ")

    print(f"\n\n  Your current portfolio (equal weight):")
    print(f"     Return: {m['annualised_return']:.1%}  |  Volatility: {m['annualised_volatility']:.1%}  |  Sharpe: {m['sharpe_ratio']:.2f}")

    print()
    print("─" * 55)
    print("  CORRELATION MATRIX")
    print("─" * 55)
    print()
    print(m["correlation_matrix"].round(2))
    print()
