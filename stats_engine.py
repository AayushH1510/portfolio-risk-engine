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
  Portfolio  — diversification score
  Stress     — historical crash scenario testing            ← Tier 4
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
    "diversification_score": (
        "How spread out your risk is across your assets. "
        "Derived from the average pairwise correlation between your stocks. "
        "100 = perfectly uncorrelated (ideal). 0 = all stocks move in lockstep (concentrated risk)."
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
    """Actual date range from the price data — reflects what Twelve Data returned."""
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


def compute_treynor_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """Excess return per unit of systematic (market) risk — annualised return over beta."""
    beta = compute_beta_alpha(portfolio_returns, benchmark_returns, risk_free_rate)["beta"]
    if beta == 0:
        return 0.0

    annualised_return = compute_annualised_return(portfolio_returns)
    return (annualised_return - risk_free_rate) / beta


def compute_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Active return per unit of tracking risk vs the benchmark."""
    aligned = pd.concat(
        [portfolio_returns, benchmark_returns], axis=1, join="inner"
    ).dropna()
    active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]

    std = active_returns.std()
    if std == 0:
        return 0.0

    return (active_returns.mean() / std) * np.sqrt(TRADING_DAYS)


# ─── Step 13: Diversification score ──────────────────────────────────────────

def compute_diversification_score(corr_matrix: pd.DataFrame) -> dict:
    """
    Portfolio diversification score on a 0–100 scale.

    HOW IT WORKS:
    1. Extract the upper-triangle off-diagonal elements of the correlation matrix
       (these are the pairwise correlations between each pair of assets).
    2. Compute their average — this is the mean pairwise correlation.
    3. Map to a score: score = (1 − avg_pairwise_corr) × 100
       - avg_corr = 1.0  → score =   0  (all assets move in lockstep)
       - avg_corr = 0.0  → score = 100  (uncorrelated — ideal diversification)
       - avg_corr = −1.0 → score = 100+ (clamped to 100)
    4. Clamp to [0, 100].

    For a single asset there are no pairs, so we return 100 (N/A).

    LABELS:
      ≥ 70  — Well diversified
      40–69 — Moderate
      < 40  — Concentrated
    """
    n = len(corr_matrix)

    if n <= 1:
        return {
            "score":            100,
            "avg_pairwise_corr": 0.0,
            "label":            "Single asset",
        }

    # Upper triangle (k=1 skips the diagonal)
    mask      = np.triu(np.ones((n, n), dtype=bool), k=1)
    off_diag  = corr_matrix.values[mask]
    avg_corr  = float(off_diag.mean())
    score     = int(round(max(0, min(100, (1 - avg_corr) * 100))))

    if score >= 70:
        label = "Well diversified"
    elif score >= 40:
        label = "Moderate"
    else:
        label = "Concentrated"

    return {
        "score":             score,
        "avg_pairwise_corr": round(avg_corr, 4),
        "label":             label,
    }


# ─── Tier 3: Efficient Frontier ───────────────────────────────────────────────

def compute_efficient_frontier(
    returns: pd.DataFrame,
    n_portfolios: int = 5000,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """
    Markowitz Efficient Frontier via Monte Carlo simulation.
    Uses CAGR (geometric/compound annualisation). Fully vectorised.
    """
    n_assets  = returns.shape[1]
    n_obs     = len(returns)
    tickers   = returns.columns.tolist()

    cov_matrix  = returns.cov() * TRADING_DAYS
    raw_weights = np.random.random((n_portfolios, n_assets))
    all_weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)

    log_rets      = np.log1p(returns.values)
    port_log_rets = log_rets @ all_weights.T
    log_sums      = port_log_rets.sum(axis=0)
    all_returns   = np.exp(log_sums * (TRADING_DAYS / n_obs)) - 1

    temp        = all_weights @ cov_matrix.values
    all_vols    = np.sqrt((temp * all_weights).sum(axis=1))
    all_sharpes = (all_returns - risk_free_rate) / np.where(all_vols > 0, all_vols, np.inf)

    max_sharpe_idx = int(np.argmax(all_sharpes))
    min_vol_idx    = int(np.argmin(all_vols))

    def weights_to_dict(w):
        return {ticker: round(float(w[i]), 4) for i, ticker in enumerate(tickers)}

    return {
        "vols":    all_vols,
        "returns": all_returns,
        "sharpes": all_sharpes,

        "max_sharpe_idx":     max_sharpe_idx,
        "max_sharpe_return":  float(all_returns[max_sharpe_idx]),
        "max_sharpe_vol":     float(all_vols[max_sharpe_idx]),
        "max_sharpe_sharpe":  float(all_sharpes[max_sharpe_idx]),
        "max_sharpe_weights": weights_to_dict(all_weights[max_sharpe_idx]),

        "min_vol_idx":        min_vol_idx,
        "min_vol_return":     float(all_returns[min_vol_idx]),
        "min_vol_vol":        float(all_vols[min_vol_idx]),
        "min_vol_weights":    weights_to_dict(all_weights[min_vol_idx]),

        "n_portfolios": n_portfolios,
        "tickers":      tickers,
    }


# ─── Monte Carlo simulation — with Cholesky decomposition ────────────────────

# Per-day drift adjustment applied to simulated returns before the Cholesky
# step — bear/bull assume a ±10%/yr headwind/tailwind on top of historical
# drift; base leaves historical drift untouched.
MC_SCENARIO_DRIFT = {
    "bear": -0.10 / 252,
    "base": 0.0,
    "bull":  0.10 / 252,
}


def compute_monte_carlo(
    portfolio_returns: pd.Series,
    asset_returns: pd.DataFrame,
    weights: list[float],
    portfolio_value: float = 10_000,
    n_simulations: int = 1_000,
    n_days: int = 252,
    scenario: str = "base",
) -> dict:
    """
    Simulate correlated asset price paths using Cholesky decomposition.
    Falls back to univariate normal if Cholesky fails.

    `scenario` shifts the daily drift used in the simulation — "bear"
    (-10%/yr headwind), "base" (historical drift, unchanged, default), or
    "bull" (+10%/yr tailwind). The drift shift is applied on top of each
    asset's own historical mean daily return, not in place of it.
    """
    if scenario not in MC_SCENARIO_DRIFT:
        raise ValueError(f"Unknown scenario {scenario!r}. Expected one of {list(MC_SCENARIO_DRIFT)}.")
    drift = MC_SCENARIO_DRIFT[scenario]

    n_assets       = asset_returns.shape[1]
    w              = np.array(weights)
    daily_mean_vec = asset_returns.mean().values + drift
    cov_daily      = asset_returns.cov().values

    try:
        L = np.linalg.cholesky(cov_daily)
        price_paths    = np.zeros((n_days + 1, n_simulations))
        price_paths[0] = portfolio_value
        for day in range(1, n_days + 1):
            Z              = np.random.standard_normal((n_simulations, n_assets))
            day_asset_rets = daily_mean_vec + Z @ L.T
            day_port_rets  = day_asset_rets @ w
            price_paths[day] = price_paths[day - 1] * (1 + day_port_rets)
        method = "cholesky"

    except np.linalg.LinAlgError:
        daily_mean     = portfolio_returns.mean() + drift
        daily_std      = portfolio_returns.std()
        random_returns = np.random.normal(daily_mean, daily_std, (n_days, n_simulations))
        price_paths    = np.zeros((n_days + 1, n_simulations))
        price_paths[0] = portfolio_value
        for day in range(1, n_days + 1):
            price_paths[day] = price_paths[day - 1] * (1 + random_returns[day - 1])
        method = "fallback_univariate"

    final_values = price_paths[-1]
    p5_idx  = np.argsort(final_values)[int(0.05 * n_simulations)]
    p50_idx = np.argsort(final_values)[int(0.50 * n_simulations)]
    p95_idx = np.argsort(final_values)[int(0.95 * n_simulations)]

    return {
        "all_paths":       price_paths,
        "final_values":    final_values,
        "percentile_5":    price_paths[:, p5_idx],
        "percentile_50":   price_paths[:, p50_idx],
        "percentile_95":   price_paths[:, p95_idx],
        "p5_final":        float(final_values[p5_idx]),
        "p50_final":       float(final_values[p50_idx]),
        "p95_final":       float(final_values[p95_idx]),
        "prob_profit":     float(np.mean(final_values > portfolio_value)),
        "prob_loss_10pct": float(np.mean(final_values < portfolio_value * 0.9)),
        "n_simulations":   n_simulations,
        "n_days":          n_days,
        "portfolio_value": portfolio_value,
        "daily_mean":      float(portfolio_returns.mean() + drift),
        "daily_std":       float(portfolio_returns.std()),
        "method":          method,
        "scenario":        scenario,
    }


# ─── Tier 4: Stress testing ────────────────────────────────────────────────────

def compute_stress_scenario(
    prices: pd.DataFrame,
    crash_start: str,
    crash_end: str,
    tickers: list[str],
    weights: list[float],
) -> dict | None:
    """
    Applies portfolio weights to one historical crash window.

    `prices` must span from crash_start through a recovery-search period well
    past crash_end — the crash-window stats (return, worst day) only look at
    [crash_start, crash_end], but recovery_days searches forward through the
    rest of `prices` for when the portfolio gets back to its pre-crash peak.

    Tickers with no data during the crash window are dropped and the
    remaining weights are reweighted proportionally to sum to 1 (their
    weight isn't redistributed to cash — it just isn't tracked in this
    scenario). Returns None if none of the requested tickers have data for
    the window at all.
    """
    crash_prices = prices.loc[crash_start:crash_end]

    valid = [t for t in tickers if t in crash_prices.columns and crash_prices[t].notna().sum() >= 2]
    if not valid:
        return None

    valid_weights_raw = [weights[tickers.index(t)] for t in valid]
    total_w = sum(valid_weights_raw)
    valid_weights = np.array([w / total_w for w in valid_weights_raw])

    # Crash-window stats
    crash_returns       = crash_prices[valid].dropna().pct_change().dropna()
    port_crash_returns  = crash_returns.dot(valid_weights)
    portfolio_return     = float((1 + port_crash_returns).prod() - 1)
    worst_day             = float(port_crash_returns.min())

    # Recovery — same tickers/weights, full extended series
    full_returns      = prices[valid].dropna().pct_change().dropna()
    port_full_returns = full_returns.dot(valid_weights)
    wealth_index      = (1 + port_full_returns).cumprod()

    crash_wealth = wealth_index.loc[crash_start:crash_end]
    trough_date  = crash_wealth.idxmin()
    peak_value   = wealth_index.loc[:trough_date].max()

    recovery_days = None
    after_trough  = wealth_index.loc[trough_date:]
    recovered     = after_trough[after_trough >= peak_value]
    if len(recovered) > 0:
        recovery_days = int(
            wealth_index.index.get_loc(recovered.index[0])
            - wealth_index.index.get_loc(trough_date)
        )

    return {
        "portfolio_return": portfolio_return,
        "worst_day":        worst_day,
        "recovery_days":    recovery_days,
        "excluded_tickers": [t for t in tickers if t not in valid],
    }


# ─── Tier 5: Backtesting ───────────────────────────────────────────────────────

def compute_backtest(
    returns: pd.DataFrame,
    weights: list[float],
    benchmark_returns: pd.Series,
) -> dict:
    """
    Runs three strategies through the same historical period so they're
    directly comparable: the user's portfolio (fixed weights from day 1,
    compounded daily), an equal-weight split across the same tickers, and
    the S&P 500 benchmark. All three are aligned to their common trading
    days first — a strategy can't be compared to a day it has no return for.
    """
    n_assets      = returns.shape[1]
    equal_weights = np.array([1.0 / n_assets] * n_assets)

    your_returns  = compute_portfolio_returns(returns, weights)
    equal_returns = returns.dot(equal_weights)

    aligned = pd.concat(
        [your_returns, equal_returns, benchmark_returns], axis=1, join="inner"
    ).dropna()
    aligned.columns = ["your_portfolio", "equal_weight", "sp500"]

    def _strategy_stats(rets: pd.Series) -> dict:
        cum_returns = (1 + rets).cumprod() - 1
        drawdown    = compute_max_drawdown(rets)

        annual_returns = {}
        for year in sorted(set(rets.index.year)):
            year_rets = rets[rets.index.year == year]
            annual_returns[str(year)] = float((1 + year_rets).prod() - 1)

        return {
            "cumulative_returns": {
                "dates":  cum_returns.index.strftime("%Y-%m-%d").tolist(),
                "values": cum_returns.values.tolist(),
            },
            "annualised_return":     float(compute_annualised_return(rets)),
            "annualised_volatility": float(compute_volatility(rets)),
            "sharpe_ratio":          float(compute_sharpe_ratio(rets)),
            "max_drawdown":          float(drawdown["max_drawdown"]),
            "annual_returns":        annual_returns,
        }

    return {
        "your_portfolio": _strategy_stats(aligned["your_portfolio"]),
        "equal_weight":   _strategy_stats(aligned["equal_weight"]),
        "sp500":          _strategy_stats(aligned["sp500"]),
        "period": {
            "start":   aligned.index[0].strftime("%Y-%m-%d"),
            "end":     aligned.index[-1].strftime("%Y-%m-%d"),
            "n_days":  len(aligned),
            "n_years": round(len(aligned) / TRADING_DAYS, 1),
        },
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
    include_heavy: bool = True,
) -> dict:
    """
    Full pipeline: prices → returns → all metrics.

    include_heavy=False skips the three computations expensive enough to be
    worth deferring behind a fast "summary" response — efficient frontier
    (n_frontier_portfolios simulated portfolios), Monte Carlo (a 252-day
    Python loop per call), and backtesting (cheap on its own, but pointless
    to compute without the other two since /api/analyse-summary's callers
    don't render it) — leaving "efficient_frontier", "monte_carlo", and
    "backtest" as None. Every other key is unaffected: the fast metrics are
    identical whichever way this is called, so a caller merging a later
    include_heavy=True response on top never sees any of them change.
    """
    returns           = compute_returns(prices)
    portfolio_returns = compute_portfolio_returns(returns, weights)
    per_ticker_cumulative_returns = {
        ticker: (1 + returns[ticker]).cumprod() - 1
        for ticker in returns.columns
    }
    cov_matrix        = compute_volatility_matrix(returns)
    corr_matrix       = compute_correlation_matrix(returns)
    drawdown_result   = compute_max_drawdown(portfolio_returns)
    rolling           = compute_rolling_metrics(portfolio_returns, window=rolling_window)
    frontier          = compute_efficient_frontier(returns, n_portfolios=n_frontier_portfolios) if include_heavy else None
    monte_carlo       = compute_monte_carlo(
        portfolio_returns=portfolio_returns,
        asset_returns=returns,
        weights=weights,
        portfolio_value=portfolio_value,
        n_simulations=n_mc_simulations,
    ) if include_heavy else None

    risk_95 = compute_cvar(portfolio_returns, confidence=0.95, portfolio_value=portfolio_value)
    risk_99 = compute_cvar(portfolio_returns, confidence=0.99, portfolio_value=portfolio_value)
    div_score = compute_diversification_score(corr_matrix)

    result = {
        "period":                compute_period(prices),
        "annualised_return":     compute_annualised_return(portfolio_returns),
        "annualised_volatility": compute_volatility_from_cov(weights, cov_matrix),
        "sharpe_ratio":          compute_sharpe_ratio(portfolio_returns),
        "sortino_ratio":         compute_sortino_ratio(portfolio_returns),
        "var_cvar":              risk_95,
        "var_cvar_99":           risk_99,
        "max_drawdown":          drawdown_result,
        "rolling":               rolling,
        "efficient_frontier":    frontier,
        "monte_carlo":           monte_carlo,
        "correlation_matrix":    corr_matrix,
        "cov_matrix":            cov_matrix,
        "returns":               returns,
        "portfolio_returns":     portfolio_returns,
        "per_ticker_cumulative_returns": per_ticker_cumulative_returns,
        "drawdown_series":       drawdown_result["drawdown_series"],
        "diversification_score": div_score,
        "tooltips":              METRIC_TOOLTIPS,
    }

    if benchmark_prices is not None:
        benchmark_returns           = compute_returns(benchmark_prices).iloc[:, 0]
        result["beta_alpha"]        = compute_beta_alpha(portfolio_returns, benchmark_returns)
        result["treynor_ratio"]     = compute_treynor_ratio(portfolio_returns, benchmark_returns)
        result["information_ratio"] = compute_information_ratio(portfolio_returns, benchmark_returns)
        result["backtest"]          = compute_backtest(returns, weights, benchmark_returns) if include_heavy else None
    else:
        result["beta_alpha"]        = None
        result["treynor_ratio"]     = None
        result["information_ratio"] = None
        result["backtest"]          = None

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
    m  = compute_all_metrics(prices=portfolio_prices, weights=weights, portfolio_value=10_000, benchmark_prices=benchmark_prices)
    vc = m["var_cvar"]
    v9 = m["var_cvar_99"]
    ds = m["diversification_score"]

    print(f"\n  VaR  95%:  {vc['var_pct']:.2%}  ->  -${abs(vc['var_dollar']):,.0f}")
    print(f"  CVaR 95%:  {vc['cvar_pct']:.2%}  ->  -${abs(vc['cvar_dollar']):,.0f}")
    print(f"  VaR  99%:  {v9['var_pct']:.2%}  ->  -${abs(v9['var_dollar']):,.0f}")
    print(f"  CVaR 99%:  {v9['cvar_pct']:.2%}  ->  -${abs(v9['cvar_dollar']):,.0f}")
    print(f"\n  Diversification score: {ds['score']}/100 - {ds['label']}")
    print(f"  Avg pairwise correlation: {ds['avg_pairwise_corr']:.4f}")