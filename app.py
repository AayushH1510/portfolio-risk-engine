"""
app.py
------
Portfolio Risk Analysis Engine — beginner-friendly UI.
Run with: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from data_fetcher import fetch_with_benchmark, validate_tickers
from stats_engine import compute_all_metrics

#  Page config

st.set_page_config(
    page_title="Portfolio Risk Engine",
    page_icon=None,
    layout="wide",
)

st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    letter-spacing: -0.01em;
  }

  .block-container {
    padding-top: 1.75rem;
    max-width: 1200px;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background-color: #0a1020 !important;
    border-right: 1px solid #1e2d40;
  }
  section[data-testid="stSidebar"] .stMarkdown p,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stCaption {
    color: #7a8fa6 !important;
    font-size: 12px !important;
  }
  section[data-testid="stSidebar"] h2 {
    color: #c9d1d9 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
  }

  /* ── Headings ── */
  h1 {
    font-size: 22px !important;
    font-weight: 600 !important;
    color: #e6edf3 !important;
    letter-spacing: -0.03em !important;
    border-bottom: 1px solid #1e2d40;
    padding-bottom: 0.6rem;
    margin-bottom: 0.5rem !important;
  }
  h2 {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #c9d1d9 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    margin-top: 0.5rem !important;
  }
  h3 {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #8b949e !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
  }
  h4 {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #c9d1d9 !important;
  }

  /* ── Metric cards ── */
  [data-testid="stMetric"] {
    background: #0f1724;
    border: 1px solid #1e2d40;
    border-radius: 6px;
    padding: 14px 16px !important;
  }
  [data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    color: #7a8fa6 !important;
  }
  [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 22px !important;
    font-weight: 500 !important;
    color: #e6edf3 !important;
  }
  [data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
  }
  /* Green positive deltas */
  [data-testid="stMetricDelta"][data-direction="up"] {
    color: #00d4aa !important;
  }
  [data-testid="stMetricDelta"][data-direction="down"] {
    color: #f85149 !important;
  }

  /* ── Insight boxes ── */
  .insight-box {
    background: #0f1724;
    border-left: 3px solid #1e2d40;
    border-radius: 0 4px 4px 0;
    padding: 12px 16px;
    margin: 8px 0 20px 0;
    font-size: 13px;
    line-height: 1.65;
    color: #8b949e;
    font-family: 'Inter', sans-serif;
  }
  .insight-box .label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #3a4a5c;
    margin-bottom: 5px;
  }
  .insight-box strong { color: #c9d1d9; font-weight: 500; }
  .insight-box.good  { border-color: #00d4aa; }
  .insight-box.good  .label { color: #00d4aa; }
  .insight-box.warn  { border-color: #d29922; }
  .insight-box.warn  .label { color: #d29922; }
  .insight-box.bad   { border-color: #f85149; }
  .insight-box.bad   .label { color: #f85149; }

  /* ── Learn cards ── */
  .learn-card {
    background: #0f1724;
    border: 1px solid #1e2d40;
    border-radius: 6px;
    padding: 16px 18px;
    margin-bottom: 10px;
  }
  .learn-card h4 { margin: 0 0 6px; font-size: 13px; color: #c9d1d9; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
  .learn-card p  { margin: 0; font-size: 13px; color: #7a8fa6; line-height: 1.65; }
  .learn-card .example {
    margin-top: 10px;
    font-size: 12px;
    font-family: 'IBM Plex Mono', monospace;
    background: #080d14;
    border-radius: 4px;
    border-left: 2px solid #00d4aa;
    padding: 8px 12px;
    color: #00d4aa;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #1e2d40;
    background: transparent;
  }
  .stTabs [data-baseweb="tab"] {
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #4a5a6a !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent;
  }
  .stTabs [aria-selected="true"] {
    color: #00d4aa !important;
    border-bottom: 2px solid #00d4aa !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: transparent !important;
    border: 1px solid #00d4aa !important;
    color: #00d4aa !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    padding: 10px 24px !important;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: #00d4aa !important;
    color: #080d14 !important;
  }

  /* ── Dividers ── */
  hr { border-color: #1e2d40 !important; margin: 1.5rem 0 !important; }

  /* ── Alerts ── */
  .stAlert {
    background: #0f1724 !important;
    border: 1px solid #1e2d40 !important;
    border-radius: 4px !important;
    font-size: 13px !important;
    color: #8b949e !important;
  }

  /* ── Expanders ── */
  .streamlit-expanderHeader {
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #4a5a6a !important;
    background: #0a1020 !important;
    border: 1px solid #1e2d40 !important;
    border-radius: 4px !important;
  }
  .streamlit-expanderContent {
    background: #0f1724 !important;
    border: 1px solid #1e2d40 !important;
    border-top: none !important;
  }

  /* ── Sliders ── */
  .stSlider [data-baseweb="slider"] [role="slider"] {
    background: #00d4aa !important;
    border-color: #00d4aa !important;
  }
  .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] {
    color: #00d4aa !important;
  }

  /* ── Captions ── */
  .stCaption, .stMarkdown p { color: #4a5a6a !important; font-size: 12px !important; }

  /* ── Success / warning / error ── */
  .stSuccess { background: #0d2016 !important; border: 1px solid #00d4aa !important; color: #00d4aa !important; border-radius: 4px !important; }
  .stWarning { background: #1a1400 !important; border: 1px solid #d29922 !important; border-radius: 4px !important; }
  .stError   { background: #1a0a0a !important; border: 1px solid #f85149 !important; border-radius: 4px !important; }

  /* ── Dataframe ── */
  .stDataFrame { border: 1px solid #1e2d40 !important; border-radius: 4px !important; }

  /* ── Radio ── */
  .stRadio label { font-size: 12px !important; color: #7a8fa6 !important; }
</style>
""",
    unsafe_allow_html=True,
)


#  Helpers


def insight(label, text, tone="neutral"):
    cls = {"good": "good", "warn": "warn", "bad": "bad"}.get(tone, "")
    st.markdown(
        f'<div class="insight-box {cls}">'
        f'<div class="label">What this means for you</div>'
        f"{text}"
        f"</div>",
        unsafe_allow_html=True,
    )


def learn_card(title, body, example=None):
    ex = f'<div class="example">Example: {example}</div>' if example else ""
    st.markdown(
        f'<div class="learn-card"><h4>{title}</h4><p>{body}</p>{ex}</div>',
        unsafe_allow_html=True,
    )


#  Dynamic analysis functions


def analyse_return(ann_return, portfolio_value, benchmark_return=None):
    money = portfolio_value * (1 + ann_return)
    currency = "$"
    text = (
        f"Your portfolio grew at <strong>{ann_return:.1%} per year</strong> on average. "
        f"In simple terms: if you had put {currency}{portfolio_value:,.0f} in at the start, "
        f"you'd have roughly <strong>{currency}{money:,.0f}</strong> after one year. "
    )
    if benchmark_return is not None:
        diff = ann_return - benchmark_return
        if diff > 0:
            text += (
                f"That's <strong>{diff:.1%} better</strong> than just buying the whole "
                f"S&P 500 index — your specific stock picks genuinely outperformed the market."
            )
        else:
            text += (
                f"That's <strong>{abs(diff):.1%} less</strong> than the S&P 500 returned "
                f"over the same period. The market outperformed your specific picks."
            )
    tone = "good" if ann_return > 0.15 else "warn" if ann_return > 0 else "bad"
    return text, tone


def analyse_volatility(vol):
    daily_swing = vol / np.sqrt(252)
    text = (
        f"Your portfolio moves by about <strong>{vol:.1%} per year</strong>. "
        f"Day to day, that's roughly <strong>{daily_swing:.1%}</strong> on a typical day. "
    )
    if vol < 0.15:
        text += (
            "That's quite calm — similar to a steady, diversified fund. Smooth ride."
        )
        tone = "good"
    elif vol < 0.25:
        text += "That's moderate — typical for a stock portfolio. Some ups and downs, but manageable."
        tone = "warn"
    elif vol < 0.40:
        text += (
            "That's quite bumpy. Your balance will jump around noticeably. "
            "Make sure you're comfortable with that before investing real money."
        )
        tone = "warn"
    else:
        text += (
            "That's very high volatility — this portfolio swings dramatically. "
            "Big potential gains, but also big potential losses. High risk."
        )
        tone = "bad"
    return text, tone


def analyse_sharpe(sharpe):
    if sharpe > 2:
        verdict = "excellent — genuinely rare and impressive"
        tone = "good"
    elif sharpe > 1:
        verdict = "good — you're being well rewarded for the risk you're taking"
        tone = "good"
    elif sharpe > 0:
        verdict = "okay — you're making money, but not a lot relative to the risk"
        tone = "warn"
    else:
        verdict = "poor — you'd have been better off leaving money in a savings account"
        tone = "bad"
    text = (
        f"Your Sharpe ratio is <strong>{sharpe:.2f}</strong>, which is {verdict}. "
        f"Think of it as a value-for-money score for risk. "
        f"Above 1.0 means every unit of risk you took was rewarded with solid returns."
    )
    return text, tone


def analyse_sortino(sortino, sharpe):
    diff = sortino - sharpe
    text = (
        f"Your Sortino ratio is <strong>{sortino:.2f}</strong> vs your Sharpe of {sharpe:.2f}. "
        f"The gap of {diff:.2f} tells you that a lot of your portfolio's movement was "
    )
    if diff > 0.5:
        text += (
            "<strong>upward swings</strong> — your bad days weren't that bad relative to "
            "your good days. That's a healthy sign for this portfolio."
        )
        tone = "good"
    else:
        text += (
            "split fairly evenly between good days and bad days — "
            "the portfolio swings in both directions similarly."
        )
        tone = "warn"
    return text, tone


def analyse_drawdown(drawdown, portfolio_value):
    loss = abs(drawdown) * portfolio_value
    text = (
        f"At its worst point, your portfolio dropped <strong>{abs(drawdown):.1%}</strong> "
        f"from its peak — that's <strong>${loss:,.0f}</strong> on a ${portfolio_value:,.0f} portfolio. "
    )
    if abs(drawdown) < 0.15:
        text += "That's a mild dip. Most stock portfolios see bigger drops than this."
        tone = "good"
    elif abs(drawdown) < 0.30:
        text += (
            "That's significant. Ask yourself: would you have stayed calm and held on, "
            "or would you have panicked and sold? Most people sell at the bottom — which "
            "locks in the loss permanently."
        )
        tone = "warn"
    else:
        text += (
            "That's a major drop. Seeing your balance fall this far is extremely stressful. "
            "Many investors would have sold in a panic. Only invest what you can leave alone "
            "through drops this large."
        )
        tone = "bad"
    return text, tone


def analyse_var(var_pct, cvar_pct, portfolio_value):
    var_dollar = abs(var_pct * portfolio_value)
    cvar_dollar = abs(cvar_pct * portfolio_value)
    gap = cvar_dollar - var_dollar
    text = (
        f"On your worst 5% of days historically, you could lose up to "
        f"<strong>${var_dollar:,.0f}</strong> in a single day on a ${portfolio_value:,.0f} portfolio. "
        f"But on the very worst of those bad days, the average loss jumps to "
        f"<strong>${cvar_dollar:,.0f}</strong>. "
        f"The ${gap:,.0f} gap between them shows how much worse the extreme bad days get "
        f"compared to a 'normal' bad day."
    )
    tone = "good" if abs(var_pct) < 0.02 else "warn" if abs(var_pct) < 0.035 else "bad"
    return text, tone


def analyse_beta_alpha(beta, alpha, benchmark_return):
    if beta > 1.5:
        beta_desc = (
            f"a <strong>very high Beta of {beta:.2f}</strong>. This means when the market "
            f"drops 10%, your portfolio tends to drop around {beta*10:.0f}%. "
            f"You're taking on significantly more market risk than average. "
            f"High potential returns, but much sharper falls during downturns."
        )
        tone = "warn"
    elif beta > 1.1:
        beta_desc = (
            f"a <strong>Beta of {beta:.2f}</strong>, which is slightly above the market. "
            f"When the market drops 10%, you'd typically drop around {beta*10:.0f}%. "
            f"A bit more sensitive than average, common for growth-oriented portfolios."
        )
        tone = "warn"
    elif beta > 0.8:
        beta_desc = (
            f"a <strong>Beta of {beta:.2f}</strong> — roughly in line with the market. "
            f"Your portfolio moves with the overall market, not much more, not much less."
        )
        tone = "good"
    else:
        beta_desc = (
            f"a <strong>low Beta of {beta:.2f}</strong> — more defensive than the market. "
            f"When the market drops 10%, you'd typically only drop {beta*10:.0f}%. "
            f"Lower risk, but also potentially lower returns in a bull market."
        )
        tone = "good"

    alpha_desc = f" Your Alpha of <strong>{alpha:.2%}</strong> means "
    if alpha > 0.05:
        alpha_desc += (
            "your stock picks genuinely added value above what the market predicted. "
            "That's real outperformance — not just luck from market conditions."
        )
    elif alpha > 0:
        alpha_desc += "you slightly beat expectations. A positive result."
    else:
        alpha_desc += (
            "the market's overall rise did more work than your specific picks. "
            "You'd have done similarly just buying an index fund."
        )

    text = f"Your portfolio has {beta_desc}{alpha_desc}"
    return text, tone


def analyse_rolling_vol(recent_vol, avg_vol):
    diff = recent_vol - avg_vol
    text = (
        f"Your portfolio's risk over the past {rolling_window} days is "
        f"<strong>{recent_vol:.1%}</strong>, compared to its long-run average of "
        f"<strong>{avg_vol:.1%}</strong>. "
    )
    if diff > 0.05:
        text += (
            "Risk is <strong>rising</strong> right now — your portfolio is bumpier than "
            "usual lately. This might reflect broader market uncertainty or news affecting "
            "your specific stocks."
        )
        tone = "warn"
    elif diff < -0.05:
        text += (
            "Risk is <strong>falling</strong> — your portfolio has been unusually calm recently. "
            "A quieter period. Worth reviewing if your goals have changed."
        )
        tone = "good"
    else:
        text += (
            "Risk is <strong>stable</strong> — recent behaviour is close to your historical "
            "average. No unusual turbulence right now."
        )
        tone = "good"
    return text, tone


def analyse_frontier(your_sharpe, max_sharpe, max_sharpe_weights, tickers):
    gap = max_sharpe - your_sharpe
    top_ticker = max(max_sharpe_weights, key=max_sharpe_weights.get)
    top_weight = max_sharpe_weights[top_ticker]
    text = (
        f"The chart above shows <strong>5,000 different ways</strong> to split your money "
        f"across these stocks. Your current split scores a Sharpe of <strong>{your_sharpe:.2f}</strong>. "
        f"The mathematically best split scores <strong>{max_sharpe:.2f}</strong> — "
    )
    if gap < 0.1:
        text += "you're already very close to optimal. Your current allocation is working well."
        tone = "good"
    elif gap < 0.3:
        text += (
            f"a modest improvement is possible. The model suggests putting more weight on "
            f"<strong>{top_ticker} ({top_weight:.0%})</strong> for better risk-adjusted returns."
        )
        tone = "warn"
    else:
        text += (
            f"a meaningful improvement is possible by rebalancing. "
            f"The model particularly favours <strong>{top_ticker} ({top_weight:.0%})</strong>. "
            f"Remember: this is based on past data — past performance doesn't guarantee future results."
        )
        tone = "warn"
    return text, tone


def analyse_correlation(corr_matrix):
    mask = np.ones(corr_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    avg_corr = corr_matrix.values[mask].mean()
    text = (
        f"On average, your stocks move together with a correlation of "
        f"<strong>{avg_corr:.2f}</strong> (scale: -1 to +1). "
    )
    if avg_corr > 0.7:
        text += (
            "That's high — your stocks tend to rise and fall together. "
            "If one drops sharply, the others likely will too. "
            "Consider adding stocks from different industries (healthcare, energy, consumer goods) "
            "to spread your risk more effectively."
        )
        tone = "warn"
    elif avg_corr > 0.4:
        text += (
            "That's moderate — your stocks don't move in perfect lockstep, which is healthy. "
            "Some diversification is working, but there's room to spread further."
        )
        tone = "warn"
    else:
        text += (
            "That's low — your stocks are fairly independent of each other. "
            "When one falls, the others don't necessarily follow. Good diversification."
        )
        tone = "good"
    return text, tone


#  Sidebar

with st.sidebar:
    st.title("Settings")
    st.markdown("---")

    # Define portfolio_value early so all sidebar sections can reference it.
    # Section 4 renders the real number_input which overwrites this default.
    portfolio_value = st.session_state.get("portfolio_value_input", 10_000)

    # 1. Tickers
    st.subheader("1. Choose your stocks")
    st.caption(
        "Type the ticker symbols separated by commas. "
        "A ticker is the short code for a company — AAPL = Apple, MSFT = Microsoft, TSLA = Tesla."
    )
    tickers_input = st.text_input(
        "Tickers",
        value="AAPL, MSFT, GOOGL",
        help="Find any company's ticker at finance.yahoo.com — search the company name and look for the symbol.",
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    # 2. Weights
    st.subheader("2. Set your weights")

    # Toggle between % sliders and $ amount inputs
    weight_mode = st.radio(
        "How do you want to split your portfolio?",
        options=["% sliders", "$ amounts"],
        horizontal=True,
        help=(
            "% sliders: drag to set what percentage of your portfolio goes into each stock.\n"
            "$ amounts: type in how many dollars you are putting into each stock — "
            "the percentages are calculated for you automatically."
        ),
    )

    n = len(tickers)
    weights = []

    if weight_mode == "% sliders":
        st.caption(
            "Each slider's max shrinks as others claim their share. The last stock fills whatever is left."
        )

        budget = 100
        raw_vals = []
        default_w = round(100 // n)

        for i, ticker in enumerate(tickers):
            key = f"w_{ticker}"

            if i == n - 1:
                last_val = max(0, budget)
                st.markdown(
                    f"**{ticker}** &nbsp;"
                    f"<span style='font-size:13px;color:gray;font-weight:400'>"
                    f"auto-set to {last_val}%</span>",
                    unsafe_allow_html=True,
                )
                raw_vals.append(last_val)
            else:
                if key in st.session_state:
                    clamped = min(st.session_state[key], budget)
                    if clamped != st.session_state[key]:
                        st.session_state[key] = clamped

                val = st.slider(
                    ticker,
                    min_value=0,
                    max_value=max(budget, 1),
                    value=min(default_w, budget),
                    key=key,
                )
                raw_vals.append(val)
                budget -= val

        weights = [v / 100 for v in raw_vals]
        total_weight = sum(raw_vals)

        if total_weight != 100:
            st.caption(
                f"Total: {total_weight}% — slide down to give room to the last stock."
            )
        else:
            st.caption("100%")

    else:
        # Dollar amount mode — mirrors % slider logic exactly:
        # budget shrinks top-to-bottom, last ticker fills remainder, no sync errors.
        st.caption(
            "Each slider's max shrinks as others claim their share. The last stock fills whatever is left."
        )

        # Reset all keys when portfolio_value changes so amounts re-split equally
        prev_pv = st.session_state.get("prev_portfolio_value", None)
        if prev_pv != portfolio_value:
            for t in tickers:
                st.session_state.pop(f"d_{t}", None)
            st.session_state["prev_portfolio_value"] = portfolio_value

        step = round(max(0.01, portfolio_value / 200), 2)
        default_equal = round(portfolio_value / n, 2)
        budget = float(portfolio_value)
        dollar_vals = []

        for i, ticker in enumerate(tickers):
            key = f"d_{ticker}"

            if i == n - 1:
                # Last ticker: show as locked text — remainder of budget
                last_amt = round(max(0.0, budget), 2)
                st.markdown(
                    f"**{ticker}** &nbsp;"
                    f"<span style='font-size:13px;color:gray;font-weight:400'>"
                    f"auto-set to ${last_amt:,.2f}</span>",
                    unsafe_allow_html=True,
                )
                dollar_vals.append(last_amt)
            else:
                # Clamp saved value to current budget without resetting to zero
                if key in st.session_state:
                    clamped = min(float(st.session_state[key]), budget)
                    if clamped != st.session_state[key]:
                        st.session_state[key] = clamped

                amt = st.slider(
                    ticker,
                    min_value=0.0,
                    max_value=max(budget, 0.01),
                    value=min(float(st.session_state.get(key, default_equal)), budget),
                    step=step,
                    key=key,
                    format="$%.2f",
                    help=f"Drag to set how many dollars go into {ticker}.",
                )
                dollar_vals.append(amt)
                budget -= amt

        total_dollars = sum(dollar_vals)
        weights = (
            [v / total_dollars for v in dollar_vals]
            if total_dollars > 0
            else [1 / n] * n
        )

        # Live summary
        if total_dollars > 0:
            st.markdown(
                "<div style='font-size:13px;color:gray;margin-top:4px'>"
                + "  ".join(
                    f"<b>{t}</b> ${v:,.2f} ({w:.1%})"
                    for t, v, w in zip(tickers, dollar_vals, weights)
                )
                + f"&nbsp;·&nbsp; total <b>${total_dollars:,.2f}</b>"
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Slide at least one ticker above zero.")

        # Keep portfolio_value in sync with what user actually set
        if abs(total_dollars - portfolio_value) > 0.01 and total_dollars > 0:
            st.session_state["portfolio_value_input"] = total_dollars
            portfolio_value = total_dollars
    st.markdown("---")

    # 3. Time period — preset selector
    st.subheader("3. Time period")
    st.caption(
        "How much history should we analyse? "
        "More history = more reliable results. "
        "Under 6 months can give misleading numbers."
    )

    period_options = {
        "1 month": 1,
        "3 months": 3,
        "6 months": 6,
        "1 year": 12,
        "3 years": 36,
        "5 years": 60,
        "Max (10yr)": 120,
    }
    selected_period = st.selectbox(
        "Quick select",
        options=list(period_options.keys()),
        index=3,
        help="Pick a preset. Or toggle below to set exact dates.",
    )

    end_date = pd.Timestamp.today()
    months_back = period_options[selected_period]
    start_date = end_date - pd.DateOffset(months=months_back)

    st.caption(
        f"\U0001f4c5 {start_date.strftime('%d %b %Y')} \u2192 {end_date.strftime('%d %b %Y')}"
    )

    use_custom = st.toggle("Set custom dates instead", value=False)
    if use_custom:
        col1, col2 = st.columns(2)
        with col1:
            start_date = pd.Timestamp(
                st.date_input(
                    "From",
                    value=start_date.date(),
                )
            )
        with col2:
            end_date = pd.Timestamp(
                st.date_input(
                    "To",
                    value=pd.Timestamp.today().date(),
                )
            )
        st.caption(
            f"\U0001f4c5 {start_date.strftime('%d %b %Y')} \u2192 {end_date.strftime('%d %b %Y')}"
        )

    st.markdown("---")

    # 4. Advanced
    st.subheader("4. Advanced settings")
    st.caption(
        "Fine to leave as defaults — but here's what they mean if you're curious."
    )

    # Portfolio value
    portfolio_value = st.number_input(
        "My portfolio value ($)",
        key="portfolio_value_input",
        min_value=1,
        max_value=10_000_000,
        value=10_000,
        step=100,
        help=(
            "Enter the total amount you're analysing. "
            "This converts % figures into real dollar amounts — "
            "so instead of 'you could lose 2%' it says 'you could lose $200'. "
            "Any amount works, even $100."
        ),
    )

    # Rolling window — short labels that don't get cut off
    rolling_window = st.selectbox(
        "Risk chart window",
        options=[20, 30, 60, 90],
        index=1,
        format_func=lambda x: {
            20: "20 days (very reactive)",
            30: "30 days (recommended)",
            60: "60 days (smoother)",
            90: "90 days (long term)",
        }[x],
        help=(
            "How many past days to use when drawing the rolling risk chart. "
            "Shorter = reacts quickly to recent events. "
            "Longer = smoother line, less affected by single days."
        ),
    )

    # Confidence level — simple radio, no broken slider
    with st.expander("VaR confidence level — advanced"):
        st.caption(
            "Controls how we calculate your 'bad day' risk figures. "
            "95% is the industry standard — fine for almost everyone."
        )
        confidence = st.radio(
            "Confidence level",
            options=[0.90, 0.95, 0.99],
            index=1,
            format_func=lambda x: {
                0.90: "90%  —  worst 10% of days",
                0.95: "95%  —  industry standard ",
                0.99: "99%  —  only extreme days",
            }[x],
            help="Higher = more conservative estimate of risk.",
        )

    show_benchmark = st.toggle(
        "Compare vs S&P 500",
        value=True,
        help=(
            "The S&P 500 is a basket of 500 large US companies — it's the standard "
            "benchmark for 'how well is the overall market doing?' "
            "Turning this on shows whether your picks beat just buying everything."
        ),
    )

    st.markdown("---")
    run = st.button("RUN ANALYSIS", type="primary", use_container_width=True)


#  Main area

st.title("Portfolio Risk Analysis Engine")

tab_analyse, tab_learn = st.tabs(["Analysis", "Learn"])


#
# LEARN TAB
#

with tab_learn:
    st.markdown("### Everything explained in plain English")
    st.caption(
        "No jargon. No assumed knowledge. If you're new to investing, start here."
    )

    st.markdown("#### The basics")
    learn_card(
        "What is a stock?",
        "A stock is a tiny slice of ownership in a company. If Apple is worth $3 trillion and you own one share, you own a tiny fraction of that company. When the company grows, your slice is worth more. When it shrinks, so does your slice.",
        "If you bought 1 Apple share at $150 and it rises to $180, you made $30 — a 20% return.",
    )
    learn_card(
        "What is a portfolio?",
        "A portfolio is the collection of stocks you own. Instead of betting everything on one company, most people spread their money across several. If one company has a bad year, the others might be fine. This spreading is called diversification.",
        "You put 40% in Apple, 40% in Microsoft, 20% in Google. If Apple drops 20% but Microsoft rises 10%, your overall loss is only 4%.",
    )
    learn_card(
        "What is a return?",
        "Return is how much your investment grew (or shrank) as a percentage. An annual return of 15% means if you started the year with $10,000, you ended with $11,500.",
        "The S&P 500 has historically returned about 10% per year on average — though some years it's up 30%, others it's down 20%.",
    )

    st.markdown("#### Risk metrics")
    learn_card(
        "What is volatility?",
        "Volatility measures how much your portfolio jumps around day to day. High volatility means big swings — some days you're up 3%, others you're down 3%. Low volatility means a smoother ride. Higher potential returns usually come with higher volatility.",
        "Gold is low volatility. Tesla is high volatility. A savings account has almost zero volatility.",
    )
    learn_card(
        "What is a drawdown?",
        "A drawdown is the biggest drop from a peak before recovery. If your portfolio hit $15,000 then fell to $10,500 before going back up, that's a 30% drawdown. Most people panic-sell at the bottom — understanding this number helps you mentally prepare.",
        "During COVID in 2020, the S&P 500 dropped 34% in 5 weeks. People who sold locked in that loss. People who held recovered fully within 5 months.",
    )
    learn_card(
        "What is VaR (Value at Risk)?",
        "VaR answers: on a typical bad day (the worst 5% of all days historically), what is the most I would expect to lose? It is your bad day budget — not a guarantee, but a realistic estimate based on past data.",
        "VaR of $200 at 95% means: on 95% of days you won't lose more than $200. But on that worst 5% of days, you might lose more.",
    )
    learn_card(
        "What is CVaR (Expected Shortfall)?",
        "CVaR goes one step further than VaR. It asks: on the very worst days (beyond VaR), what do you lose on average? It is always a bigger number than VaR. Banks and regulators prefer CVaR because it tells you how bad the bad days truly get.",
        "If VaR is $200, CVaR might be $320. On the worst 5% of days, your average loss is $320 — not just $200.",
    )

    st.markdown("#### Performance ratios")
    learn_card(
        "What is the Sharpe Ratio?",
        "The Sharpe ratio asks: am I being paid enough for the risk I am taking? It compares your return to what you would earn with zero risk (like a government savings bond). Above 1.0 is good. Above 2.0 is excellent. Below 0 means a savings account would have done better.",
        "Portfolio A returns 15% with low volatility → Sharpe 1.8. Portfolio B returns 20% with very high volatility → Sharpe 0.9. Portfolio A is actually the better choice on a risk-adjusted basis.",
    )
    learn_card(
        "What is the Sortino Ratio?",
        "Like Sharpe, but fairer. Sharpe penalises you for all volatility — even the good days when your portfolio jumps up. Sortino only penalises the bad days. If your portfolio is volatile mainly because it keeps surging upward, Sortino will score it higher than Sharpe.",
        "A portfolio that often surges upward looks better on Sortino than Sharpe. That is why Sortino is usually the higher of the two numbers.",
    )

    st.markdown("#### Market comparison")
    learn_card(
        "What is Beta?",
        "Beta measures how sensitive your portfolio is to the overall stock market. Beta of 1.0 means you move exactly with the market. Beta of 1.3 means when the market drops 10%, you tend to drop 13%. Beta of 0.7 means you would only drop 7% — more defensive.",
        "Utility companies have low beta (0.3–0.6) — stable but slow. Tech stocks often have high beta (1.2–1.8) — more exciting but riskier.",
    )
    learn_card(
        "What is Alpha?",
        "Alpha is the return you earned above and beyond what the market predicted you should earn based on your risk level. Positive alpha means your specific stock picks added real value. Negative alpha means you took on risk without being rewarded for it.",
        "If the market predicted you should return 18% (based on your beta), but you actually returned 25%, your alpha is +7%. You genuinely outperformed.",
    )

    st.markdown("#### The Efficient Frontier")
    learn_card(
        "What is the Efficient Frontier?",
        "Imagine plotting every possible way to split your money across your chosen stocks as a dot on a chart. The curved edge along the top-left of all those dots is the Efficient Frontier — these are the portfolios giving the best possible return for each level of risk. Anything below the curve means you are taking unnecessary risk.",
        "The star (★) marks the mathematically optimal split — the one with the best risk-adjusted return based on your chosen stocks' historical data.",
    )

    st.markdown("#### Correlation")
    learn_card(
        "What is correlation?",
        "Correlation (from -1 to +1) measures how much two stocks move together. Near +1 means they rise and fall in sync — owning both gives you little protection. Near 0 means they are independent. Near -1 means they move opposite to each other — the best natural hedge.",
        "Apple and Microsoft both being tech companies correlate around 0.7–0.8. Adding gold (GLD) might have near-zero correlation with tech — so if tech crashes, gold might hold steady.",
    )


#
# ANALYSIS TAB
#

with tab_analyse:
    if not run:
        st.info(
            "Set up your portfolio in the sidebar and click **Run Analysis** to begin."
        )
        with st.expander("Not sure which stocks to try? Here are some starter ideas"):
            st.markdown(
                """
            **All-American tech** — `AAPL, MSFT, GOOGL, NVDA, META`
            Five of the biggest tech companies. High growth, higher risk.

            **Steady and diversified** — `SPY, QQQ, GLD, TLT`
            A mix of the broad market, tech, gold, and bonds. More balanced.

            **High risk, high reward** — `TSLA, NVDA, AMD, PLTR`
            Exciting companies with big swings. Not for the faint-hearted.
            """
            )
        st.stop()

    #  Fetch and compute

    with st.spinner("Fetching market data and running analysis..."):
        try:
            valid, invalid = validate_tickers(tickers)
            if invalid:
                st.error(
                    f"These tickers weren't found: **{', '.join(invalid)}**. "
                    f"Double-check the spelling — use Yahoo Finance symbols."
                )
                st.stop()

            portfolio_prices, benchmark_prices = fetch_with_benchmark(
                tickers=tickers,
                start_date=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
                end_date=pd.Timestamp(end_date).strftime("%Y-%m-%d"),
            )

            if portfolio_prices.empty:
                st.error(
                    "No data returned. Try a different date range or check your tickers."
                )
                st.stop()

            m = compute_all_metrics(
                prices=portfolio_prices,
                weights=weights,
                portfolio_value=portfolio_value,
                benchmark_prices=benchmark_prices if show_benchmark else None,
                rolling_window=rolling_window,
            )

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    # Unpack
    t = m["tooltips"]
    p = m["period"]
    vc = m["var_cvar"]
    dd = m["max_drawdown"]
    r = m["rolling"]
    f = m["efficient_frontier"]
    ba = m["beta_alpha"]
    ann_vol = m["annualised_volatility"]
    bench_return = ba["benchmark_return"] if ba else None

    #  Warnings

    if p["n_days"] < 120:
        st.warning(
            f"Short time period warning — your analysis only covers {p['n_days']} trading days "
            f"({p['n_years']} years). For reliable results, we recommend at least 6 months (120+ trading days). "
            f"Numbers based on short periods can be misleading — a stock that happened to surge "
            f"over 3 months might look incredible but could fall just as fast."
        )

    if ba and ba["beta"] > 1.8:
        st.warning(
            f"High risk warning — your portfolio has a Beta of {ba['beta']:.2f}, which is very high. "
            f"This means it tends to move {ba['beta']:.1f}x as much as the overall market. "
            f"If the market drops 20%, your portfolio could drop around {ba['beta']*20:.0f}%. "
            f"Make sure you understand and are comfortable with this level of risk."
        )

    st.caption(
        f"Data: {p['start']} — {p['end']}  ·  {p['n_days']} trading days  ·  {p['n_years']} years"
    )

    #  Section 1: Summary metrics

    st.markdown("## Your results at a glance")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "Annual Return", f"{m['annualised_return']:.1%}", help=t["annualised_return"]
    )
    col2.metric(
        "Volatility",
        f"{m['annualised_volatility']:.1%}",
        help=t["annualised_volatility"],
    )
    col3.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}", help=t["sharpe_ratio"])
    col4.metric("Sortino Ratio", f"{m['sortino_ratio']:.2f}", help=t["sortino_ratio"])
    col5.metric("Max Drawdown", f"{dd['max_drawdown']:.1%}", help=t["max_drawdown"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"VaR (95%)", f"-${abs(vc['var_dollar']):,.0f}", help=t["var"])
    col2.metric(f"CVaR (95%)", f"-${abs(vc['cvar_dollar']):,.0f}", help=t["cvar"])
    if ba:
        col3.metric("Beta", f"{ba['beta']:.2f}", help=t["beta"])
        col4.metric("Alpha", f"{ba['alpha']:.2%}", help=t["alpha"])

    text, tone = analyse_return(m["annualised_return"], portfolio_value, bench_return)
    insight("Overall performance", text, tone)

    text, tone = analyse_volatility(m["annualised_volatility"])
    insight("Volatility", text, tone)

    st.markdown("---")

    #  Section 2: Cumulative returns

    st.markdown("## How your portfolio grew over time")
    st.caption(
        "This chart shows the total return from the start of the period. "
        "10% means every $1 you invested is now worth $1.10. "
        "The dashed line (if shown) is the S&P 500 for comparison."
    )

    port_cumret = (1 + m["portfolio_returns"]).cumprod() - 1
    fig_ret = go.Figure()

    fig_ret.add_trace(
        go.Scatter(
            x=port_cumret.index,
            y=port_cumret.values,
            name="Your Portfolio",
            line=dict(color="#2563eb", width=2.5),
            hovertemplate="%{x}<br>Total return: %{y:.1%}<extra></extra>",
        )
    )

    if show_benchmark and benchmark_prices is not None:
        bench_rets = benchmark_prices.pct_change().dropna().iloc[:, 0]
        bench_cumret = (1 + bench_rets).cumprod() - 1
        bench_cumret = bench_cumret.reindex(port_cumret.index)
        fig_ret.add_trace(
            go.Scatter(
                x=bench_cumret.index,
                y=bench_cumret.values,
                name="S&P 500 (the whole market)",
                line=dict(color="#94a3b8", width=1.5, dash="dash"),
                hovertemplate="%{x}<br>S&P 500 return: %{y:.1%}<extra></extra>",
            )
        )

    fig_ret.update_layout(
        yaxis_tickformat=".0%",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=0, r=0, t=30, b=0),
        height=360,
    )
    st.plotly_chart(fig_ret, use_container_width=True)

    text, tone = analyse_return(m["annualised_return"], portfolio_value, bench_return)
    insight("Growth chart", text, tone)

    st.markdown("---")

    #  Section 3: Drawdown + Rolling vol

    st.markdown("## Risk over time")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Drawdown")
        st.caption(
            "How far below its peak your portfolio was at each point in time. "
            "0% means you're at an all-time high. A dip to -20% means your portfolio "
            "was 20% below its highest value at that moment."
        )
        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                x=dd["drawdown_series"].index,
                y=dd["drawdown_series"].values,
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.15)",
                line=dict(color="#ef4444", width=1.5),
                hovertemplate="%{x}<br>Drop from peak: %{y:.1%}<extra></extra>",
            )
        )
        fig_dd.update_layout(
            yaxis_tickformat=".0%", margin=dict(l=0, r=0, t=10, b=0), height=280
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        text, tone = analyse_drawdown(dd["max_drawdown"], portfolio_value)
        insight("Drawdown", text, tone)

    with col_right:
        st.markdown("### Rolling volatility")
        st.caption(
            f"Risk level recalculated using only the most recent {rolling_window} days at each point. "
            f"Peaks = stressful, volatile periods. Troughs = calm stretches. "
            f"Right-most value = your risk level today."
        )
        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Scatter(
                x=r["rolling_volatility"].index,
                y=r["rolling_volatility"].values,
                line=dict(color="#f59e0b", width=2),
                fill="tozeroy",
                fillcolor="rgba(245,158,11,0.1)",
                hovertemplate="%{x}<br>Risk level: %{y:.1%}<extra></extra>",
            )
        )
        fig_vol.update_layout(
            yaxis_tickformat=".0%", margin=dict(l=0, r=0, t=10, b=0), height=280
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        recent_vol = r["rolling_volatility"].dropna().iloc[-1]
        text, tone = analyse_rolling_vol(recent_vol, ann_vol)
        insight("Rolling volatility", text, tone)

    st.markdown("---")

    #  Section 4: Risk-adjusted ratios

    st.markdown("## Was the risk worth it?")
    st.caption(
        "A big return that came with massive swings isn't necessarily good. "
        "These scores tell you how much return you got *per unit of risk taken*."
    )

    col1, col2 = st.columns(2)
    with col1:
        text, tone = analyse_sharpe(m["sharpe_ratio"])
        insight("Sharpe ratio", text, tone)
    with col2:
        text, tone = analyse_sortino(m["sortino_ratio"], m["sharpe_ratio"])
        insight("Sortino ratio", text, tone)

    text, tone = analyse_var(vc["var_pct"], vc["cvar_pct"], portfolio_value)
    insight("Your risk in dollars — VaR & CVaR", text, tone)

    if ba:
        text, tone = analyse_beta_alpha(ba["beta"], ba["alpha"], ba["benchmark_return"])
        insight("How you compare to the market — Beta & Alpha", text, tone)

    st.markdown("---")

    #  Section 5: Efficient Frontier

    st.markdown("## Could you do better with the same stocks?")
    st.caption(
        "Each dot is a different way to split your money across your stocks. "
        "Dots further up = more return. Dots further right = more risk. "
        "The best portfolios are up and to the left. Where does yours sit?"
    )

    fig_ef = go.Figure()
    fig_ef.add_trace(
        go.Scatter(
            x=f["vols"],
            y=f["returns"],
            mode="markers",
            marker=dict(
                color=f["sharpes"],
                colorscale="Viridis",
                size=4,
                opacity=0.6,
                colorbar=dict(title="Sharpe score", thickness=12),
            ),
            name="5,000 simulated portfolios",
            hovertemplate="Risk: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>",
        )
    )
    fig_ef.add_trace(
        go.Scatter(
            x=[f["max_sharpe_vol"]],
            y=[f["max_sharpe_return"]],
            mode="markers+text",
            marker=dict(
                symbol="star",
                size=18,
                color="#f59e0b",
                line=dict(color="white", width=1),
            ),
            text=["★ Best mix"],
            textposition="top right",
            name=f"Best mix (score: {f['max_sharpe_sharpe']:.2f})",
            hovertemplate=(
                f"Best mix<br>Return: {f['max_sharpe_return']:.1%}<br>"
                f"Risk: {f['max_sharpe_vol']:.1%}<br>Score: {f['max_sharpe_sharpe']:.2f}<extra></extra>"
            ),
        )
    )
    fig_ef.add_trace(
        go.Scatter(
            x=[f["min_vol_vol"]],
            y=[f["min_vol_return"]],
            mode="markers+text",
            marker=dict(
                symbol="diamond",
                size=14,
                color="#10b981",
                line=dict(color="white", width=1),
            ),
            text=["◆ Safest mix"],
            textposition="top right",
            name="Safest mix",
            hovertemplate=(
                f"Safest mix<br>Return: {f['min_vol_return']:.1%}<br>"
                f"Risk: {f['min_vol_vol']:.1%}<extra></extra>"
            ),
        )
    )
    fig_ef.add_trace(
        go.Scatter(
            x=[m["annualised_volatility"]],
            y=[m["annualised_return"]],
            mode="markers+text",
            marker=dict(
                symbol="circle",
                size=14,
                color="#2563eb",
                line=dict(color="white", width=1),
            ),
            text=["● Your portfolio"],
            textposition="top right",
            name=f"Your portfolio",
            hovertemplate=(
                f"Your portfolio<br>Return: {m['annualised_return']:.1%}<br>"
                f"Risk: {m['annualised_volatility']:.1%}<br>Score: {m['sharpe_ratio']:.2f}<extra></extra>"
            ),
        )
    )
    fig_ef.update_layout(
        xaxis_title="Risk level →  (less risk on the left)",
        yaxis_title="Return →  (more return at the top)",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=0, r=0, t=40, b=0),
        height=460,
    )
    st.plotly_chart(fig_ef, use_container_width=True)

    text, tone = analyse_frontier(
        m["sharpe_ratio"],
        f["max_sharpe_sharpe"],
        f["max_sharpe_weights"],
        tickers,
    )
    insight("Efficient Frontier", text, tone)

    with st.expander("Optimal weights (max Sharpe)"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Suggested split**")
            for ticker, w in f["max_sharpe_weights"].items():
                st.markdown(f"- **{ticker}**: {w:.1%}")
            st.caption("Based on historical data only — not financial advice.")
        with c2:
            st.markdown("**What you'd get**")
            st.markdown(f"- Return: **{f['max_sharpe_return']:.1%}**")
            st.markdown(f"- Risk: **{f['max_sharpe_vol']:.1%}**")
            st.markdown(f"- Score: **{f['max_sharpe_sharpe']:.2f}**")

    st.markdown("---")

    #  Section 5b: Monte Carlo simulation

    st.markdown("## What could your portfolio be worth in a year?")
    st.caption(
        "We ran 1,000 simulations of your portfolio's possible future over the next 12 months, "
        "based on its historical returns and volatility. "
        "Each line is one possible future — not a prediction, but an honest range of outcomes."
    )

    mc = m["monte_carlo"]

    # Build the chart
    fig_mc = go.Figure()

    # Plot a sample of individual paths (translucent grey) — not all 1000, too slow
    sample_idx = np.random.choice(
        mc["n_simulations"], size=min(200, mc["n_simulations"]), replace=False
    )
    days = list(range(mc["n_days"] + 1))

    for idx in sample_idx:
        fig_mc.add_trace(
            go.Scatter(
                x=days,
                y=mc["all_paths"][:, idx],
                mode="lines",
                line=dict(color="rgba(148,163,184,0.15)", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Shaded band between 5th and 95th percentile
    fig_mc.add_trace(
        go.Scatter(
            x=days + days[::-1],
            y=list(mc["percentile_95"]) + list(mc["percentile_5"])[::-1],
            fill="toself",
            fillcolor="rgba(37,99,235,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="90% of outcomes land here",
            hoverinfo="skip",
        )
    )

    # 5th percentile — bad scenario
    fig_mc.add_trace(
        go.Scatter(
            x=days,
            y=mc["percentile_5"],
            mode="lines",
            line=dict(color="#ef4444", width=2, dash="dash"),
            name=f"Bad scenario (5th %ile) — ${mc['p5_final']:,.0f}",
            hovertemplate="Day %{x}<br>Value: $%{y:,.0f}<extra>Bad scenario</extra>",
        )
    )

    # Median — middle scenario
    fig_mc.add_trace(
        go.Scatter(
            x=days,
            y=mc["percentile_50"],
            mode="lines",
            line=dict(color="#2563eb", width=2.5),
            name=f"Middle scenario (median) — ${mc['p50_final']:,.0f}",
            hovertemplate="Day %{x}<br>Value: $%{y:,.0f}<extra>Middle scenario</extra>",
        )
    )

    # 95th percentile — good scenario
    fig_mc.add_trace(
        go.Scatter(
            x=days,
            y=mc["percentile_95"],
            mode="lines",
            line=dict(color="#10b981", width=2, dash="dash"),
            name=f"Good scenario (95th %ile) — ${mc['p95_final']:,.0f}",
            hovertemplate="Day %{x}<br>Value: $%{y:,.0f}<extra>Good scenario</extra>",
        )
    )

    fig_mc.update_layout(
        xaxis_title="Trading days from today (252 = 1 year)",
        yaxis_title="Portfolio value ($)",
        yaxis_tickformat="$,.0f",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0),
        height=440,
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # Outcome summary cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Starting value",
        f"${mc['portfolio_value']:,.0f}",
        help="Your current portfolio value.",
    )
    col2.metric(
        "Median outcome",
        f"${mc['p50_final']:,.0f}",
        delta=f"{(mc['p50_final']/mc['portfolio_value']-1):.1%}",
        help="The middle outcome — half of simulations end above this, half below.",
    )
    col3.metric(
        "Chance of profit",
        f"{mc['prob_profit']:.0%}",
        help="Percentage of simulations where you end the year with more than you started.",
    )
    col4.metric(
        "Chance of -10% loss",
        f"{mc['prob_loss_10pct']:.0%}",
        help="Percentage of simulations where you lose more than 10% of your portfolio.",
    )

    # Plain-English insight
    prob_profit_pct = mc["prob_profit"] * 100
    prob_loss_pct = mc["prob_loss_10pct"] * 100
    gain_median = mc["p50_final"] - mc["portfolio_value"]
    gain_good = mc["p95_final"] - mc["portfolio_value"]
    loss_bad = mc["portfolio_value"] - mc["p5_final"]

    mc_text = (
        f"Based on your portfolio's historical behaviour, we simulated <strong>1,000 possible futures</strong> "
        f"for the next 12 months. In <strong>{prob_profit_pct:.0f}% of scenarios</strong> you end the year "
        f"with more than you started. "
        f"The middle outcome is <strong>${mc['p50_final']:,.0f}</strong> "
        f"(a {'gain' if gain_median >= 0 else 'loss'} of ${abs(gain_median):,.0f}). "
        f"In a good year (top 5% of outcomes) you could reach <strong>${mc['p95_final']:,.0f}</strong> "
        f"(+${gain_good:,.0f}). "
        f"In a bad year (bottom 5%) you could fall to <strong>${mc['p5_final']:,.0f}</strong> "
        f"(-${loss_bad:,.0f}). "
    )
    if prob_loss_pct > 20:
        mc_text += (
            f"<strong>Worth noting:</strong> there's a {prob_loss_pct:.0f}% chance of losing more than 10% — "
            f"make sure you're comfortable with that before investing."
        )
        mc_tone = "warn"
    elif mc["prob_profit"] > 0.7:
        mc_text += f"The odds are in your favour, but remember — past performance does not guarantee future results."
        mc_tone = "good"
    else:
        mc_text += "This is a fairly balanced risk profile — meaningful upside with meaningful downside."
        mc_tone = "warn"

    insight("Monte Carlo — your possible futures", mc_text, mc_tone)

    st.markdown("---")

    #  Section 6: Correlation + Allocation

    st.markdown("## How your stocks relate to each other")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Correlation matrix")
        st.caption(
            "Dark red = move very similarly (less protection if one drops). "
            "Blue = move opposite (better protection). "
            "You want a mix of colours, not all red."
        )
        corr = m["correlation_matrix"]
        fig_corr = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                text=corr.round(2).values,
                texttemplate="%{text}",
                textfont=dict(size=13),
                hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.2f}<extra></extra>",
            )
        )
        fig_corr.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300)
        st.plotly_chart(fig_corr, use_container_width=True)

        text, tone = analyse_correlation(corr)
        insight("Correlation", text, tone)

    with col_right:
        st.markdown("### Allocation")
        st.caption("How your money is currently divided across your stocks.")
        fig_pie = go.Figure(
            go.Pie(
                labels=tickers,
                values=[w * 100 for w in weights],
                hole=0.4,
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
            )
        )
        fig_pie.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    #  Section 7: Raw data

    with st.expander("View raw price data"):
        st.caption(
            f"Closing prices used in this analysis. Showing last 20 of {len(portfolio_prices)} trading days."
        )
        st.dataframe(
            portfolio_prices.tail(20).style.format("{:.2f}"), use_container_width=True
        )

    st.caption(
        "This tool is for educational purposes only and does not constitute financial advice. "
        "Past performance does not guarantee future results. Always do your own research."
    )
