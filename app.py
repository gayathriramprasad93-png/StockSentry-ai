import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="StockSentry AI",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background-color: #0b1120; }
[data-testid="stMetric"] {
    background: #111c30;
    border: 1px solid #253650;
    padding: 14px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 StockSentry AI")
st.caption("Indian Stock Intelligence • Risk Detection • Machine Learning")
st.info(
    "Educational research prototype. Predictions are uncertain and are not "
    "financial advice. Historical data may be delayed or unavailable."
)

POPULAR_STOCKS = {
    "Reliance Industries": "RELIANCE",
    "HDFC Bank": "HDFCBANK",
    "ICICI Bank": "ICICIBANK",
    "State Bank of India": "SBIN",
    "Axis Bank": "AXISBANK",
    "Kotak Mahindra Bank": "KOTAKBANK",
    "Infosys": "INFY",
    "Tata Consultancy Services": "TCS",
    "Wipro": "WIPRO",
    "HCL Technologies": "HCLTECH",
    "Bharti Airtel": "BHARTIARTL",
    "Larsen & Toubro": "LT",
    "ITC": "ITC",
    "Mahindra & Mahindra": "M&M",
    "Maruti Suzuki": "MARUTI",
    "Tata Steel": "TATASTEEL",
    "JSW Steel": "JSWSTEEL",
    "Sun Pharmaceutical": "SUNPHARMA",
    "Tata Consumer Products": "TATACONSUM",
    "Titan Company": "TITAN",
    "Asian Paints": "ASIANPAINT",
    "Bajaj Finance": "BAJFINANCE",
    "Power Grid": "POWERGRID",
    "NTPC": "NTPC",
    "Adani Ports": "ADANIPORTS",
    "UltraTech Cement": "ULTRACEMCO",
    "Eicher Motors": "EICHERMOT",
    "Hindustan Unilever": "HINDUNILVR",
    "Coal India": "COALINDIA",
    "Tata Motors Passenger Vehicles": "TMPV",
}


def make_ticker(symbol, exchange):
    symbol = str(symbol).strip().upper()
    if symbol.endswith((".NS", ".BO")):
        return symbol
    return symbol + (".NS" if exchange == "NSE" else ".BO")


@st.cache_data(ttl=900, show_spinner=False)
def load_stock(ticker, period="5y"):
    try:
        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in data.columns for c in required):
            return pd.DataFrame()

        data = data[required].copy()
        for col in required:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.dropna(subset=["Close"])
        return data[~data.index.duplicated(keep="last")].sort_index()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=21600, show_spinner=False)
def load_annual_financials(ticker):
    try:
        statement = yf.Ticker(ticker).income_stmt
        if statement is None or statement.empty:
            return None

        output = {}
        for label, candidates in [
            ("Revenue", ["Total Revenue", "Operating Revenue"]),
            ("Net income", ["Net Income", "Net Income Common Stockholders"]),
        ]:
            row = next((r for r in candidates if r in statement.index), None)
            if row is None:
                output[label] = np.nan
                continue

            values = pd.to_numeric(
                statement.loc[row], errors="coerce"
            ).dropna()
            values.index = pd.to_datetime(values.index)
            values = values.sort_index()

            if len(values) < 2:
                output[label] = np.nan
                continue

            previous = float(values.iloc[-2])
            latest = float(values.iloc[-1])

            if previous > 0:
                output[label] = (latest / previous - 1) * 100
            else:
                output[label] = np.nan

        return output
    except Exception:
        return None


def make_features(data):
    close = data["Close"]
    volume = data["Volume"].replace(0, np.nan)
    features = pd.DataFrame(index=data.index)

    features["return_1d"] = close.pct_change()
    features["return_5d"] = close.pct_change(5)
    features["return_20d"] = close.pct_change(20)
    features["volatility_20d"] = features["return_1d"].rolling(20).std()
    features["volatility_60d"] = features["return_1d"].rolling(60).std()
    features["ma_ratio_20"] = close / close.rolling(20).mean() - 1
    features["ma_ratio_50"] = close / close.rolling(50).mean() - 1
    features["ma_ratio_200"] = close / close.rolling(200).mean() - 1
    features["drawdown_60d"] = close / close.rolling(60).max() - 1
    features["volume_ratio_20d"] = volume / volume.rolling(20).mean()

    return features.replace([np.inf, -np.inf], np.nan)


FEATURES = [
    "return_1d", "return_5d", "return_20d",
    "volatility_20d", "volatility_60d",
    "ma_ratio_20", "ma_ratio_50", "ma_ratio_200",
    "drawdown_60d", "volume_ratio_20d",
]


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("forest", RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def train_model(data):
    features = make_features(data)
    future_return = data["Close"].shift(-20) / data["Close"] - 1

    dataset = features.copy()
    dataset["target"] = (future_return < 0).astype(float)
    dataset.loc[future_return.isna(), "target"] = np.nan
    dataset = dataset.dropna(subset=["target"])

    if len(dataset) < 250:
        return None, None, "Not enough historical data. Try a longer period."

    X = dataset[FEATURES]
    y = dataset["target"].astype(int)
    split = int(len(dataset) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return None, None, "Not enough target-class variation for evaluation."

    model = build_model()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    scores = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "Test observations": len(y_test),
        "Test start": str(y_test.index.min().date()),
        "Test end": str(y_test.index.max().date()),
    }

    final_model = build_model()
    final_model.fit(X, y)
    latest = features[FEATURES].iloc[[-1]]

    if latest.isna().all(axis=None):
        return None, scores, "Latest indicators are unavailable."

    probability = final_model.predict_proba(latest)[0, 1]
    return {
        "probability": float(probability),
        "model": final_model,
    }, scores, None


def calculate_metrics(data):
    close = data["Close"]
    returns = close.pct_change().dropna()
    drawdown = (close / close.cummax() - 1) * 100

    return {
        "price": float(close.iloc[-1]),
        "return": float((close.iloc[-1] / close.iloc[0] - 1) * 100),
        "volatility": float(returns.std() * np.sqrt(252) * 100),
        "max_drawdown": float(drawdown.min()),
        "ma50": close.rolling(50).mean().iloc[-1],
        "ma200": close.rolling(200).mean().iloc[-1],
    }


def get_risk_signals(data, metrics):
    close = data["Close"]
    signals = []
    r20 = close.pct_change(20).iloc[-1]
    r60 = close.pct_change(60).iloc[-1]

    if pd.notna(r20) and r20 < -0.10:
        signals.append(("High", "Price declined over 10% in 20 sessions."))
    if pd.notna(r60) and r60 < -0.15:
        signals.append(("High", "Price declined over 15% in 60 sessions."))
    if pd.notna(metrics["ma200"]) and metrics["price"] < metrics["ma200"]:
        signals.append(("Moderate", "Price is below its 200-day average."))
    if pd.notna(metrics["volatility"]) and metrics["volatility"] > 45:
        signals.append(("High", "Annualized volatility exceeds 45%."))
    if metrics["max_drawdown"] < -30:
        signals.append(("High", "Historical maximum drawdown exceeds 30%."))

    if not signals:
        signals.append((
            "No configured warning triggered",
            "This does not mean the stock is safe or a good investment.",
        ))
    return signals


def format_inr(value):
    return f"₹{value:,.2f}"

# ==================== SIDEBAR & APP CONTROLS ====================

st.sidebar.title("📈 StockSentry AI")
st.sidebar.caption("Indian Stock Intelligence")

exchange = st.sidebar.selectbox("Exchange", ["NSE", "BSE"])
default_company = "Reliance Industries"
company_names = list(POPULAR_STOCKS.keys())

selected_company = st.sidebar.selectbox(
    "Choose a company",
    company_names,
    index=company_names.index(default_company),
)
custom_symbol = st.sidebar.text_input(
    "Or enter a ticker",
    placeholder="e.g. TCS or INFY",
)
symbol = custom_symbol.strip() or POPULAR_STOCKS[selected_company]
ticker = make_ticker(symbol, exchange)

period_label = st.sidebar.selectbox(
    "Historical data period",
    ["6mo", "1y", "2y", "5y", "10y"],
    index=3,
)
st.sidebar.caption("Data provider: Yahoo Finance")

with st.spinner(f"Loading {ticker}..."):
    stock = load_stock(ticker, period_label)

if stock.empty or len(stock) < 30:
    st.error(
        f"Historical data for {ticker} is unavailable or insufficient. "
        "Try another symbol or exchange."
    )
    st.stop()

metrics = calculate_metrics(stock)
close = stock["Close"]

# ==================== ORIGINAL TABS ====================

tabs = st.tabs([
    "📊 Market Overview",
    "⚠️ Risk Scanner",
    "🤖 ML Lab",
    "🔄 Compare Stocks",
    "🚀 Growth Scanner",
    "💼 Portfolio Analyzer",
    "🔎 Smart Screener",
    "🧠 AI Explainer",
    "🧪 Backtesting",
    "🎮 Paper Trading",
    "📰 News & Sentiment",
    "🔔 Watchlist & Alerts",
])

# ----- TAB 1: MARKET OVERVIEW -----

with tabs[0]:
    st.subheader(f"Market Overview — {ticker}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest price", format_inr(metrics["price"]))
    c2.metric("Period return", f'{metrics["return"]:.2f}%')
    c3.metric("Annualized volatility", f'{metrics["volatility"]:.2f}%')
    c4.metric("Maximum drawdown", f'{metrics["max_drawdown"]:.2f}%')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stock.index, y=close, name="Adjusted close",
        line=dict(width=2),
    ))
    fig.add_trace(go.Scatter(
        x=stock.index, y=close.rolling(50).mean(),
        name="50-session average",
    ))
    fig.add_trace(go.Scatter(
        x=stock.index, y=close.rolling(200).mean(),
        name="200-session average",
    ))
    fig.update_layout(
        template="plotly_dark",
        title=f"{ticker} historical price",
        height=480,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Adjusted historical prices are not a guarantee of future returns."
    )

    with st.expander("View recent historical data"):
        st.dataframe(stock.tail(30), use_container_width=True)

    st.download_button(
        "Download historical data (CSV)",
        stock.to_csv().encode("utf-8"),
        file_name=f"{ticker.replace('.', '_')}_history.csv",
        mime="text/csv",
    )

# ----- TAB 2: RISK SCANNER -----

with tabs[1]:
    st.subheader(f"Risk Scanner — {ticker}")
    signals = get_risk_signals(stock, metrics)

    for level, description in signals:
        if level == "High":
            st.error(f"🔴 {description}")
        elif level == "Moderate":
            st.warning(f"🟠 {description}")
        else:
            st.info(f"ℹ️ {description}")

    st.markdown("### Risk indicators")
    r1, r2, r3 = st.columns(3)
    r1.metric("50-session average", (
        format_inr(metrics["ma50"])
        if pd.notna(metrics["ma50"]) else "Unavailable"
    ))
    r2.metric("200-session average", (
        format_inr(metrics["ma200"])
        if pd.notna(metrics["ma200"]) else "Unavailable"
    ))
    r3.metric("Historical maximum drawdown",
              f'{metrics["max_drawdown"]:.2f}%')

    st.caption(
        "These are rule-based warnings, not a complete assessment of "
        "business, valuation, liquidity, or market risk."
    )

# ----- TAB 3: ML LAB -----

with tabs[2]:
    st.subheader("Machine Learning Lab")
    st.write(
        "A Random Forest estimates the historical probability that the "
        "stock's return over the following 20 trading sessions will be negative."
    )

    if st.button("Train and evaluate model", key="train_main_model"):
        with st.spinner("Training model and evaluating a chronological holdout..."):
            prediction, scores, error = train_model(stock)

        if error:
            st.warning(error)
        else:
            st.session_state["main_prediction"] = prediction
            st.session_state["main_scores"] = scores
            st.success("Model trained and evaluated.")

    if "main_scores" in st.session_state:
        scores = st.session_state["main_scores"]
        cols = st.columns(4)
        for col, name in zip(
            cols, ["Accuracy", "Precision", "Recall", "ROC-AUC"]
        ):
            col.metric(name, f'{scores[name] * 100:.1f}%')

        st.write(
            f"Chronological test window: {scores['Test start']} to "
            f"{scores['Test end']} "
            f"({scores['Test observations']} observations)."
        )

        prediction = st.session_state.get("main_prediction")
        if prediction:
            probability = prediction["probability"]
            st.metric(
                "Estimated probability of negative 20-session return",
                f"{probability * 100:.1f}%",
            )
            st.progress(float(np.clip(probability, 0, 1)))
            st.caption(
                "This is a model estimate, not a reliable probability of "
                "loss in every market environment. Past test performance "
                "does not guarantee future performance."
            )

# ----- TAB 4: COMPARE STOCKS -----

with tabs[3]:
    st.subheader("Compare Stocks")

    compare_text = st.text_input(
        "Enter up to 6 other tickers, separated by commas",
        value="TCS, INFY, HDFCBANK",
    )
    compare_period = st.selectbox(
        "Comparison period",
        ["6mo", "1y", "2y", "5y"],
        index=1,
        key="compare_period",
    )

    if st.button("Compare selected stocks", key="run_compare"):
        comparison_rows = []
        fig = go.Figure()

        symbols = [
            item.strip() for item in compare_text.split(",") if item.strip()
        ]
        symbols = symbols[:6] + [symbol]

        with st.spinner("Loading comparison data..."):
            for item in dict.fromkeys(symbols):
                comparison_ticker = make_ticker(item, exchange)
                history = load_stock(comparison_ticker, compare_period)

                if history.empty or len(history) < 10:
                    continue

                prices = history["Close"]
                daily_returns = prices.pct_change().dropna()
                drawdown = (prices / prices.cummax() - 1).min() * 100
                period_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100

                comparison_rows.append({
                    "Ticker": comparison_ticker,
                    "Latest price": round(float(prices.iloc[-1]), 2),
                    "Period return (%)": round(float(period_return), 2),
                    "Annualized volatility (%)": round(
                        float(daily_returns.std() * np.sqrt(252) * 100), 2
                    ),
                    "Maximum drawdown (%)": round(float(drawdown), 2),
                })

                normalized = prices / prices.iloc[0] * 100
                fig.add_trace(go.Scatter(
                    x=normalized.index,
                    y=normalized,
                    name=comparison_ticker,
                ))

        if comparison_rows:
            fig.update_layout(
                template="plotly_dark",
                title="Relative performance (starting value = 100)",
                yaxis_title="Indexed price",
                height=430,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
            comparison_df = pd.DataFrame(comparison_rows)
            st.dataframe(
                comparison_df, hide_index=True, use_container_width=True
            )
            st.download_button(
                "Download comparison CSV",
                comparison_df.to_csv(index=False).encode("utf-8"),
                file_name="stocksentry_comparison.csv",
                mime="text/csv",
            )
        else:
            st.warning("No comparison data was available.")

# ----- TAB 5: GROWTH SCANNER -----

with tabs[4]:
    st.subheader("🚀 Growth Scanner")
    st.write(
        "Explore historical price momentum and available annual revenue "
        "and net-income growth."
    )

    selected_names = st.multiselect(
        "Popular companies to scan",
        company_names,
        default=[
            "Reliance Industries", "HDFC Bank", "ICICI Bank",
            "Infosys", "Tata Consultancy Services", "NTPC",
        ],
        key="growth_names",
    )
    extra_symbols = st.text_input(
        "Additional ticker symbols, comma-separated",
        placeholder="TRENT, BEL, ZOMATO",
        key="growth_extra",
    )
    minimum_return = st.slider(
        "Minimum 60-session price return (%)",
        -30, 30, 0, 5,
    )

    if st.button("Scan growing stocks", key="growth_run"):
        candidates = [
            (name, POPULAR_STOCKS[name]) for name in selected_names
        ]
        candidates.extend(
            (s.strip().upper(), s.strip().upper())
            for s in extra_symbols.split(",") if s.strip()
        )

        unique = {}
        for name, raw_symbol in candidates:
            unique[make_ticker(raw_symbol, exchange)] = name
        unique = dict(list(unique.items())[:50])

        rows = []
        progress = st.progress(0)

        for index, (scan_ticker, name) in enumerate(unique.items()):
            history = load_stock(scan_ticker, "2y")
            row = {
                "Company": name,
                "Ticker": scan_ticker,
                "Latest price": np.nan,
                "20-session return (%)": np.nan,
                "60-session return (%)": np.nan,
                "1-year return (%)": np.nan,
                "Revenue growth (%)": np.nan,
                "Net income growth (%)": np.nan,
            }

            if not history.empty and len(history) >= 20:
                prices = history["Close"]
                row["Latest price"] = round(float(prices.iloc[-1]), 2)
                row["20-session return (%)"] = round(
                    float(prices.iloc[-1] / prices.iloc[-21] * 100 - 100), 2
                )
                if len(prices) >= 61:
                    row["60-session return (%)"] = round(
                        float(prices.iloc[-1] / prices.iloc[-61] * 100 - 100), 2
                    )
                if len(prices) >= 252:
                    row["1-year return (%)"] = round(
                        float(prices.iloc[-1] / prices.iloc[-252] * 100 - 100), 2
                    )

                financials = load_annual_financials(scan_ticker)
                if financials:
                    row["Revenue growth (%)"] = financials.get("Revenue", np.nan)
                    row["Net income growth (%)"] = financials.get("Net income", np.nan)

            ret = row["60-session return (%)"]
            rev = row["Revenue growth (%)"]
            income = row["Net income growth (%)"]

            if pd.isna(ret):
                row["Screen result"] = "Insufficient price history"
            elif ret < minimum_return:
                row["Screen result"] = "Below momentum threshold"
            elif pd.notna(rev) and pd.notna(income) and rev > 0 and income > 0:
                row["Screen result"] = "Positive momentum + revenue/profit growth"
            elif ret > 0:
                row["Screen result"] = "Positive momentum; fundamentals mixed/incomplete"
            else:
                row["Screen result"] = "Price threshold met; verify fundamentals"

            rows.append(row)
            progress.progress((index + 1) / max(len(unique), 1))

        progress.empty()
        results = pd.DataFrame(rows).sort_values(
            "60-session return (%)", ascending=False, na_position="last"
        )
        st.dataframe(results, hide_index=True, use_container_width=True)
        st.download_button(
            "Download Growth Scanner CSV",
            results.to_csv(index=False).encode("utf-8"),
            file_name="stocksentry_growth_scanner.csv",
            mime="text/csv",
        )

        chart_data = results.dropna(subset=["60-session return (%)"])
        if not chart_data.empty:
            fig = go.Figure(go.Bar(
                x=chart_data["60-session return (%)"],
                y=chart_data["Ticker"],
                orientation="h",
            ))
            fig.update_layout(
                template="plotly_dark",
                title="60-session price return (%)",
                height=max(350, 28 * len(chart_data)),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.warning(
            "The scanner only checks selected symbols. It does not cover every "
            "listed stock, and available financial data may be incomplete."
        )

# ----- TAB 6: PORTFOLIO ANALYZER -----

with tabs[5]:
    st.subheader("💼 Portfolio Analyzer")
    st.write(
        "Enter holdings to estimate market value, unrealized profit/loss "
        "and portfolio allocation. Holdings entered here are not saved permanently."
    )

    portfolio_text = st.text_area(
        "One holding per line: ticker, quantity, purchase price",
        placeholder="RELIANCE, 10, 1200\nTCS, 5, 3500\nINFY, 12, 1400",
        key="portfolio_input",
    )

    if st.button("Analyze portfolio", key="portfolio_run"):
        portfolio_rows = []
        for line in portfolio_text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                raw, quantity, buy_price = parts
                quantity = float(quantity)
                buy_price = float(buy_price)
                if quantity <= 0 or buy_price <= 0:
                    continue
                holding_ticker = make_ticker(raw, exchange)
                history = load_stock(holding_ticker, "6mo")
                if history.empty:
                    continue
                current_price = float(history["Close"].iloc[-1])
                cost = quantity * buy_price
                value = quantity * current_price
                portfolio_rows.append({
                    "Ticker": holding_ticker,
                    "Quantity": quantity,
                    "Purchase price": buy_price,
                    "Current price": current_price,
                    "Cost basis": cost,
                    "Market value": value,
                    "Unrealized P/L": value - cost,
                    "P/L (%)": (value / cost - 1) * 100,
                })
            except (ValueError, TypeError):
                continue

        if portfolio_rows:
            df = pd.DataFrame(portfolio_rows)
            total_value = df["Market value"].sum()
            total_cost = df["Cost basis"].sum()
            total_pl = total_value - total_cost

            c1, c2, c3 = st.columns(3)
            c1.metric("Market value", format_inr(total_value))
            c2.metric("Cost basis", format_inr(total_cost))
            c3.metric("Unrealized P/L", format_inr(total_pl),
                      f"{(total_pl / total_cost * 100):.2f}%")

            df["Allocation (%)"] = df["Market value"] / total_value * 100
            st.dataframe(df, hide_index=True, use_container_width=True)

            fig = go.Figure(go.Pie(
                labels=df["Ticker"],
                values=df["Market value"],
                hole=0.45,
            ))
            fig.update_layout(template="plotly_dark", title="Portfolio allocation")
            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "Download portfolio CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="stocksentry_portfolio.csv",
                mime="text/csv",
            )
        else:
            st.warning(
                "No valid holdings were analyzed. Use lines like "
                "'RELIANCE, 10, 1200' and check that prices are available."
            )

# ----- TAB 7: SMART STOCK SCREENER -----

with tabs[6]:
    st.subheader("🔎 Smart Stock Screener")
    st.write("Filter a selected stock universe by historical price momentum.")

    screener_names = st.multiselect(
        "Choose companies",
        company_names,
        default=[
            "Reliance Industries", "HDFC Bank", "Infosys",
            "Tata Consultancy Services", "Bharti Airtel", "NTPC",
        ],
        key="screener_names",
    )
    min_20d = st.slider(
        "Minimum 20-session return (%)", -30, 30, 0, 5,
        key="screener_min",
    )
    min_1y = st.slider(
        "Minimum 1-year return (%)", -50, 100, 0, 10,
        key="screener_year",
    )

    if st.button("Run stock screener", key="screener_run"):
        screen_rows = []
        for name in screener_names:
            scan_ticker = make_ticker(POPULAR_STOCKS[name], exchange)
            history = load_stock(scan_ticker, "2y")
            if history.empty or len(history) < 21:
                continue
            prices = history["Close"]
            r20 = float(prices.iloc[-1] / prices.iloc[-21] * 100 - 100)
            r1y = (
                float(prices.iloc[-1] / prices.iloc[-252] * 100 - 100)
                if len(prices) >= 252 else np.nan
            )
            daily = prices.pct_change().dropna()
            volatility = float(daily.std() * np.sqrt(252) * 100)

            screen_rows.append({
                "Company": name,
                "Ticker": scan_ticker,
                "Latest price": round(float(prices.iloc[-1]), 2),
                "20-session return (%)": round(r20, 2),
                "1-year return (%)": round(r1y, 2) if pd.notna(r1y) else np.nan,
                "Annualized volatility (%)": round(volatility, 2),
            })

        if screen_rows:
            screen_df = pd.DataFrame(screen_rows)
            passed = screen_df[
                (screen_df["20-session return (%)"] >= min_20d)
                & (screen_df["1-year return (%)"].isna()
                   | (screen_df["1-year return (%)"] >= min_1y))
            ].sort_values("20-session return (%)", ascending=False)

            st.write(f"{len(passed)} stocks met the selected filters.")
            st.dataframe(passed, hide_index=True, use_container_width=True)
            st.download_button(
                "Download screener CSV",
                passed.to_csv(index=False).encode("utf-8"),
                file_name="stocksentry_screener.csv",
                mime="text/csv",
            )
        else:
            st.info("No stock data was available for this scan.")

# ----- TAB 8: AI DECISION EXPLAINER -----

with tabs[7]:
    st.subheader("🧠 AI Decision Explainer")
    st.write(
        "Train the Random Forest on historical features to inspect which "
        "signals it used. Feature importance describes the model, not causation."
    )

    if st.button("Explain model for selected stock", key="explain_run"):
        with st.spinner("Training model..."):
            prediction, scores, error = train_model(stock)

        if error:
            st.warning(error)
        else:
            st.session_state["explain_prediction"] = prediction
            st.session_state["explain_scores"] = scores

    if "explain_prediction" in st.session_state:
        prediction = st.session_state["explain_prediction"]
        model = prediction["model"]
        importances = model.named_steps["forest"].feature_importances_
        importance_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": importances,
        }).sort_values("Importance", ascending=True)

        fig = go.Figure(go.Bar(
            x=importance_df["Importance"],
            y=importance_df["Feature"],
            orientation="h",
        ))
        fig.update_layout(
            template="plotly_dark",
            title="Random Forest feature importance",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.metric(
            "Estimated probability of negative 20-session return",
            f'{prediction["probability"] * 100:.1f}%',
        )

        st.markdown("### Current rule-based context")
        for level, description in get_risk_signals(stock, metrics):
            st.write(f"**{level}:** {description}")

        st.caption(
            "Feature importance is not proof that a factor caused a price move. "
            "Model probabilities can be poorly calibrated and can fail on new data."
        )
        # ----- TAB 9: STRATEGY BACKTESTING -----

with tabs[8]:
    st.subheader("🧪 Strategy Backtesting")
    st.write(
        "Test a simple moving-average strategy against buy-and-hold using "
        "historical data. This is a historical simulation, not a forecast."
    )

    fast_window = st.slider(
        "Fast moving average (sessions)", 5, 100, 20,
        key="backtest_fast",
    )
    slow_window = st.slider(
        "Slow moving average (sessions)", 30, 250, 50,
        key="backtest_slow",
    )
    transaction_cost = st.slider(
        "Estimated cost per position change (%)",
        0.0, 1.0, 0.1, 0.05,
        key="backtest_cost",
    )

    if fast_window >= slow_window:
        st.warning("The fast moving average must be shorter than the slow one.")
    elif st.button("Run backtest", key="backtest_run"):
        prices = stock["Close"].dropna().copy()

        if len(prices) <= slow_window + 5:
            st.warning("Not enough history for these moving-average settings.")
        else:
            fast_ma = prices.rolling(fast_window).mean()
            slow_ma = prices.rolling(slow_window).mean()

            # Position decided using yesterday's completed data.
            position = (fast_ma > slow_ma).astype(float).shift(1).fillna(0)
            daily_returns = prices.pct_change().fillna(0)

            # Approximate trading costs when the position changes.
            turnover = position.diff().abs().fillna(position.abs())
            costs = turnover * (transaction_cost / 100)

            strategy_returns = position * daily_returns - costs
            strategy_equity = (1 + strategy_returns).cumprod()
            benchmark_equity = (1 + daily_returns).cumprod()

            strategy_drawdown = (
                strategy_equity / strategy_equity.cummax() - 1
            )
            benchmark_drawdown = (
                benchmark_equity / benchmark_equity.cummax() - 1
            )

            strategy_total = (strategy_equity.iloc[-1] - 1) * 100
            benchmark_total = (benchmark_equity.iloc[-1] - 1) * 100
            strategy_dd = strategy_drawdown.min() * 100
            benchmark_dd = benchmark_drawdown.min() * 100

            c1, c2 = st.columns(2)
            c1.metric("Strategy return", f"{strategy_total:.2f}%")
            c2.metric("Buy-and-hold return", f"{benchmark_total:.2f}%")

            c3, c4 = st.columns(2)
            c3.metric("Strategy maximum drawdown", f"{strategy_dd:.2f}%")
            c4.metric("Benchmark maximum drawdown", f"{benchmark_dd:.2f}%")

            equity_fig = go.Figure()
            equity_fig.add_trace(go.Scatter(
                x=strategy_equity.index,
                y=strategy_equity * 100,
                name="Moving-average strategy",
            ))
            equity_fig.add_trace(go.Scatter(
                x=benchmark_equity.index,
                y=benchmark_equity * 100,
                name="Buy-and-hold",
            ))
            equity_fig.update_layout(
                template="plotly_dark",
                title="Growth of an initial 100 units",
                yaxis_title="Portfolio value",
                height=430,
                hovermode="x unified",
            )
            st.plotly_chart(equity_fig, use_container_width=True)

            st.download_button(
                "Download backtest results",
                pd.DataFrame({
                    "Price": prices,
                    "Fast MA": fast_ma,
                    "Slow MA": slow_ma,
                    "Position": position,
                    "Strategy daily return": strategy_returns,
                    "Strategy equity": strategy_equity,
                    "Buy-and-hold equity": benchmark_equity,
                }).to_csv().encode("utf-8"),
                file_name="stocksentry_backtest.csv",
                mime="text/csv",
            )

            st.warning(
                "Backtests are sensitive to dates, costs and assumptions. "
                "This simplified simulation excludes taxes, slippage beyond "
                "the configured cost, liquidity constraints and corporate-action "
                "complexities. It is not proof of future profitability."
            )


# ----- TAB 10: PAPER TRADING -----

with tabs[9]:
    st.subheader("🎮 Paper Trading Simulator")
    st.write(
        "Practice with virtual money. No real trades are placed. "
        "Balances and trades in this version are held in the current session."
    )

    if "paper_cash" not in st.session_state:
        st.session_state["paper_cash"] = 100000.0
    if "paper_holdings" not in st.session_state:
        st.session_state["paper_holdings"] = {}
    if "paper_history" not in st.session_state:
        st.session_state["paper_history"] = []

    if st.button("Reset paper portfolio", key="paper_reset"):
        st.session_state["paper_cash"] = 100000.0
        st.session_state["paper_holdings"] = {}
        st.session_state["paper_history"] = []
        st.rerun()

    paper_cash = st.session_state["paper_cash"]
    holdings = st.session_state["paper_holdings"]

    market_value = 0.0
    for holding_ticker, quantity in holdings.items():
        history = load_stock(holding_ticker, "6mo")
        if not history.empty:
            market_value += float(history["Close"].iloc[-1]) * quantity

    total_equity = paper_cash + market_value
    p1, p2, p3 = st.columns(3)
    p1.metric("Virtual cash", format_inr(paper_cash))
    p2.metric("Holdings value", format_inr(market_value))
    p3.metric("Total virtual equity", format_inr(total_equity))

    trade_symbol = st.text_input(
        "Stock ticker to trade",
        value=symbol,
        key="paper_symbol",
    )
    trade_quantity = st.number_input(
        "Quantity", min_value=1, max_value=100000, value=1,
        step=1, key="paper_quantity",
    )

    buy_col, sell_col = st.columns(2)

    with buy_col:
        if st.button("Buy virtual shares", key="paper_buy"):
            paper_ticker = make_ticker(trade_symbol, exchange)
            history = load_stock(paper_ticker, "6mo")

            if history.empty:
                st.error("Could not retrieve a price for this ticker.")
            else:
                price = float(history["Close"].iloc[-1])
                cost = price * trade_quantity

                if cost > st.session_state["paper_cash"]:
                    st.error("Insufficient virtual cash.")
                else:
                    st.session_state["paper_cash"] -= cost
                    current_quantity = st.session_state[
                        "paper_holdings"
                    ].get(paper_ticker, 0)
                    st.session_state["paper_holdings"][
                        paper_ticker
                    ] = current_quantity + trade_quantity

                    st.session_state["paper_history"].append({
                        "Action": "BUY",
                        "Ticker": paper_ticker,
                        "Quantity": trade_quantity,
                        "Price": price,
                        "Value": cost,
                    })
                    st.success(
                        f"Virtual purchase recorded: {trade_quantity} "
                        f"shares at {format_inr(price)}."
                    )
                    st.rerun()

    with sell_col:
        if st.button("Sell virtual shares", key="paper_sell"):
            paper_ticker = make_ticker(trade_symbol, exchange)
            history = load_stock(paper_ticker, "6mo")

            if history.empty:
                st.error("Could not retrieve a price for this ticker.")
            elif st.session_state["paper_holdings"].get(
                paper_ticker, 0
            ) < trade_quantity:
                st.error("You do not own enough virtual shares to sell.")
            else:
                price = float(history["Close"].iloc[-1])
                proceeds = price * trade_quantity
                remaining = (
                    st.session_state["paper_holdings"][paper_ticker]
                    - trade_quantity
                )
                st.session_state["paper_cash"] += proceeds

                if remaining == 0:
                    del st.session_state["paper_holdings"][paper_ticker]
                else:
                    st.session_state["paper_holdings"][
                        paper_ticker
                    ] = remaining

                st.session_state["paper_history"].append({
                    "Action": "SELL",
                    "Ticker": paper_ticker,
                    "Quantity": trade_quantity,
                    "Price": price,
                    "Value": proceeds,
                })
                st.success("Virtual sale recorded.")
                st.rerun()

    if holdings:
        holding_rows = []
        for holding_ticker, quantity in holdings.items():
            history = load_stock(holding_ticker, "6mo")
            if history.empty:
                continue
            price = float(history["Close"].iloc[-1])
            holding_rows.append({
                "Ticker": holding_ticker,
                "Quantity": quantity,
                "Latest price": price,
                "Market value": price * quantity,
            })

        if holding_rows:
            st.markdown("### Current virtual holdings")
            st.dataframe(
                pd.DataFrame(holding_rows),
                hide_index=True,
                use_container_width=True,
            )

    if st.session_state["paper_history"]:
        st.markdown("### Transaction history")
        trade_df = pd.DataFrame(st.session_state["paper_history"])
        st.dataframe(trade_df, hide_index=True, use_container_width=True)

        st.download_button(
            "Download trade history",
            trade_df.to_csv(index=False).encode("utf-8"),
            file_name="stocksentry_paper_trades.csv",
            mime="text/csv",
        )


# ----- TAB 11: NEWS & SENTIMENT -----

with tabs[10]:
    st.subheader("📰 News & Sentiment Analyzer")
    st.write(
        "Retrieve headlines available through Yahoo Finance and apply a "
        "simple keyword-based sentiment estimate. This is not a trained "
        "financial language model."
    )

    news_ticker = st.text_input(
        "Ticker for news",
        value=ticker,
        key="news_ticker",
    )

    positive_words = {
        "growth", "profit", "surge", "beat", "strong", "upgrade",
        "record", "gain", "positive", "expansion", "rally", "rise",
        "improve", "partnership", "approval",
    }
    negative_words = {
        "loss", "decline", "fall", "drop", "weak", "downgrade",
        "fraud", "probe", "debt", "warning", "crash", "miss",
        "negative", "risk", "lawsuit", "cut",
    }

    if st.button("Fetch financial news", key="news_fetch"):
        normalized_ticker = make_ticker(news_ticker, exchange)

        try:
            news_items = yf.Ticker(normalized_ticker).news or []
        except Exception:
            news_items = []

        news_rows = []

        for item in news_items:
            content = item.get("content", item)
            title_text = content.get("title", "")
            summary = content.get("summary", "")
            publisher = content.get("publisher", "")

            if not title_text:
                continue

            text_blob = (title_text + " " + summary).lower()
            positive_count = sum(
                word in text_blob for word in positive_words
            )
            negative_count = sum(
                word in text_blob for word in negative_words
            )

            if positive_count > negative_count:
                sentiment = "Positive"
            elif negative_count > positive_count:
                sentiment = "Negative"
            else:
                sentiment = "Neutral / unclear"

            canonical = content.get("canonicalUrl", {})
            if isinstance(canonical, dict):
                article_url = canonical.get("url", "")
            else:
                article_url = str(canonical or "")

            provider = content.get("provider", {})
            if isinstance(provider, dict):
                publisher = provider.get("displayName", publisher)

            news_rows.append({
                "Headline": title_text,
                "Publisher": publisher,
                "Sentiment estimate": sentiment,
                "URL": article_url,
            })

        if news_rows:
            news_df = pd.DataFrame(news_rows)
            st.dataframe(
                news_df, hide_index=True, use_container_width=True
            )

            for _, row in news_df.iterrows():
                st.markdown(f"**{row['Sentiment estimate']}** — {row['Headline']}")
                if row["Publisher"]:
                    st.caption(str(row["Publisher"]))
                if row["URL"]:
                    st.markdown(f"[Read original article]({row['URL']})")

            st.download_button(
                "Download news sentiment CSV",
                news_df.to_csv(index=False).encode("utf-8"),
                file_name="stocksentry_news.csv",
                mime="text/csv",
            )
        else:
            st.info(
                "No headlines were returned. News availability depends on "
                "Yahoo Finance and may vary by ticker."
            )

    st.caption(
        "Keyword-based sentiment can misunderstand context, sarcasm, "
        "negation and financial terminology. Always read the original article."
    )


# ----- TAB 12: WATCHLIST & ALERTS -----

with tabs[11]:
    st.subheader("🔔 Watchlist & Price Alerts")
    st.write(
        "Add tickers to a session watchlist and check price thresholds "
        "when you run a check. This version does not send background notifications."
    )

    if "stock_watchlist" not in st.session_state:
        st.session_state["stock_watchlist"] = []

    if "stock_alerts" not in st.session_state:
        st.session_state["stock_alerts"] = {}

    watch_symbol = st.text_input(
        "Ticker to add",
        value=symbol,
        key="watch_symbol",
    )

    if st.button("Add to watchlist", key="watch_add"):
        watch_ticker = make_ticker(watch_symbol, exchange)
        if watch_ticker not in st.session_state["stock_watchlist"]:
            st.session_state["stock_watchlist"].append(watch_ticker)
        st.success(f"{watch_ticker} added to this session's watchlist.")

    watchlist = st.session_state["stock_watchlist"]

    if watchlist:
        st.markdown("### Watchlist")
        watch_rows = []

        for watch_ticker in watchlist:
            history = load_stock(watch_ticker, "6mo")
            if history.empty:
                watch_rows.append({
                    "Ticker": watch_ticker,
                    "Latest price": np.nan,
                    "20-session return (%)": np.nan,
                    "Status": "Price unavailable",
                })
            else:
                prices = history["Close"]
                r20 = (
                    float(prices.iloc[-1] / prices.iloc[-21] * 100 - 100)
                    if len(prices) >= 21 else np.nan
                )
                watch_rows.append({
                    "Ticker": watch_ticker,
                    "Latest price": float(prices.iloc[-1]),
                    "20-session return (%)": r20,
                    "Status": "Loaded",
                })

        st.dataframe(
            pd.DataFrame(watch_rows),
            hide_index=True,
            use_container_width=True,
        )

        remove_ticker = st.selectbox(
            "Remove a ticker",
            ["None"] + watchlist,
            key="watch_remove_select",
        )
        if st.button("Remove selected ticker", key="watch_remove"):
            if remove_ticker != "None":
                st.session_state["stock_watchlist"].remove(remove_ticker)
                st.session_state["stock_alerts"].pop(remove_ticker, None)
                st.rerun()

        st.markdown("### Price thresholds")
        alert_ticker = st.selectbox(
            "Select watchlist ticker",
            watchlist,
            key="alert_ticker",
        )
        target_price = st.number_input(
            "Alert when price reaches or exceeds (₹)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            key="alert_target",
        )

        if st.button("Save price threshold", key="alert_save"):
            if target_price > 0:
                st.session_state["stock_alerts"][alert_ticker] = target_price
                st.success("Threshold saved for this session.")
            else:
                st.warning("Enter a target price greater than zero.")

        if st.button("Check price alerts now", key="alert_check"):
            alerts = st.session_state["stock_alerts"]

            if not alerts:
                st.info("No price thresholds have been saved.")
            else:
                for alert_symbol, target in alerts.items():
                    history = load_stock(alert_symbol, "6mo")

                    if history.empty:
                        st.warning(f"{alert_symbol}: price unavailable.")
                        continue

                    latest_price = float(history["Close"].iloc[-1])

                    if latest_price >= target:
                        st.success(
                            f"Threshold reached: {alert_symbol} is "
                            f"{format_inr(latest_price)}, target "
                            f"{format_inr(target)}."
                        )
                    else:
                        st.info(
                            f"{alert_symbol}: current price "
                            f"{format_inr(latest_price)}; target "
                            f"{format_inr(target)}."
                        )

        st.download_button(
            "Download watchlist",
            pd.DataFrame({"Ticker": watchlist}).to_csv(
                index=False
            ).encode("utf-8"),
            file_name="stocksentry_watchlist.csv",
            mime="text/csv",
        )
    else:
        st.info("Your watchlist is empty. Add a ticker above.")


# ==================== APP FOOTER ====================

st.divider()
st.caption(
    "StockSentry AI • Educational prototype • "
    "Verify data and model performance independently. "
    "Not financial advice."
)
