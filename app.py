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
.stApp {
    background-color: #0b1120;
}
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
    "Educational research prototype. Model estimates are uncertain "
    "and are not financial advice. Historical market data may be "
    "delayed, incomplete, or unavailable."
)

# ---------------- DATA ----------------

@st.cache_data(ttl=900, show_spinner=False)
def load_stock(ticker, period):
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


def make_features(data):
    df = pd.DataFrame(index=data.index)
    close = data["Close"]
    volume = data["Volume"].replace(0, np.nan)

    df["return_1d"] = close.pct_change()
    df["return_5d"] = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)
    df["volatility_20d"] = df["return_1d"].rolling(20).std()
    df["volatility_60d"] = df["return_1d"].rolling(60).std()
    df["ma_ratio_20"] = close / close.rolling(20).mean() - 1
    df["ma_ratio_50"] = close / close.rolling(50).mean() - 1
    df["ma_ratio_200"] = close / close.rolling(200).mean() - 1
    df["drawdown_60d"] = close / close.rolling(60).max() - 1
    df["volume_ratio_20d"] = volume / volume.rolling(20).mean()

    return df.replace([np.inf, -np.inf], np.nan)


FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "volatility_60d",
    "ma_ratio_20",
    "ma_ratio_50",
    "ma_ratio_200",
    "drawdown_60d",
    "volume_ratio_20d",
]


def train_model(data):
    features = make_features(data)

    # Label: did the price fall over the following 20 sessions?
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
        return None, None, "Insufficient target-class variation for evaluation."

    model = Pipeline([
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

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "Test observations": len(y_test),
        "Test start": str(y_test.index.min().date()),
        "Test end": str(y_test.index.max().date()),
    }

    # Refit on all labelled history for the latest estimate.
    final_model = Pipeline([
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
    final_model.fit(X, y)

    latest = features[FEATURES].iloc[[-1]]

    if latest.isna().all(axis=None):
        return None, metrics, "Latest indicators are unavailable."

    probability = final_model.predict_proba(latest)[0, 1]

    return {
        "probability": float(probability),
        "model": final_model,
    }, metrics, None


def calculate_metrics(data):
    close = data["Close"]
    returns = close.pct_change().dropna()

    period_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
    volatility = (
        returns.std() * np.sqrt(252) * 100
        if len(returns) > 1 else np.nan
    )
    drawdown = (close / close.cummax() - 1) * 100

    return {
        "price": float(close.iloc[-1]),
        "return": float(period_return),
        "volatility": float(volatility),
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
        signals.append(("High", "Annualized historical volatility exceeds 45%."))

    if metrics["max_drawdown"] < -30:
        signals.append(("High", "Historical maximum drawdown exceeds 30%."))

    if not signals:
        signals.append((
            "No configured warning triggered",
            "This does not mean the stock is safe or a good investment."
        ))

    return signals


def format_inr(value):
    return f"₹{value:,.2f}"


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("Analysis settings")

    exchange = st.selectbox("Exchange", ["NSE", "BSE"])
    default_symbol = "RELIANCE" if exchange == "NSE" else "500325"

    symbol = st.text_input(
        "Stock symbol",
        value=default_symbol,
        help="Examples: RELIANCE, TCS, INFY, HDFCBANK",
    ).strip().upper()

    period = st.selectbox(
        "Historical period",
        ["1y", "2y", "5y", "10y"],
        index=2,
    )

    compare_symbols = st.text_input(
        "Compare stocks",
        value="TCS, INFY",
        help="Enter comma-separated symbols.",
    )

    st.caption("NSE symbols use .NS; BSE codes use .BO.")

suffix = ".NS" if exchange == "NSE" else ".BO"
ticker = (
    symbol if symbol.endswith((".NS", ".BO"))
    else symbol + suffix
)

with st.spinner(f"Downloading historical data for {ticker}..."):
    data = load_stock(ticker, period)

if data.empty or len(data) < 30:
    st.error(
        f"No usable data found for {ticker}. "
        "Check the symbol or try another stock."
    )
    st.stop()

metrics = calculate_metrics(data)
close = data["Close"]
latest_date = data.index[-1].date()

st.caption(
    f"Ticker: {ticker} • Latest downloaded observation: {latest_date}"
)

# ---------------- SUMMARY ----------------

c1, c2, c3, c4 = st.columns(4)

c1.metric("Latest adjusted price", format_inr(metrics["price"]))
c2.metric("Period return", f"{metrics['return']:+.2f}%")
c3.metric(
    "Annualized volatility",
    f"{metrics['volatility']:.2f}%"
    if pd.notna(metrics["volatility"]) else "N/A",
)
c4.metric("Maximum drawdown", f"{metrics['max_drawdown']:.2f}%")

st.caption(
    "The latest downloaded observation may not be a live market quote."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Market Overview",
    "🛡️ Risk Scanner",
    "🧠 ML Lab",
    "⚖️ Compare Stocks",
])

# ---------------- OVERVIEW ----------------

with tab1:
    st.subheader(f"{ticker} price history")

    chart = go.Figure()

    chart.add_trace(go.Scatter(
        x=data.index,
        y=close,
        name="Adjusted close",
        line=dict(width=2),
    ))

    chart.add_trace(go.Scatter(
        x=data.index,
        y=close.rolling(50).mean(),
        name="50-day average",
    ))

    chart.add_trace(go.Scatter(
        x=data.index,
        y=close.rolling(200).mean(),
        name="200-day average",
    ))

    chart.update_layout(
        template="plotly_dark",
        height=450,
        yaxis_title="Price (INR)",
        hovermode="x unified",
    )

    st.plotly_chart(chart, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Daily returns")

        returns = close.pct_change() * 100

        fig = go.Figure(go.Scatter(
            x=returns.index,
            y=returns,
            name="Daily return",
        ))

        fig.update_layout(
            template="plotly_dark",
            height=320,
            yaxis_title="Return (%)",
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Trading volume")

        fig = go.Figure(go.Bar(
            x=data.index,
            y=data["Volume"],
            name="Volume",
        ))

        fig.update_layout(
            template="plotly_dark",
            height=320,
            yaxis_title="Shares traded",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download historical data (CSV)",
        data.to_csv().encode("utf-8"),
        file_name=f"{ticker.replace('.', '_')}_history.csv",
        mime="text/csv",
    )

# ---------------- RISK SCANNER ----------------

with tab2:
    st.subheader("Risk and stagnation indicators")

    signals = get_risk_signals(data, metrics)

    for severity, message in signals:
        if severity == "High":
            st.error(f"**{severity} risk signal:** {message}")
        elif severity == "Moderate":
            st.warning(f"**{severity} risk signal:** {message}")
        else:
            st.info(message)

    st.markdown("### Recent price performance")

    windows = []

    for days in [5, 20, 60, 120, 252]:
        if len(close) > days:
            change = (
                close.iloc[-1] / close.iloc[-days - 1] - 1
            ) * 100

            windows.append({
                "Trading sessions": days,
                "Price return (%)": round(float(change), 2),
            })

    if windows:
        st.dataframe(
            pd.DataFrame(windows),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("### Risk report")

    report_rows = [
        {"Indicator": "Ticker", "Value": ticker},
        {"Indicator": "Last observation", "Value": str(latest_date)},
        {"Indicator": "Latest adjusted price", "Value": metrics["price"]},
        {"Indicator": "Period return (%)", "Value": metrics["return"]},
        {"Indicator": "Annualized volatility (%)", "Value": metrics["volatility"]},
        {"Indicator": "Maximum drawdown (%)", "Value": metrics["max_drawdown"]},
        {"Indicator": "50-day average", "Value": metrics["ma50"]},
        {"Indicator": "200-day average", "Value": metrics["ma200"]},
    ]

    for i, (severity, message) in enumerate(signals, start=1):
        report_rows.append({
            "Indicator": f"Risk signal {i}",
            "Value": f"{severity}: {message}",
        })

    report = pd.DataFrame(report_rows)

    st.download_button(
        "Download investigation report (CSV)",
        report.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker.replace('.', '_')}_risk_report.csv",
        mime="text/csv",
    )

# ---------------- MACHINE LEARNING ----------------

with tab3:
    st.subheader("20-session downside-risk classifier")

    st.write(
        "The model estimates whether the next 20 trading sessions "
        "will have a negative return. It does not predict a guaranteed "
        "price or tell you what to buy."
    )

    st.warning(
        "This is preliminary historical testing. Overlapping future "
        "return labels can make nearby observations dependent. "
        "Metrics are not evidence of trading profitability."
    )

    if st.button("Train and evaluate ML model", type="primary"):
        with st.spinner("Training Random Forest..."):
            result, scores, error = train_model(data)

        if error:
            st.error(error)

        else:
            a, b, c = st.columns(3)

            a.metric(
                "Estimated downside probability",
                f"{result['probability'] * 100:.1f}%",
            )

            b.metric("Test accuracy", f"{scores['Accuracy'] * 100:.1f}%")
            c.metric("Test ROC-AUC", f"{scores['ROC-AUC']:.3f}")

            d, e, f = st.columns(3)

            d.metric("Test precision", f"{scores['Precision']:.3f}")
            e.metric("Test recall", f"{scores['Recall']:.3f}")
            f.metric("Test observations", scores["Test observations"])

            st.caption(
                f"Chronological test period: {scores['Test start']} "
                f"to {scores['Test end']}."
            )

            forest = result["model"].named_steps["forest"]

            importance = pd.DataFrame({
                "Feature": FEATURES,
                "Importance": forest.feature_importances_,
            }).sort_values("Importance", ascending=True)

            fig = go.Figure(go.Bar(
                x=importance["Importance"],
                y=importance["Feature"],
                orientation="h",
            ))

            fig.update_layout(
                template="plotly_dark",
                title="Model feature importance",
                height=420,
            )

            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "Feature importance shows how the model used its inputs; "
                "it does not establish causation."
            )

            st.session_state["ml_report"] = {
                "Ticker": ticker,
                "Downside probability": result["probability"],
                **scores,
            }

    if "ml_report" in st.session_state:
        ml_report = st.session_state["ml_report"]

        if ml_report["Ticker"] == ticker:
            st.download_button(
                "Download ML evaluation (CSV)",
                pd.DataFrame([ml_report]).to_csv(index=False).encode("utf-8"),
                file_name=f"{ticker.replace('.', '_')}_ml_report.csv",
                mime="text/csv",
            )

# ---------------- STOCK COMPARISON ----------------

with tab4:
    st.subheader("Compare Indian stocks")

    symbols = list(dict.fromkeys([
        s.strip().upper()
        for s in compare_symbols.split(",")
        if s.strip()
    ]))

    comparison_rows = []
    comparison_chart = go.Figure()

    with st.spinner("Loading comparison data..."):
        for item in symbols[:6]:
            comp_ticker = (
                item if item.endswith((".NS", ".BO"))
                else item + suffix
            )

            comp = load_stock(comp_ticker, period)

            if comp.empty or len(comp) < 30:
                continue

            comp_close = comp["Close"]
            comp_return = (
                comp_close.iloc[-1] / comp_close.iloc[0] - 1
            ) * 100

            daily = comp_close.pct_change().dropna()
            comp_vol = (
                daily.std() * np.sqrt(252) * 100
                if len(daily) > 1 else np.nan
            )

            comp_drawdown = (
                comp_close / comp_close.cummax() - 1
            ).min() * 100

            comparison_rows.append({
                "Ticker": comp_ticker,
                "Latest price": round(float(comp_close.iloc[-1]), 2),
                "Period return (%)": round(float(comp_return), 2),
                "Annualized volatility (%)": round(float(comp_vol), 2),
                "Maximum drawdown (%)": round(float(comp_drawdown), 2),
            })

            normalized = comp_close / comp_close.iloc[0] * 100

            comparison_chart.add_trace(go.Scatter(
                x=normalized.index,
                y=normalized,
                name=comp_ticker,
            ))

    main_normalized = close / close.iloc[0] * 100

    comparison_chart.add_trace(go.Scatter(
        x=main_normalized.index,
        y=main_normalized,
        name=ticker,
        line=dict(width=3),
    ))

    comparison_chart.update_layout(
        template="plotly_dark",
        title="Relative performance (starting value = 100)",
        yaxis_title="Indexed price",
        height=430,
        hovermode="x unified",
    )

    st.plotly_chart(comparison_chart, use_container_width=True)

    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)

        st.dataframe(
            comparison_df,
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download comparison (CSV)",
            comparison_df.to_csv(index=False).encode("utf-8"),
            file_name="stocksentry_comparison.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "No comparison data was available. Try NSE symbols such as "
            "TCS, INFY or RELIANCE."
        )

st.divider()

st.caption(
    "StockSentry AI • Educational prototype • "
    "Verify market data and model performance independently."
)
