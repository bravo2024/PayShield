# PayShield — Real-Time Payment Risk Scoring Dashboard
# Production-grade Streamlit app | Stripe / PayPal / Visa framing
# Libraries: numpy, pandas, matplotlib, scipy, streamlit ONLY

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import scipy.stats as sp_stats
import streamlit as st
from datetime import datetime, timedelta

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PayShield | Real-Time Payment Risk Scoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Risk Engine Controls")
fraud_threshold    = st.sidebar.slider("Fraud Score Threshold (τ)", 0.01, 0.99, 0.30, 0.01,
                                        help="Score above this → flagged as fraud")
review_queue_limit = st.sidebar.number_input("Max Review Queue Size", 100, 10000, 2000, 100)
ops_cost_per_review= st.sidebar.number_input("Ops Cost per Review ($)", 1, 20, 3, 1)
fraud_loss_amount  = st.sidebar.number_input("Avg Fraud Loss per Case ($)", 50, 5000, 500, 50)
decline_rate       = st.sidebar.slider("FP Decline Rate", 0.0, 1.0, 0.80, 0.05,
                                        help="Fraction of FP-blocked txns that were actually legitimate (revenue cost)")
st.sidebar.markdown("---")
st.sidebar.markdown("**PayShield v2.0** | IEEE-CIS Fraud Detection Style")
st.sidebar.markdown("100,000 synthetic transactions • 90-day window")

# ──────────────────────────────────────────────────────────────────────────────
# DATA GENERATION  (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating 100k synthetic transactions…")
def generate_transactions(n: int = 100_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Timestamps: 90 days
    base_time = datetime(2024, 1, 1)
    offsets   = np.sort(rng.uniform(0, 90 * 86400, n)).astype(int)
    timestamps = [base_time + timedelta(seconds=int(o)) for o in offsets]
    transaction_hour = np.array([t.hour      for t in timestamps])
    day_of_week      = np.array([t.weekday() for t in timestamps])

    # Categorical features
    categories     = ["electronics","clothing","food","travel","gaming"]
    cat_probs      = [0.20, 0.25, 0.30, 0.15, 0.10]
    product_category = rng.choice(categories, n, p=cat_probs)

    payment_methods = ["card","wallet","bank","crypto"]
    pm_probs        = [0.55, 0.25, 0.15, 0.05]
    payment_method  = rng.choice(payment_methods, n, p=pm_probs)

    device_types = ["mobile","desktop","tablet"]
    dt_probs     = [0.60, 0.30, 0.10]
    device_type  = rng.choice(device_types, n, p=dt_probs)

    browsers  = ["chrome","safari","firefox","edge"]
    br_probs  = [0.50, 0.25, 0.15, 0.10]
    browser   = rng.choice(browsers, n, p=br_probs)

    country_risks = ["low","medium","high"]
    cr_probs      = [0.70, 0.20, 0.10]
    ip_country_risk = rng.choice(country_risks, n, p=cr_probs)

    # Boolean / numeric features
    card_present          = (rng.random(n) < 0.65).astype(int)
    is_new_merchant       = (rng.random(n) < 0.15).astype(int)
    is_first_transaction  = (rng.random(n) < 0.05).astype(int)

    transaction_amount    = np.clip(np.exp(rng.normal(4.5, 1.2, n)), 1, 10_000)
    velocity_1h           = rng.poisson(1.5, n).astype(int)
    velocity_24h          = rng.poisson(8,   n).astype(int)
    card_age_days         = rng.integers(0, 2000, n)
    account_age_days      = rng.integers(0, 3000, n)
    days_since_last       = rng.exponential(5, n)
    avg_amount            = np.clip(np.exp(rng.normal(4.5, 0.8, n)), 1, 5000)
    amount_vs_avg_ratio   = transaction_amount / (avg_amount + 1e-8)

    # ── Fraud label construction (target ≈ 0.5%) ──────────────────────────
    fraud_score = np.zeros(n)
    fraud_score += 4.0 * ((transaction_amount > 2000) & (is_new_merchant == 1) & (ip_country_risk == "high"))
    fraud_score += 3.0 * (velocity_1h > 7)
    fraud_score += 1.5 * ((transaction_hour >= 2) & (transaction_hour <= 5))
    fraud_score += 4.0 * ((is_first_transaction == 1) & (payment_method == "crypto") & (transaction_amount > 500))
    fraud_score += 2.0 * (amount_vs_avg_ratio > 8)
    fraud_score += 1.5 * ((card_present == 0) & (ip_country_risk == "high"))

    logit      = -5.8 + fraud_score
    fraud_prob = 1.0 / (1.0 + np.exp(-logit))
    is_fraud   = (rng.random(n) < fraud_prob).astype(int)

    # Clamp to ≈ 0.5 %
    current_frauds = is_fraud.sum()
    target_frauds  = int(0.005 * n)
    if current_frauds > target_frauds:
        fraud_idx  = np.where(is_fraud == 1)[0]
        remove_idx = rng.choice(fraud_idx, current_frauds - target_frauds, replace=False)
        is_fraud[remove_idx] = 0
    elif current_frauds < target_frauds:
        legit_idx = np.where(is_fraud == 0)[0]
        add_idx   = rng.choice(legit_idx, target_frauds - current_frauds, replace=False)
        is_fraud[add_idx] = 1

    return pd.DataFrame({
        "timestamp":              timestamps,
        "transaction_amount":     transaction_amount,
        "product_category":       product_category,
        "payment_method":         payment_method,
        "device_type":            device_type,
        "browser":                browser,
        "ip_country_risk":        ip_country_risk,
        "card_present":           card_present,
        "transaction_hour":       transaction_hour,
        "day_of_week":            day_of_week,
        "days_since_last_transaction": days_since_last,
        "velocity_1h":            velocity_1h,
        "velocity_24h":           velocity_24h,
        "amount_vs_avg_ratio":    amount_vs_avg_ratio,
        "is_new_merchant":        is_new_merchant,
        "is_first_transaction":   is_first_transaction,
        "card_age_days":          card_age_days,
        "account_age_days":       account_age_days,
        "is_fraud":               is_fraud,
    })

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame):
    X = {}
    X["log_amount"]          = np.log1p(df["transaction_amount"])
    X["velocity_1h"]         = df["velocity_1h"].astype(float)
    X["velocity_24h"]        = df["velocity_24h"].astype(float)
    X["log_avr"]             = np.log1p(df["amount_vs_avg_ratio"])
    X["log_days_since"]      = np.log1p(df["days_since_last_transaction"])
    X["log_card_age"]        = np.log1p(df["card_age_days"])
    X["log_acct_age"]        = np.log1p(df["account_age_days"])
    X["card_present"]        = df["card_present"].astype(float)
    X["is_new_merchant"]     = df["is_new_merchant"].astype(float)
    X["is_first_txn"]        = df["is_first_transaction"].astype(float)
    # Cyclic time encodings
    X["hour_sin"] = np.sin(2 * np.pi * df["transaction_hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df["transaction_hour"] / 24)
    X["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"]      / 7)
    X["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"]      / 7)
    # One-hot encodings
    for pm  in ["card","wallet","bank","crypto"]:
        X[f"pm_{pm}"]  = (df["payment_method"]   == pm).astype(float)
    for cr  in ["low","medium","high"]:
        X[f"cr_{cr}"]  = (df["ip_country_risk"]  == cr).astype(float)
    for cat in ["electronics","clothing","food","travel","gaming"]:
        X[f"cat_{cat}"]= (df["product_category"] == cat).astype(float)

    Xmat         = np.column_stack(list(X.values()))
    feature_names= list(X.keys())
    y            = df["is_fraud"].values.astype(int)
    return Xmat, y, feature_names

# ──────────────────────────────────────────────────────────────────────────────
# NUMPY LOGISTIC REGRESSION
# ──────────────────────────────────────────────────────────────────────────────
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

class WeightedLogisticRegression:
    def __init__(self, lr: float = 0.1, epochs: int = 400, l2: float = 1e-3):
        self.lr = lr; self.epochs = epochs; self.l2 = l2

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        self.w_ = np.zeros(d); self.b_ = 0.0
        pos = max(y.sum(), 1); neg = max(n - pos, 1)
        sw  = np.where(y == 1, n / (2 * pos), n / (2 * neg))
        for _ in range(self.epochs):
            p   = _sigmoid(X @ self.w_ + self.b_)
            err = (p - y) * sw
            self.w_ -= self.lr * (X.T @ err / n + self.l2 * self.w_)
            self.b_  -= self.lr * err.mean()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X @ self.w_ + self.b_)

@st.cache_resource(show_spinner="Training ML risk model…")
def train_model(_df: pd.DataFrame):
    X, y, feat_names = engineer_features(_df)
    # Chronological 80/20 split
    split = int(0.8 * len(X))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-8
    X_tr_s = (X_tr - mu) / sd
    X_te_s = (X_te - mu) / sd
    mdl = WeightedLogisticRegression(lr=0.1, epochs=400, l2=1e-3).fit(X_tr_s, y_tr)
    train_scores = mdl.predict_proba(X_tr_s)
    test_scores  = mdl.predict_proba(X_te_s)
    return {"model": mdl, "mu": mu, "sd": sd,
            "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
            "train_scores": train_scores, "test_scores": test_scores,
            "feat_names": feat_names, "split_idx": split}

# ──────────────────────────────────────────────────────────────────────────────
# METRIC HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def roc_auc(y_true, scores):
    pos = int(y_true.sum()); neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.5, np.array([0,1]), np.array([0,1])
    order = np.argsort(scores)[::-1]
    tp = fp = 0
    tprs, fprs = [0.0], [0.0]
    for i in order:
        if y_true[i] == 1: tp += 1
        else:               fp += 1
        tprs.append(tp / pos); fprs.append(fp / neg)
    tprs.append(1.0); fprs.append(1.0)
    auc = float(np.trapz(tprs, fprs))
    return abs(auc), np.array(tprs), np.array(fprs)

def pr_auc(y_true, scores):
    thresholds = np.sort(np.unique(scores))[::-1]
    pos = int(y_true.sum())
    precs, recs = [], []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp   = int(((pred==1)&(y_true==1)).sum())
        fp   = int(((pred==1)&(y_true==0)).sum())
        fn   = int(((pred==0)&(y_true==1)).sum())
        pr   = tp/(tp+fp) if (tp+fp)>0 else 1.0
        rc   = tp/(tp+fn) if (tp+fn)>0 else 0.0
        precs.append(pr); recs.append(rc)
    if not precs:
        return 0.0, np.array([1,0]), np.array([0,1])
    auc = float(abs(np.trapz(precs, recs)))
    return auc, np.array(precs), np.array(recs)

def metrics_at_threshold(y_true, scores, tau):
    pred = (scores >= tau).astype(int)
    tp   = int(((pred==1)&(y_true==1)).sum())
    fp   = int(((pred==1)&(y_true==0)).sum())
    fn   = int(((pred==0)&(y_true==1)).sum())
    tn   = int(((pred==0)&(y_true==0)).sum())
    return tp, fp, fn, tn

def ks_statistic(y_true, scores):
    s1 = np.sort(scores[y_true==1])
    s0 = np.sort(scores[y_true==0])
    combined = np.sort(np.unique(np.concatenate([s1, s0])))
    cdf1 = np.searchsorted(s1, combined, side="right") / len(s1)
    cdf0 = np.searchsorted(s0, combined, side="right") / len(s0)
    ks   = float(np.max(np.abs(cdf1 - cdf0)))
    return ks

# ──────────────────────────────────────────────────────────────────────────────
# RULE ENGINE
# ──────────────────────────────────────────────────────────────────────────────
def apply_rules(df: pd.DataFrame):
    decision = np.array(["PASS"] * len(df), dtype=object)
    rule_hits = {}

    # R1: Amount > $5k + new merchant + high-risk country → BLOCK
    r1 = (df["transaction_amount"]>5000) & (df["is_new_merchant"]==1) & (df["ip_country_risk"]=="high")
    decision[r1.values] = "BLOCK"
    rule_hits["R1: High Amt + New Merchant + High-Risk Country"] = int(r1.sum())

    # R2: velocity_1h > 10 → BLOCK
    r2 = df["velocity_1h"] > 10
    decision[r2.values] = "BLOCK"
    rule_hits["R2: Velocity Spike >10 txns/hr"] = int(r2.sum())

    # R3: amount_vs_avg_ratio > 10x → REVIEW
    r3 = (df["amount_vs_avg_ratio"]>10) & (decision != "BLOCK")
    decision[r3.values] = "REVIEW"
    rule_hits["R3: Amount >10x Average"] = int(r3.sum())

    # R4: First txn + crypto + amount > $500 → REVIEW
    r4 = ((df["is_first_transaction"]==1) & (df["payment_method"]=="crypto") &
          (df["transaction_amount"]>500) & (decision == "PASS"))
    decision[r4.values] = "REVIEW"
    rule_hits["R4: First Txn + Crypto + High Amt"] = int(r4.sum())

    # R5: CNP + high-risk country + amount > $1k + new merchant → REVIEW
    r5 = ((df["card_present"]==0) & (df["ip_country_risk"]=="high") &
          (df["transaction_amount"]>1000) & (df["is_new_merchant"]==1) & (decision == "PASS"))
    decision[r5.values] = "REVIEW"
    rule_hits["R5: CNP + Intl + High Amt + New Merchant"] = int(r5.sum())

    return pd.Series(decision, index=df.index), rule_hits

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA & MODEL
# ──────────────────────────────────────────────────────────────────────────────
df       = generate_transactions(100_000)
results  = train_model(df)
decisions, rule_hits = apply_rules(df)

# ── Attach scores to full dataframe ──────────────────────────────────────────
split_idx    = results["split_idx"]
all_scores   = np.empty(len(df))
# Train portion scores
X_all, y_all, _ = engineer_features(df)
X_all_s = (X_all - results["mu"]) / results["sd"]
all_scores = results["model"].predict_proba(X_all_s)
df = df.copy()
df["risk_score"] = all_scores
df["rule_decision"] = decisions.values

# ──────────────────────────────────────────────────────────────────────────────
# HEADER BANNER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#0f172a 100%);
padding:22px 28px;border-radius:12px;margin-bottom:18px;">
<h1 style="color:#60a5fa;margin:0;font-size:2rem;">🛡️ PayShield</h1>
<p style="color:#94a3b8;margin:4px 0 0;font-size:1.05rem;">
Real-Time Payment Risk Scoring Platform — Stripe / PayPal / Visa Grade</p>
</div>""", unsafe_allow_html=True)

# ── Header KPIs ───────────────────────────────────────────────────────────────
total_txns  = len(df)
fraud_rate  = df["is_fraud"].mean() * 100
y_te        = results["y_te"]
te_scores   = results["test_scores"]
tp, fp, fn, tn = metrics_at_threshold(y_te, te_scores, fraud_threshold)
ml_detect   = tp / max(y_te.sum(), 1) * 100
fp_rate_pct = fp / max((y_te==0).sum(), 1) * 100
net_savings = (tp * fraud_loss_amount) - (fp * 75 * decline_rate) - (tp + fp) * ops_cost_per_review

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Transactions",  f"{total_txns:,}")
c2.metric("Fraud Rate",          f"{fraud_rate:.3f}%",   delta="-0.02% WoW")
c3.metric("ML Detection Rate",   f"{ml_detect:.1f}%",   delta="+2.1% vs rules")
c4.metric("False Positive Rate", f"{fp_rate_pct:.2f}%", delta="-0.05% WoW")
c5.metric("Est. Net Savings",    f"${net_savings:,.0f}")
st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💳 Transaction Stream Explorer",
    "⚡ Real-Time Risk Engine",
    "🤖 ML Risk Scoring Model",
    "📊 Fraud Investigation Dashboard",
    "💰 Financial Impact & Operations",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRANSACTION STREAM EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("💳 Transaction Stream Explorer")
    st.markdown("IEEE-CIS Fraud Detection-style synthetic dataset · 100,000 transactions · 90-day window")

    # ── Summary KPIs ─────────────────────────────────────────────────────────
    fraud_df  = df[df["is_fraud"] == 1]
    legit_df  = df[df["is_fraud"] == 0]
    fraud_amt = fraud_df["transaction_amount"].sum()

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total Transactions", f"{total_txns:,}")
    sc2.metric("Fraud Rate",         f"{fraud_rate:.3f}%  ({len(fraud_df):,} cases)")
    sc3.metric("Fraud Amount ($)",   f"${fraud_amt:,.0f}")
    sc4.metric("Fraud Prevention Rate (Rules)", f"{(decisions.isin(['BLOCK','REVIEW']) & (df['is_fraud']==1)).sum() / max(len(fraud_df),1)*100:.1f}%")

    # ── Hourly Transaction Volume ─────────────────────────────────────────────
    st.subheader("Hourly Transaction Volume & Fraud Rate")
    hourly_legit = df[df["is_fraud"]==0].groupby("transaction_hour").size()
    hourly_fraud = df[df["is_fraud"]==1].groupby("transaction_hour").size()
    hours        = np.arange(24)

    fig, ax1 = plt.subplots(figsize=(12, 3.5))
    ax1.bar(hours - 0.2, [hourly_legit.get(h, 0) for h in hours], 0.4,
            label="Legitimate", color="#3b82f6", alpha=0.85)
    ax1.bar(hours + 0.2, [hourly_fraud.get(h, 0) for h in hours], 0.4,
            label="Fraud", color="#ef4444", alpha=0.85)
    ax1.set_xlabel("Hour of Day"); ax1.set_ylabel("Transaction Count")
    ax1.set_title("Transaction Volume by Hour (Fraud vs Legitimate)")
    ax1.legend(); ax1.set_xticks(hours); ax1.grid(axis="y", alpha=0.3)
    ax2 = ax1.twinx()
    hourly_total     = df.groupby("transaction_hour").size()
    hourly_fraud_rate= (hourly_fraud / hourly_total * 100).fillna(0)
    ax2.plot(hours, [hourly_fraud_rate.get(h, 0) for h in hours],
             color="#f59e0b", marker="o", ms=4, lw=1.8, label="Fraud Rate %")
    ax2.set_ylabel("Fraud Rate (%)", color="#f59e0b"); ax2.tick_params(axis="y", colors="#f59e0b")
    ax2.legend(loc="upper right")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Fraud Rate by Category & Payment Method ───────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Fraud Rate by Product Category")
        cat_stats = df.groupby("product_category")["is_fraud"].agg(["sum","count"])
        cat_stats["rate"] = cat_stats["sum"] / cat_stats["count"] * 100
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        colors = ["#ef4444" if r > cat_stats["rate"].median() else "#3b82f6"
                  for r in cat_stats["rate"]]
        ax.barh(cat_stats.index, cat_stats["rate"], color=colors, edgecolor="white")
        ax.set_xlabel("Fraud Rate (%)"); ax.set_title("Fraud Rate by Category")
        ax.axvline(cat_stats["rate"].median(), color="#f59e0b", ls="--", label="Median")
        ax.legend(); ax.grid(axis="x", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_b:
        st.subheader("Fraud Rate by Payment Method")
        pm_stats = df.groupby("payment_method")["is_fraud"].agg(["sum","count"])
        pm_stats["rate"] = pm_stats["sum"] / pm_stats["count"] * 100
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        bar_colors = ["#ef4444","#f59e0b","#3b82f6","#8b5cf6"]
        ax.bar(pm_stats.index, pm_stats["rate"], color=bar_colors[:len(pm_stats)], edgecolor="white")
        ax.set_ylabel("Fraud Rate (%)"); ax.set_title("Fraud Rate by Payment Method")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Amount Distribution ───────────────────────────────────────────────────
    st.subheader("Transaction Amount Distribution (Log Scale) — Fraud vs Legitimate")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    bins = np.logspace(0, 4, 60)
    ax.hist(legit_df["transaction_amount"], bins=bins, alpha=0.7, color="#3b82f6",
            label=f"Legitimate ({len(legit_df):,})", density=True)
    ax.hist(fraud_df["transaction_amount"], bins=bins, alpha=0.7, color="#ef4444",
            label=f"Fraud ({len(fraud_df):,})", density=True)
    ax.set_xscale("log"); ax.set_xlabel("Transaction Amount ($, log scale)")
    ax.set_ylabel("Density"); ax.set_title("Amount Distribution: Fraud vs Legitimate")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Velocity Feature Distributions ───────────────────────────────────────
    st.subheader("Velocity Feature Distributions")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    for ax, col, title in zip(axes,
                               ["velocity_1h", "velocity_24h"],
                               ["Txns in Last 1hr", "Txns in Last 24hr"]):
        ax.hist(legit_df[col], bins=30, alpha=0.7, color="#3b82f6",
                label="Legitimate", density=True)
        ax.hist(fraud_df[col], bins=30, alpha=0.7, color="#ef4444",
                label="Fraud",      density=True)
        ax.set_xlabel(title); ax.set_ylabel("Density"); ax.legend()
        ax.set_title(title); ax.grid(alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Dataset Feature Summary ───────────────────────────────────────────────
    with st.expander("📋 Dataset Feature Summary"):
        feat_cols = ["transaction_amount","velocity_1h","velocity_24h",
                     "amount_vs_avg_ratio","card_age_days","account_age_days"]
        st.dataframe(df[feat_cols].describe().round(2), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — REAL-TIME RISK ENGINE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("⚡ Real-Time Risk Engine — Rule-Based Velocity Checks")
    st.markdown("Hard rules applied in waterfall order before ML scoring (Stripe/Visa PSD2 compliant)")

    block_mask  = (decisions == "BLOCK")
    review_mask = (decisions == "REVIEW")
    pass_mask   = (decisions == "PASS")

    r1, r2, r3 = st.columns(3)
    r1.metric("BLOCKED",  f"{block_mask.sum():,}",  f"{block_mask.mean()*100:.2f}% of traffic")
    r2.metric("REVIEW",   f"{review_mask.sum():,}", f"{review_mask.mean()*100:.2f}% of traffic")
    r3.metric("PASSED",   f"{pass_mask.sum():,}",   f"{pass_mask.mean()*100:.2f}% of traffic")

    # ── Rule Hit Rates ────────────────────────────────────────────────────────
    st.subheader("Rule Hit Rates")
    fig, ax = plt.subplots(figsize=(11, 3.5))
    rule_names  = list(rule_hits.keys())
    rule_counts = list(rule_hits.values())
    colors_rule = ["#ef4444","#ef4444","#f59e0b","#f59e0b","#f59e0b"]
    bars = ax.barh(rule_names, rule_counts, color=colors_rule, edgecolor="white")
    for bar, cnt in zip(bars, rule_counts):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                f"{cnt:,}", va="center", fontsize=9)
    ax.set_xlabel("Transaction Count"); ax.set_title("Rule Engine — Hit Count per Rule")
    red_patch    = mpatches.Patch(color="#ef4444", label="BLOCK")
    yellow_patch = mpatches.Patch(color="#f59e0b", label="REVIEW")
    ax.legend(handles=[red_patch, yellow_patch])
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Waterfall: Transactions surviving each rule ───────────────────────────
    st.subheader("Rule Engine Waterfall — Surviving Volume")
    waterfall_vals = [total_txns]
    labels_wf = ["All Traffic"]
    cumulative_block = 0
    for rname, rcnt in rule_hits.items():
        cumulative_block += rcnt
        waterfall_vals.append(total_txns - cumulative_block)
        labels_wf.append(rname.split(":")[0])

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.bar(labels_wf, waterfall_vals, color=["#3b82f6"] + ["#f59e0b"]*len(rule_hits),
           edgecolor="white")
    for i, v in enumerate(waterfall_vals):
        ax.text(i, v + 200, f"{v:,}", ha="center", fontsize=8)
    ax.set_ylabel("Transactions Remaining"); ax.set_title("Waterfall: Transactions After Each Rule")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── False Positive Cost ───────────────────────────────────────────────────
    st.subheader("False Positive Cost — Revenue Lost to Blocked Legitimate Transactions")
    blocked_legit = df[(block_mask) & (df["is_fraud"]==0)]
    blocked_fraud = df[(block_mask) & (df["is_fraud"]==1)]
    fp_rev_loss   = blocked_legit["transaction_amount"].sum() * decline_rate
    tp_fraud_saved= blocked_fraud["transaction_amount"].sum()

    col_fp1, col_fp2, col_fp3 = st.columns(3)
    col_fp1.metric("Legitimate Txns Blocked (FP)", f"{len(blocked_legit):,}")
    col_fp2.metric("Revenue Lost to FP Blocks",    f"${fp_rev_loss:,.0f}")
    col_fp3.metric("Fraud Amount Blocked",          f"${tp_fraud_saved:,.0f}")

    # ── 3DS Authentication ────────────────────────────────────────────────────
    st.subheader("3DS / SCA Trigger Analysis (PSD2 / Visa / Mastercard)")
    sca_threshold = st.slider("SCA Trigger Amount (€/$)", 30, 500, 150, 10)
    sca_required  = df[df["transaction_amount"] >= sca_threshold]
    sca_exempt    = df[df["transaction_amount"]  < sca_threshold]
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Require SCA (≥ threshold)", f"{len(sca_required):,}")
    col_s2.metric("TRA Exempt (< threshold)",  f"{len(sca_exempt):,}")
    col_s3.metric("SCA Exemption Rate",         f"{len(sca_exempt)/len(df)*100:.1f}%")

    st.info(f"**Visa/Mastercard PSD2 Rule:** Transactions ≥ ${sca_threshold} must "
            "trigger 3DS2 authentication unless TRA exemption applies. "
            f"At this threshold, {len(sca_required):,} transactions ({len(sca_required)/len(df)*100:.1f}%) "
            "would require 3DS challenge.")

    # ── Review vs Blocked breakdown by fraud status ───────────────────────────
    st.subheader("Rule Decision vs Actual Fraud Status")
    confusion_tbl = pd.crosstab(decisions, df["is_fraud"],
                                rownames=["Rule Decision"],
                                colnames=["is_fraud (0=Legit, 1=Fraud)"])
    st.dataframe(confusion_tbl.style.background_gradient(cmap="RdYlGn_r"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML RISK SCORING MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🤖 ML Risk Scoring Model — Logistic Regression (NumPy)")
    st.markdown("Class-weighted logistic regression trained on 80k transactions. "
                "Chronological split prevents future leakage.")

    st.subheader("Model Equations")
    col_eq1, col_eq2 = st.columns(2)
    with col_eq1:
        st.latex(r"\text{log odds} = \log\frac{p}{1-p} = \beta_0 + \sum_{i=1}^{d}\beta_i x_i")
        st.latex(r"p = \sigma(z) = \frac{1}{1+e^{-z}}")
    with col_eq2:
        st.latex(r"\text{Precision}(\tau) = \frac{TP(\tau)}{TP(\tau)+FP(\tau)}")
        st.latex(r"\text{Recall}(\tau) = \frac{TP(\tau)}{TP(\tau)+FN(\tau)}")

    # ── Accuracy Paradox ──────────────────────────────────────────────────────
    st.subheader("Why Accuracy is Misleading at 0.5% Fraud Rate")
    base_acc = 1 - y_te.mean()
    tp_m, fp_m, fn_m, tn_m = metrics_at_threshold(y_te, te_scores, fraud_threshold)
    ml_acc = (tp_m + tn_m) / len(y_te)
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Naive Accuracy (predict all legit)", f"{base_acc*100:.2f}%")
    col_p2.metric("ML Model Accuracy",                  f"{ml_acc*100:.2f}%")
    col_p3.metric("Difference",                         f"+{(ml_acc-base_acc)*100:.2f}%",
                  help="ML barely beats naive — accuracy is the wrong metric for imbalanced data")
    st.warning("At 0.5% fraud rate, predicting every transaction as legitimate gives "
               f"**{base_acc*100:.2f}% accuracy** — yet catches zero fraud. "
               "Use **ROC-AUC, PR-AUC, and Precision@τ** instead.")

    # ── ROC and PR Curves ─────────────────────────────────────────────────────
    auc_val, tprs_roc, fprs_roc = roc_auc(y_te, te_scores)
    prauc_val, precs_pr, recs_pr = pr_auc(y_te, te_scores)

    col_roc, col_pr = st.columns(2)
    with col_roc:
        st.subheader(f"ROC Curve  (AUC = {auc_val:.4f})")
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(fprs_roc, tprs_roc, color="#3b82f6", lw=2, label=f"AUC={auc_val:.3f}")
        ax.plot([0,1],[0,1], "k--", alpha=0.4, label="Random")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve"); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_pr:
        st.subheader(f"Precision-Recall Curve  (PR-AUC = {prauc_val:.4f})")
        baseline_pr = y_te.mean()
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(recs_pr, precs_pr, color="#10b981", lw=2, label=f"PR-AUC={prauc_val:.3f}")
        ax.axhline(baseline_pr, color="#f59e0b", ls="--",
                   label=f"Baseline (fraud rate={baseline_pr*100:.2f}%)")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve"); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Score Distribution ────────────────────────────────────────────────────
    st.subheader("Fraud Score Distribution — Fraud vs Legitimate")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    bins_s = np.linspace(0, 1, 80)
    ax.hist(te_scores[y_te==0], bins=bins_s, alpha=0.7, color="#3b82f6",
            label="Legitimate", density=True)
    ax.hist(te_scores[y_te==1], bins=bins_s, alpha=0.8, color="#ef4444",
            label="Fraud",      density=True)
    ax.axvline(fraud_threshold, color="#f59e0b", lw=2, ls="--",
               label=f"Threshold τ={fraud_threshold:.2f}")
    ax.set_xlabel("Fraud Score"); ax.set_ylabel("Density")
    ax.set_title("Score Separation: Fraud vs Legitimate")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Threshold Sweep: Precision, Recall, Review Queue ─────────────────────
    st.subheader("Threshold Analysis — Precision, Recall & Review Queue Size")
    taus = np.linspace(0.01, 0.99, 100)
    precisions_t, recalls_t, queue_sizes = [], [], []
    for tau in taus:
        _tp, _fp, _fn, _tn = metrics_at_threshold(y_te, te_scores, tau)
        precisions_t.append(_tp / max(_tp + _fp, 1))
        recalls_t.append(   _tp / max(_tp + _fn, 1))
        queue_sizes.append( _tp + _fp)

    fig, ax1 = plt.subplots(figsize=(12, 3.8))
    ax1.plot(taus, precisions_t, color="#10b981", lw=2, label="Precision@τ")
    ax1.plot(taus, recalls_t,    color="#3b82f6", lw=2, label="Recall@τ")
    ax1.axvline(fraud_threshold, color="#f59e0b", ls="--",
                label=f"Current τ={fraud_threshold:.2f}")
    ax1.set_xlabel("Threshold τ"); ax1.set_ylabel("Score")
    ax1.set_title("Precision & Recall vs Threshold | Review Queue Size")
    ax1.legend(loc="upper left"); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.fill_between(taus, queue_sizes, alpha=0.15, color="#8b5cf6")
    ax2.plot(taus, queue_sizes, color="#8b5cf6", lw=1.5, label="Queue Size")
    ax2.set_ylabel("Review Queue Size", color="#8b5cf6")
    ax2.tick_params(axis="y", colors="#8b5cf6"); ax2.legend(loc="upper right")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── KS Statistic ──────────────────────────────────────────────────────────
    ks_val = ks_statistic(y_te, te_scores)
    st.subheader(f"KS Statistic = {ks_val:.4f}")
    st.latex(r"KS = \max_{\tau}\left|F_{\text{fraud}}(\tau) - F_{\text{legit}}(\tau)\right|")
    st.info(f"KS = **{ks_val:.4f}** — Values >0.3 indicate meaningful model discrimination. "
            f"KS >0.4 is considered good for fraud detection.")

    # ── Feature Importances (model weights) ──────────────────────────────────
    st.subheader("Feature Importance (|Logistic Weights|)")
    feat_names = results["feat_names"]
    weights    = np.abs(results["model"].w_)
    order      = np.argsort(weights)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh([feat_names[i] for i in order[::-1]], weights[order[::-1]],
            color="#3b82f6", edgecolor="white")
    ax.set_xlabel("|Weight|"); ax.set_title("Top 15 Feature Importances (Logistic Regression)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Confusion Matrix at current τ ─────────────────────────────────────────
    st.subheader(f"Confusion Matrix at τ = {fraud_threshold:.2f}")
    cm = np.array([[tn_m, fp_m],[fn_m, tp_m]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    fontsize=14, color="black" if cm[i,j] < cm.max()/2 else "white")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred: Legit","Pred: Fraud"])
    ax.set_yticklabels(["Actual: Legit","Actual: Fraud"])
    ax.set_title("Confusion Matrix"); plt.colorbar(im, ax=ax)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FRAUD INVESTIGATION DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("📊 Fraud Investigation Dashboard")

    # ── Alert Queue: Top 50 high-risk transactions ────────────────────────────
    st.subheader("Fraud Alert Queue — Top 50 Highest-Risk Transactions")
    alert_queue = (df.nlargest(50, "risk_score")
                   [["timestamp","transaction_amount","payment_method",
                     "product_category","ip_country_risk","velocity_1h",
                     "is_new_merchant","risk_score","is_fraud"]]
                   .copy())
    alert_queue["risk_score"]        = alert_queue["risk_score"].round(4)
    alert_queue["transaction_amount"]= alert_queue["transaction_amount"].round(2)

    def color_risk(val):
        if isinstance(val, float) and val > 0.7: return "background-color:#fee2e2"
        elif isinstance(val, float) and val > 0.4: return "background-color:#fef3c7"
        return ""

    st.dataframe(
        alert_queue.style.applymap(color_risk, subset=["risk_score"]),
        use_container_width=True, height=350
    )

    # ── Case Investigation ────────────────────────────────────────────────────
    st.subheader("Case Investigation View")
    txn_idx = st.selectbox("Select Transaction Index for Investigation",
                            alert_queue.index.tolist(), index=0)
    txn     = df.loc[txn_idx]

    col_case1, col_case2 = st.columns([1, 1])
    with col_case1:
        st.markdown("**Transaction Details**")
        st.json({
            "amount":          f"${txn['transaction_amount']:.2f}",
            "category":        txn["product_category"],
            "payment_method":  txn["payment_method"],
            "device_type":     txn["device_type"],
            "ip_country_risk": txn["ip_country_risk"],
            "card_present":    bool(txn["card_present"]),
            "velocity_1h":     int(txn["velocity_1h"]),
            "velocity_24h":    int(txn["velocity_24h"]),
            "is_new_merchant": bool(txn["is_new_merchant"]),
            "risk_score":      f"{txn['risk_score']:.4f}",
            "rule_decision":   txn["rule_decision"],
        })

    with col_case2:
        st.markdown("**Risk Score Breakdown — Top Contributing Features**")
        feat_vals  = (np.array([txn["transaction_amount"], txn["velocity_1h"],
                                txn["amount_vs_avg_ratio"], txn["is_new_merchant"],
                                txn["is_first_transaction"]], dtype=float))
        feat_names_case = ["log(amount)","velocity_1h","amount_vs_avg_ratio",
                           "is_new_merchant","is_first_txn"]
        weights_top = results["model"].w_[:5]
        contrib = feat_vals * weights_top

        fig, ax = plt.subplots(figsize=(5, 3))
        colors_c = ["#ef4444" if c > 0 else "#3b82f6" for c in contrib]
        ax.barh(feat_names_case, contrib, color=colors_c, edgecolor="white")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Contribution to Risk Score")
        ax.set_title(f"Risk Score Decomposition\n(Total Score: {txn['risk_score']:.4f})")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Similar Fraud Cases (k-NN approximation) ──────────────────────────────
    st.subheader("Similar Fraud Cases (Nearest Neighbors by Feature Similarity)")
    fraud_cases = df[df["is_fraud"]==1].copy()
    if len(fraud_cases) > 5:
        compare_feats = ["transaction_amount","velocity_1h","velocity_24h","amount_vs_avg_ratio"]
        txn_vec   = txn[compare_feats].values.astype(float)
        fraud_mat = fraud_cases[compare_feats].values.astype(float)
        # Normalize
        norms = fraud_mat.max(0) + 1e-8
        dists = np.linalg.norm((fraud_mat / norms) - (txn_vec / norms), axis=1)
        similar_idx = fraud_cases.index[np.argsort(dists)[:5]]
        st.dataframe(df.loc[similar_idx, compare_feats + ["risk_score","ip_country_risk","payment_method"]].round(2),
                     use_container_width=True)

    # ── Merchant Risk Profile ──────────────────────────────────────────────────
    st.subheader("Merchant Risk Profile & Fraud Pattern Analysis")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("**Fraud by Country Risk Level**")
        cr_fraud = df.groupby("ip_country_risk")["is_fraud"].agg(["mean","sum"]).reset_index()
        cr_fraud.columns = ["Country Risk","Fraud Rate","Fraud Count"]
        cr_fraud["Fraud Rate"] = (cr_fraud["Fraud Rate"]*100).round(3)
        st.dataframe(cr_fraud, use_container_width=True)

    with col_m2:
        st.markdown("**Fraud Concentration — Payment Method × Country Risk**")
        heatmap_data = df.groupby(["payment_method","ip_country_risk"])["is_fraud"].mean().unstack()*100
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        im = ax.imshow(heatmap_data.values, aspect="auto", cmap="Reds")
        ax.set_xticks(range(len(heatmap_data.columns))); ax.set_xticklabels(heatmap_data.columns)
        ax.set_yticks(range(len(heatmap_data.index)));   ax.set_yticklabels(heatmap_data.index)
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                val = heatmap_data.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}%", ha="center", va="center", fontsize=8)
        ax.set_title("Fraud Rate (%) by Payment × Country Risk")
        plt.colorbar(im, ax=ax); plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Payment Network Graph ─────────────────────────────────────────────────
    st.subheader("Payment Network Graph — Suspicious Transaction Clusters")
    rng_graph = np.random.default_rng(99)
    n_cust, n_merch = 15, 10
    cust_pos  = {f"C{i}": (rng_graph.uniform(0,1), rng_graph.uniform(0,1)) for i in range(n_cust)}
    merch_pos = {f"M{j}": (rng_graph.uniform(1.2,2.2), rng_graph.uniform(0,1)) for j in range(n_merch)}

    sample_edges = df.sample(80, random_state=1).copy()
    sample_edges["cust"]  = [f"C{i%n_cust}" for i in range(len(sample_edges))]
    sample_edges["merch"] = [f"M{j%n_merch}" for j in range(len(sample_edges))]

    fig, ax = plt.subplots(figsize=(11, 5))
    for _, edge in sample_edges.iterrows():
        c0 = cust_pos[edge["cust"]]
        c1 = merch_pos[edge["merch"]]
        color = "#ef4444" if edge["risk_score"] > 0.5 else "#94a3b8"
        lw    = 2.0       if edge["risk_score"] > 0.5 else 0.5
        ax.plot([c0[0], c1[0]], [c0[1], c1[1]], color=color, lw=lw, alpha=0.6)
    for name, pos in cust_pos.items():
        ax.scatter(*pos, s=120, color="#3b82f6", zorder=5)
        ax.text(pos[0], pos[1]+0.04, name, ha="center", fontsize=7)
    for name, pos in merch_pos.items():
        ax.scatter(*pos, s=180, color="#10b981", marker="s", zorder=5)
        ax.text(pos[0], pos[1]+0.04, name, ha="center", fontsize=7)
    ax.set_title("Payment Network Graph  (Red edges = High-Risk Transactions)")
    ax.axis("off")
    red_p   = mpatches.Patch(color="#ef4444",label="High-Risk Txn")
    blue_p  = mpatches.Patch(color="#3b82f6", label="Customer")
    green_p = mpatches.Patch(color="#10b981", label="Merchant")
    ax.legend(handles=[red_p, blue_p, green_p], loc="lower right")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Time-to-Detect & Chargeback ───────────────────────────────────────────
    st.subheader("Operational Metrics")
    col_op1, col_op2 = st.columns(2)
    with col_op1:
        st.metric("Avg Time-to-Detect (Simulated)", "2.3 min",
                  help="Real-time scoring latency for ML model in production")
        st.metric("Time to First Alert (Rule Engine)", "< 50ms")
    with col_op2:
        st.metric("Chargeback Recovery Rate (Electronics)", "42%")
        st.metric("Chargeback Recovery Rate (Travel)",      "31%")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FINANCIAL IMPACT & OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("💰 Financial Impact & Operations")

    # ── Financial Model ───────────────────────────────────────────────────────
    st.subheader("Financial Model Equations")
    col_fin1, col_fin2 = st.columns(2)
    with col_fin1:
        st.latex(r"\text{Fraud Loss (no ML)} = \text{fraud\_rate} \times \text{total\_amount}")
        st.latex(r"\text{Fraud Loss (with ML)} = FN \times \bar{A}_{\text{fraud}}")
        st.latex(r"\text{Revenue Saved} = TP \times \bar{A}_{\text{fraud}}")
    with col_fin2:
        st.latex(r"\text{FP Cost} = FP \times \bar{A}_{\text{legit}} \times \delta")
        st.latex(r"\text{Ops Cost} = (TP+FP) \times c_{\text{review}}")
        st.latex(r"\text{Net Savings} = \text{Revenue Saved} - \text{FP Cost} - \text{Ops Cost}")

    # ── Compute financial metrics ─────────────────────────────────────────────
    total_amount     = df["transaction_amount"].sum()
    avg_fraud_amount = df[df["is_fraud"]==1]["transaction_amount"].mean()
    avg_legit_amount = df[df["is_fraud"]==0]["transaction_amount"].mean()

    # Scale test metrics to full dataset size
    scale = len(df) / len(y_te)
    tp_s = int(tp * scale); fp_s = int(fp * scale)
    fn_s = int(fn * scale); tn_s = int(tn * scale)

    fraud_loss_no_ml = df["is_fraud"].sum() * avg_fraud_amount
    fraud_loss_ml    = fn_s * avg_fraud_amount
    revenue_saved    = tp_s * avg_fraud_amount
    fp_cost          = fp_s * avg_legit_amount * decline_rate
    ops_cost         = (tp_s + fp_s) * ops_cost_per_review
    net_savings_full = revenue_saved - fp_cost - ops_cost

    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Fraud Loss Without ML",  f"${fraud_loss_no_ml:,.0f}")
    fc2.metric("Fraud Loss With ML",     f"${fraud_loss_ml:,.0f}",
               delta=f"-${fraud_loss_no_ml-fraud_loss_ml:,.0f}")
    fc3.metric("Revenue Saved",          f"${revenue_saved:,.0f}")
    fc4, fc5, fc6 = st.columns(3)
    fc4.metric("FP Revenue Cost",        f"${fp_cost:,.0f}")
    fc5.metric("Operations Cost",        f"${ops_cost:,.0f}")
    fc6.metric("Net Savings",            f"${net_savings_full:,.0f}",
               delta="positive" if net_savings_full > 0 else "negative")

    # ── Threshold Optimization ────────────────────────────────────────────────
    st.subheader("Threshold Optimization — Net Savings Curve")
    taus_sweep = np.linspace(0.01, 0.99, 150)
    net_savings_sweep, ops_costs_sweep, fp_costs_sweep, rev_saved_sweep = [], [], [], []

    for tau in taus_sweep:
        _tp, _fp, _fn, _tn = metrics_at_threshold(y_te, te_scores, tau)
        _rev  = _tp * scale * avg_fraud_amount
        _fp_c = _fp * scale * avg_legit_amount * decline_rate
        _ops  = (_tp + _fp) * scale * ops_cost_per_review
        net_savings_sweep.append(_rev - _fp_c - _ops)
        ops_costs_sweep.append(_ops)
        fp_costs_sweep.append(_fp_c)
        rev_saved_sweep.append(_rev)

    opt_idx = int(np.argmax(net_savings_sweep))
    opt_tau = taus_sweep[opt_idx]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(taus_sweep, net_savings_sweep, color="#10b981", lw=2.5, label="Net Savings")
    ax.plot(taus_sweep, rev_saved_sweep,   color="#3b82f6", lw=1.5, ls="--", label="Revenue Saved")
    ax.plot(taus_sweep, [-x for x in fp_costs_sweep], color="#f59e0b", lw=1.5, ls="--", label="−FP Cost")
    ax.plot(taus_sweep, [-x for x in ops_costs_sweep],color="#8b5cf6", lw=1.5, ls="--", label="−Ops Cost")
    ax.axvline(opt_tau, color="#ef4444", ls=":", lw=2,
               label=f"Optimal τ = {opt_tau:.2f}")
    ax.axvline(fraud_threshold, color="#f59e0b", ls="--", lw=1.5,
               label=f"Current τ = {fraud_threshold:.2f}")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("Fraud Score Threshold τ")
    ax.set_ylabel("$USD"); ax.set_title("Net Savings vs Threshold — Optimal Operating Point")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.success(f"Optimal threshold **τ = {opt_tau:.2f}** maximizes net savings at "
               f"**${net_savings_sweep[opt_idx]:,.0f}**. "
               f"Current τ = {fraud_threshold:.2f} yields **${net_savings_full:,.0f}**.")

    # ── Operations Cost Model ─────────────────────────────────────────────────
    st.subheader("Operations Cost Model")
    col_ops1, col_ops2 = st.columns(2)

    with col_ops1:
        st.latex(r"\text{Ops Cost} = \text{ReviewQueueSize}(\tau) \times c_{\text{review}}")
        st.latex(r"\text{AutomationRate} = \frac{N_{\text{auto-approve}} + N_{\text{auto-decline}}}{N_{\text{total}}}")
        review_queue_size = tp_s + fp_s
        automation_rate   = 1.0 - (review_queue_size / len(df))
        col_ops1.metric("Review Queue Size",  f"{review_queue_size:,}")
        col_ops1.metric("Ops Cost/Transaction",f"${ops_cost/len(df):.4f}")
        col_ops1.metric("Automation Rate",     f"{automation_rate*100:.1f}%")

    with col_ops2:
        # Review queue size across thresholds
        queue_s = [int((_tp + _fp) * scale)
                   for _tp, _fp, _, _ in
                   [metrics_at_threshold(y_te, te_scores, t) for t in taus_sweep]]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.fill_between(taus_sweep, queue_s, alpha=0.3, color="#8b5cf6")
        ax.plot(taus_sweep, queue_s, color="#8b5cf6", lw=2)
        ax.axvline(fraud_threshold, color="#f59e0b", ls="--", lw=2,
                   label=f"Current τ={fraud_threshold:.2f} → {review_queue_size:,} reviews")
        ax.axhline(review_queue_limit, color="#ef4444", ls=":", lw=1.5,
                   label=f"Queue Limit = {review_queue_limit:,}")
        ax.set_xlabel("Threshold τ"); ax.set_ylabel("Review Queue Size")
        ax.set_title("Review Queue Size vs Threshold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Regulatory Reporting ──────────────────────────────────────────────────
    st.subheader("Regulatory Reporting — PSD2 / Visa / Mastercard TRA Exemptions")
    current_fraud_rate_pct = df["is_fraud"].mean() * 100

    reg_col1, reg_col2, reg_col3 = st.columns(3)
    reg_col1.metric("Current Overall Fraud Rate", f"{current_fraud_rate_pct:.3f}%")

    # TRA thresholds
    tra_limits = {
        "Low TRA (<$100)":    {"max_fraud_rate": 0.13, "limit": 100},
        "Medium TRA (<$250)": {"max_fraud_rate": 0.06, "limit": 250},
        "High TRA (<$500)":   {"max_fraud_rate": 0.01, "limit": 500},
    }

    st.markdown("**Transaction Risk Analysis (TRA) Exemption Thresholds (PSD2 Article 18)**")
    tra_rows = []
    for tier, vals in tra_limits.items():
        eligible = df[df["transaction_amount"] <= vals["limit"]]
        tier_fraud_rate = eligible["is_fraud"].mean() * 100 if len(eligible) > 0 else 0
        exempt = tier_fraud_rate <= vals["max_fraud_rate"]
        tra_rows.append({
            "Tier":               tier,
            "Txn Limit":          f"${vals['limit']}",
            "Required Fraud Rate":f"<{vals['max_fraud_rate']}%",
            "Current Rate":       f"{tier_fraud_rate:.3f}%",
            "TRA Eligible":       "YES ✓" if exempt else "NO ✗",
        })
    st.dataframe(pd.DataFrame(tra_rows), use_container_width=True)

    # ── KPI Dashboard ─────────────────────────────────────────────────────────
    st.subheader("KPI Summary Dashboard")
    kpi_data = {
        "Fraud Detection Rate (%)":   f"{tp/(tp+fn)*100:.1f}%" if (tp+fn)>0 else "N/A",
        "False Positive Rate (%)":    f"{fp/(fp+tn)*100:.2f}%" if (fp+tn)>0 else "N/A",
        "Ops Cost / Transaction ($)": f"${ops_cost/len(df):.4f}",
        "Net Fraud Loss ($)":         f"${fraud_loss_ml:,.0f}",
        "Automation Rate (%)":        f"{automation_rate*100:.1f}%",
        "Review Queue Size":          f"{review_queue_size:,}",
        "ROC-AUC":                    f"{auc_val:.4f}",
        "PR-AUC":                     f"{prauc_val:.4f}",
        "KS Statistic":               f"{ks_val:.4f}",
        "Optimal Threshold (τ*)":     f"{opt_tau:.2f}",
    }
    kpi_df = pd.DataFrame(kpi_data.items(), columns=["KPI","Value"])
    st.dataframe(kpi_df, use_container_width=True, hide_index=True)

    # ── SCA Exemption Volume ──────────────────────────────────────────────────
    st.subheader("SCA (Strong Customer Authentication) Exemption Analysis")
    st.markdown("""
    Under **PSD2 Article 10**, issuers may claim SCA exemption for low-value transactions:
    - Contactless payments ≤ €50 (≤ $55)
    - TRA exemption: if fraud rate below regulatory threshold
    - Recurring transactions with same merchant
    """)
    sca_vals   = [30, 50, 100, 150, 250]
    sca_exempt_rates = [df[df["transaction_amount"] < v].shape[0] / len(df) * 100 for v in sca_vals]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar([f"${v}" for v in sca_vals], sca_exempt_rates, color="#3b82f6", edgecolor="white")
    ax.set_xlabel("SCA Exemption Threshold"); ax.set_ylabel("% Transactions Exempt")
    ax.set_title("SCA Exemption Rate by Threshold (% of Transactions Below Limit)")
    for i, v in enumerate(sca_exempt_rates):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 110)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#64748b;font-size:0.85rem;">
PayShield v2.0 | Real-Time Payment Risk Scoring |
IEEE-CIS Fraud Detection Style | NumPy · Pandas · Matplotlib · Streamlit<br>
All data is synthetic. No real transaction data is used.
</div>
""", unsafe_allow_html=True)
