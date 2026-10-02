import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

train = pd.read_csv("data/train_test.csv", parse_dates=["date"])
train["rate_per_mile"] = train["posted_rate"] / train["distance"]
train["is_bad"] = (train["rate_per_mile"] < 1.2) | (train["rate_per_mile"] > 4.0)

def make_features(df, drop_cols=()):
    X = df[["distance", "weight", "market_index", "quote_signal"]].drop(columns=list(drop_cols))
    return X.join(pd.get_dummies(df["equipment"]).astype(int))

def report(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{label:58s} MAE={mae:7.1f}  RMSE={rmse:7.1f}  MAPE={mape:5.2f}%")

def fit_eval(tr, te, label, drop_bad_train=True, drop_cols=()):
    tr_fit = tr[~tr["is_bad"]] if drop_bad_train else tr
    model = HistGradientBoostingRegressor(random_state=0)   # handles NaN natively
    model.fit(make_features(tr_fit, drop_cols), tr_fit["posted_rate"])
    pred = model.predict(make_features(te, drop_cols))
    ok = ~te["is_bad"].values
    y = te["posted_rate"].values
    report(y[ok], pred[ok], label + " [clean test]")
    report(y, pred, label + " [all test]")

# ---------- Experiment 1: does the split change the answer? ----------
print("\n=== Experiment 1: random vs time split ===")
tr_a, te_a = train_test_split(train, test_size=0.2, random_state=42)
cutoff = pd.Timestamp("2025-09-01")
tr_b, te_b = train[train["date"] < cutoff], train[train["date"] >= cutoff]
for drop in (False, True):
    fit_eval(tr_a, te_a, f"random 80/20 | drop_bad_train={drop}", drop_bad_train=drop)
    fit_eval(tr_b, te_b, f"time split   | drop_bad_train={drop}", drop_bad_train=drop)

# ---------- Experiment 2: December-style feature availability ----------
print("\n=== Experiment 2: what if market features aren't available? ===")
fit_eval(tr_b, te_b, "A: real market features")
fit_eval(tr_b, te_b, "B: drop market features", drop_cols=("market_index", "quote_signal"))
recent = tr_b[tr_b["date"] >= tr_b["date"].max() - pd.Timedelta(days=6)]
te_c = te_b.assign(market_index=recent["market_index"].mean(),
                   quote_signal=recent["quote_signal"].mean())
fit_eval(tr_b, te_c, "C: frozen last-7-day values")

def clean_scores(tr, te, drop_cols=()):
    tr_fit = tr[~tr["is_bad"]]
    model = HistGradientBoostingRegressor(random_state=0)
    model.fit(make_features(tr_fit, drop_cols), tr_fit["posted_rate"])
    pred = model.predict(make_features(te, drop_cols))
    ok = ~te["is_bad"].values
    y, p = te["posted_rate"].values[ok], pred[ok]
    return mean_absolute_error(y, p), np.mean(np.abs((y - p) / y)) * 100

variants = {
    "all features":    (),
    "no market_index": ("market_index",),
    "no quote_signal": ("quote_signal",),
    "neither":         ("market_index", "quote_signal"),
}

print("\n=== Experiment 3: several time windows ===")
for start in ["2025-05-01", "2025-07-01", "2025-09-01"]:      # your choice to change
    s = pd.Timestamp(start)
    tr = train[train["date"] < s]
    te = train[(train["date"] >= s) & (train["date"] < s + pd.DateOffset(months=2))]
    for name, cols in variants.items():
        mae, mape = clean_scores(tr, te, cols)
        print(f"test from {start} (+2mo) | {name:16s} MAE={mae:6.1f}  MAPE={mape:5.2f}%")

print("\n=== Experiment 4: same variants, random split ===")
for name, cols in variants.items():
    mae, mape = clean_scores(tr_a, te_a, cols)
    print(f"random 80/20 | {name:16s} MAE={mae:6.1f}  MAPE={mape:5.2f}%")

def clean_scores_seed(tr, te, drop_cols=(), seed=0):
    tr_fit = tr[~tr["is_bad"]]
    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(make_features(tr_fit, drop_cols), tr_fit["posted_rate"])
    pred = model.predict(make_features(te, drop_cols))
    ok = ~te["is_bad"].values
    y, p = te["posted_rate"].values[ok], pred[ok]
    return mean_absolute_error(y, p)

print("\n=== Experiment 5: monthly rolling-origin, 3 seeds each ===")
rows = []
for m in pd.period_range("2025-04", "2025-10", freq="M"):
    tr = train[train["date"] < m.start_time]
    te = train[(train["date"] >= m.start_time) & (train["date"] < (m + 1).start_time)]
    for name, cols in variants.items():
        maes = [clean_scores_seed(tr, te, cols, s) for s in (0, 1, 2)]
        rows.append({"month": str(m), "variant": name, "MAE": np.mean(maes), "seed_spread": np.ptp(maes)})
res = pd.DataFrame(rows)
print(res.pivot(index="month", columns="variant", values="MAE").round(1))
print(res.groupby("variant")[["MAE", "seed_spread"]].agg(["mean", "std"]).round(1))

print("\n=== Market vs rate per mile, by month (clean rows) ===")
clean = train[~train["is_bad"]]
print(clean.groupby(clean["date"].dt.to_period("M")).agg(
    market=("market_index", "mean"), qs=("quote_signal", "mean"),
    rpm=("rate_per_mile", "mean")).round(3))