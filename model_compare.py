import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

train = pd.read_csv("data/train_test.csv", parse_dates=["date"])
train["rpm"] = train["posted_rate"] / train["distance"]
train["is_bad"] = (train["rpm"] < 1.2) | (train["rpm"] > 4.0)
train["mi_daily"] = train.groupby("date")["market_index"].transform("mean")  # features only, per date
eq = pd.get_dummies(train["equipment"]).astype(int)
EQ = list(eq.columns)
train = train.join(eq)

# ---- Diagnosis: what is quote_signal? ----
clean = train[~train["is_bad"]]
g = clean.groupby(clean["date"].dt.to_period("M"))
print(pd.DataFrame({
    "std(qs-rpm)": g.apply(lambda d: (d["quote_signal"] - d["rpm"]).std()),
    "std(qs+rpm)": g.apply(lambda d: (d["quote_signal"] + d["rpm"]).std()),
}).round(3))
dc = clean.groupby("date").apply(lambda d: d["quote_signal"].corr(d["rpm"]))
print(dc.groupby(dc.index.to_period("M")).agg(
    lambda s: f"+:{(s > 0.5).sum()}  -:{(s < -0.5).sum()}  mixed:{(s.abs() <= 0.5).sum()}"))

# ---- Rolling-origin machinery ----
MONTHS = pd.period_range("2025-04", "2025-10", freq="M")

def folds():
    for m in MONTHS:
        tr = train[(train["date"] < m.start_time) & ~train["is_bad"]]
        te = train[(train["date"] >= m.start_time) & (train["date"] < (m + 1).start_time) & ~train["is_bad"]]
        yield str(m), tr, te

def evaluate(make_model, cols, target="rpm", train_cap=None):
    rows = []
    for name, tr, te in folds():
        if train_cap and len(tr) > train_cap:
            tr = tr.sample(train_cap, random_state=0)
        model = make_model()
        model.fit(tr[cols], tr["rpm"] if target == "rpm" else tr["posted_rate"])
        pred = model.predict(te[cols])
        if target == "rpm":
            pred = pred * te["distance"].values
        err = np.abs(te["posted_rate"].values - pred)
        rows.append({"month": name, "MAE": err.mean(),
                     "MAPE": (err / te["posted_rate"].values).mean() * 100})
    return pd.DataFrame(rows).set_index("month")

def summarize(title, results):
    print(f"\n=== {title} ===")
    mae = pd.DataFrame({k: v["MAE"] for k, v in results.items()})
    mape = pd.DataFrame({k: v["MAPE"] for k, v in results.items()})
    print(mae.round(1))
    print("MAE  mean/std:\n", mae.agg(["mean", "std"]).round(1))
    print("MAPE mean:\n", mape.mean().round(2))
    first = mae.columns[0]
    print(f"months each column beats '{first}':\n", (mae.sub(mae[first], axis=0) < 0).sum())

# ---- Candidate models ----
hgb = lambda: HistGradientBoostingRegressor(random_state=0)
ridge = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
rf = lambda: make_pipeline(SimpleImputer(strategy="median"),
                           RandomForestRegressor(n_estimators=100, min_samples_leaf=3, n_jobs=-1, random_state=0))
svr = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVR(C=3.0))

class MedianRPM:                       # naive yardstick: same $/mile for everyone
    def fit(self, X, y): self.m = float(np.median(y)); return self
    def predict(self, X): return np.full(len(X), self.m)

# ---- Experiment A: which features? (HGB, target = rate per mile) ----
FEATURES = {
    "base":          ["distance", "weight"] + EQ,
    "+market_row":   ["distance", "weight", "market_index"] + EQ,
    "+market_daily": ["distance", "weight", "mi_daily"] + EQ,
}
summarize("A: feature sets", {k: evaluate(hgb, c) for k, c in FEATURES.items()})

# ---- Experiment B: target definition (HGB, +market_daily) ----
cols = FEATURES["+market_daily"]
summarize("B: target", {"target=rpm": evaluate(hgb, cols, "rpm"),
                        "target=dollars": evaluate(hgb, cols, "dollars")})

# ---- Experiment C: candidate models (target = rate per mile) ----
summarize("C: models", {
    "naive median rpm": evaluate(MedianRPM, cols),
    "ridge": evaluate(ridge, cols),
    "random forest": evaluate(rf, cols),
    "hist gradient boosting": evaluate(hgb, cols),
    "svr (8k rows)": evaluate(svr, cols, train_cap=8000),
})