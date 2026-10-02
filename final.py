import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

train = pd.read_csv("data/train_test.csv", parse_dates=["date"])
val = pd.read_csv("data/validation.csv", parse_dates=["date"])
dec_raw = pd.read_csv("data/december_chart_inputs.csv")        # keep original strings for output
train["rpm"] = train["posted_rate"] / train["distance"]
train["is_bad"] = (train["rpm"] < 1.2) | (train["rpm"] > 4.0)
EQ = sorted(train["equipment"].unique())

def make_X(df):
    X = df[["distance", "weight"]].astype(float).copy()
    for e in EQ:
        X[e] = (df["equipment"] == e).astype(int)
    return X

def fit(tr, seed=0):
    m = HistGradientBoostingRegressor(random_state=seed)
    m.fit(make_X(tr), tr["rpm"])
    return m

# ---------- 1. Calendar check on out-of-fold errors ----------
MONTHS = pd.period_range("2025-04", "2025-10", freq="M")
parts = []
for m in MONTHS:
    tr = train[(train["date"] < m.start_time) & ~train["is_bad"]]
    te = train[(train["date"] >= m.start_time) & (train["date"] < (m + 1).start_time) & ~train["is_bad"]].copy()
    te["pred"] = fit(tr).predict(make_X(te)) * te["distance"].values
    parts.append(te)
pool = pd.concat(parts)
pool["ratio"] = pool["posted_rate"] / pool["pred"] - 1                    # + means actual above prediction
pool["rel"] = pool["ratio"] - pool.groupby(pool["date"].dt.to_period("M"))["ratio"].transform("mean")
print("\n=== 1a. Error by weekday (relative to month average) ===")
print(pool.groupby(pool["date"].dt.day_name())["rel"].agg(["mean", "size"]).round(4))
HOL = pd.to_datetime(["2025-05-26", "2025-07-04", "2025-09-01"])
pool["hol_gap"] = pool["date"].apply(lambda d: min(((d - h).days for h in HOL), key=abs))
print("\n=== 1b. Days from nearest holiday (-4..+4) ===")
print(pool[pool["hol_gap"].abs() <= 4].groupby("hol_gap")["rel"].agg(["mean", "size"]).round(4))
print("\n=== 1c. Is there a shared daily factor? ===")
print("std of daily mean error :", round(pool.groupby("date")["rel"].mean().std(), 4))
print("expected if pure noise  :", round(pool["rel"].std() / np.sqrt(pool.groupby("date").size().mean()), 4))

# ---------- 2. Final model (base features) + main deliverables ----------
final_model = fit(train[~train["is_bad"]])
val["rpm_hat"] = final_model.predict(make_X(val))
val["pred_main"] = val["rpm_hat"] * val["distance"]
assert val["pred_main"].notna().all() and (val["pred_main"] > 0).all()
pd.DataFrame({"load_id": val["load_id"], "predicted_rate": val["pred_main"].round(2)}) \
  .to_csv("validation_predictions_main.csv", index=False)

dec_out = dec_raw.copy()
dec_out["predicted_rate"] = (final_model.predict(make_X(dec_raw)) * dec_raw["distance"].values).round(2)
dec_out.to_csv("december_predictions_main.csv", index=False)
print("\n=== 2. December predictions ===")
print(dec_out["predicted_rate"].describe().round(2))

# ---------- 3. quote_signal regimes in the validation file (no labels used) ----------
C = 4.15
def day_regime(g):
    d_plus = np.median(np.abs(g["quote_signal"] - g["rpm_hat"]))
    d_minus = np.median(np.abs((C - g["quote_signal"]) - g["rpm_hat"]))
    if min(d_plus, d_minus) > 0.15:
        return "noise"
    return "+" if d_plus < d_minus else "-"
reg = val.groupby("date")[["quote_signal", "rpm_hat"]].apply(day_regime)
print("\n=== 3. Detected regime per day, by month ===")
print(reg.groupby(reg.index.to_period("M")).value_counts().unstack(fill_value=0))
val["regime"] = val["date"].map(reg)
dec_rpm = np.where(val["regime"] == "+", val["quote_signal"],
          np.where(val["regime"] == "-", C - val["quote_signal"], val["rpm_hat"]))
val["pred_decoded"] = np.clip(dec_rpm, 0.5, None) * val["distance"]
pd.DataFrame({"load_id": val["load_id"], "predicted_rate": val["pred_decoded"].round(2)}) \
  .to_csv("validation_predictions_decoded.csv", index=False)
print("median |decoded - main| / main:", round((val["pred_decoded"] / val["pred_main"] - 1).abs().median(), 4))