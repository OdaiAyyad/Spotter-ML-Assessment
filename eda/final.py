import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from pandas.tseries.holiday import USFederalHolidayCalendar

train = pd.read_csv("data/train_test.csv", parse_dates=["date"])
val = pd.read_csv("data/validation.csv", parse_dates=["date"])
dec_raw = pd.read_csv("data/december_chart_inputs.csv")
dec = dec_raw.copy()
dec["date"] = pd.to_datetime(dec["date"])
train["rpm"] = train["posted_rate"] / train["distance"]
train["is_bad"] = (train["rpm"] < 1.2) | (train["rpm"] > 4.0)
EQ = sorted(train["equipment"].unique())
HOLS = USFederalHolidayCalendar().holidays(start="2024-12-01", end="2026-02-28")


def prep(df):
    df = df.copy()
    for e in EQ:
        df[e] = (df["equipment"] == e).astype(int)
    df["weight"] = df["weight"].astype(float)
    df["dow"] = df["date"].dt.dayofweek
    df["dom"] = df["date"].dt.day
    gap = {}
    for u in df["date"].drop_duplicates():
        diffs = np.asarray((u - HOLS).days)
        gap[u] = int(diffs[np.abs(diffs).argmin()])
    df["hol_gap"] = df["date"].map(gap).clip(-7, 7)
    return df


train, val, dec = prep(train), prep(val), prep(dec)
MONTHS = pd.period_range("2025-04", "2025-10", freq="M")
BASE = ["distance", "weight"] + EQ
# holiday-gap feature hurts out of time (overfits holiday patterns in training period)
SETS = {"base": BASE, "+dow": BASE + ["dow"], "+dow+hol": BASE + ["dow", "hol_gap"],
        "+dow+hol+dom": BASE + ["dow", "hol_gap", "dom"]}


def fold_data(m):
    tr = train[(train["date"] < m.start_time) & ~train["is_bad"]]
    te = train[(train["date"] >= m.start_time) & (train["date"] < (m + 1).start_time)]
    return tr, te


def fit_predict(tr, te, cols, seed=0):
    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(tr[cols], tr["rpm"])
    return model.predict(te[cols])


print("rolling backtest, MAE by month:")
mae, mape = {}, {}
for name, cols in SETS.items():
    a, b = [], []
    for m in MONTHS:
        tr, te = fold_data(m)
        te = te[~te["is_bad"]]
        err = np.abs(fit_predict(tr, te, cols) * te["distance"].values - te["posted_rate"].values)
        a.append(err.mean())
        b.append((err / te["posted_rate"].values).mean() * 100)
    mae[name], mape[name] = a, b
mae = pd.DataFrame(mae, index=[str(m) for m in MONTHS])
print(mae.round(1))
print("MAE mean :", mae.mean().round(1).to_dict())
print("MAPE mean:", pd.DataFrame(mape).mean().round(2).to_dict())
print("months better than base:", (mae.sub(mae["base"], axis=0) < 0).sum().to_dict())

# quote_signal is noise in validation (corr ~0) so decoding adds nothing there
print("\nquote_signal diagnostic:")
ref = []
for m in MONTHS:
    tr, te = fold_data(m)
    te = te.copy()
    te["rpm_hat"] = fit_predict(tr, te, BASE)
    ref.append(te)
ref = pd.concat(ref)
clean_all = train[~train["is_bad"]]
full_base = HistGradientBoostingRegressor(random_state=0).fit(clean_all[BASE], clean_all["rpm"])
val_ = val.copy()
val_["rpm_hat"] = full_base.predict(val_[BASE])


def qs_summary(df):
    g = df.groupby(df["date"].dt.to_period("M"))
    return pd.DataFrame({
        "corr(qs,rpm_hat)": g.apply(lambda x: x["quote_signal"].corr(x["rpm_hat"])),
        "qs_mean": g["quote_signal"].mean(), "qs_std": g["quote_signal"].std(),
        "rpm_hat_mean": g["rpm_hat"].mean()}).round(3)


print("training months (out-of-fold)"); print(qs_summary(ref))
print("validation months");             print(qs_summary(val_))

CHOSEN = "+dow"
cols = SETS[CHOSEN]
final = HistGradientBoostingRegressor(random_state=0).fit(clean_all[cols], clean_all["rpm"])
val["pred"] = final.predict(val[cols]) * val["distance"]
from pathlib import Path
Path("eda/output").mkdir(parents=True, exist_ok=True)
pd.DataFrame({"load_id": val["load_id"], "predicted_rate": val["pred"].round(2)}) \
  .to_csv("eda/output/validation_predictions.csv", index=False)
dec_out = dec_raw.copy()
dec_out["predicted_rate"] = (final.predict(dec[cols]) * dec["distance"]).round(2)
dec_out.to_csv("eda/output/december_predictions.csv", index=False)
print(f"\nfinal model: {CHOSEN}")
print("validation $/mile by month:\n",
      (val["pred"] / val["distance"]).groupby(val["date"].dt.to_period("M")).mean().round(3))
print(dec_out.assign(weekday=dec["date"].dt.day_name())[["date", "weekday", "predicted_rate"]].to_string(index=False))
