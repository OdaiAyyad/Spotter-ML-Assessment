import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import HistGradientBoostingRegressor

train = pd.read_csv("data/train_test.csv", parse_dates=["date"])
train["rpm"] = train["posted_rate"] / train["distance"]
train["is_bad"] = (train["rpm"] < 1.2) | (train["rpm"] > 4.0)
eq = pd.get_dummies(train["equipment"]).astype(int)
EQ = list(eq.columns)
train = train.join(eq)
MONTHS = pd.period_range("2025-04", "2025-10", freq="M")
BASE = ["distance", "weight"] + EQ
WITH = BASE + ["market_index"]


def fold_data(m):
    tr = train[(train["date"] < m.start_time) & ~train["is_bad"]]
    te = train[(train["date"] >= m.start_time) & (train["date"] < (m + 1).start_time)]
    return tr, te


def fit_predict(tr, te, cols, seed=0, **params):
    model = HistGradientBoostingRegressor(random_state=seed, **params)
    model.fit(tr[cols], tr["rpm"])
    return model.predict(te[cols]) * te["distance"].values


# signed error reveals whether market_index pushes predictions wrong direction
print("signed error (pred - actual) on clean rows:")
rows = []
for m in MONTHS:
    tr, te = fold_data(m)
    te = te[~te["is_bad"]]
    row = {"month": str(m), "mi_test": te["market_index"].mean(), "mi_train": tr["market_index"].mean()}
    for name, cols in [("base", BASE), ("+mkt", WITH)]:
        err = fit_predict(tr, te, cols) - te["posted_rate"].values
        row[f"bias_{name}"], row[f"MAE_{name}"] = err.mean(), np.abs(err).mean()
    rows.append(row)
print(pd.DataFrame(rows).set_index("month").round(2))

print("\nHGB tuning, mean and worst-month MAE:")
res = []
for lr, leaves, msl in product([0.05, 0.1], [15, 31], [20, 100]):
    for fname, cols in [("base", BASE), ("+mkt", WITH)]:
        maes = []
        for m in MONTHS:
            tr, te = fold_data(m)
            te = te[~te["is_bad"]]
            p = fit_predict(tr, te, cols, learning_rate=lr, max_leaf_nodes=leaves,
                            min_samples_leaf=msl, max_iter=200, early_stopping=False)
            maes.append(np.abs(p - te["posted_rate"].values).mean())
        res.append({"lr": lr, "leaves": leaves, "min_leaf": msl, "features": fname,
                    "MAE_mean": np.mean(maes), "MAE_worst": np.max(maes)})
print(pd.DataFrame(res).sort_values("MAE_mean").round(1).to_string(index=False))

print("\nerror analysis by distance / equipment / missing weight:")
parts = []
for m in MONTHS:
    tr, te = fold_data(m)
    te = te[~te["is_bad"]].copy()
    te["pred"] = fit_predict(tr, te, BASE)
    parts.append(te)
pool = pd.concat(parts)
pool["bias"] = pool["pred"] - pool["posted_rate"]
pool["abs_err"] = pool["bias"].abs()
pool["ape"] = pool["abs_err"] / pool["posted_rate"] * 100
pool["dist_bucket"] = pd.cut(pool["distance"], [0, 300, 600, 1000, 1500, 2000, 4000])
pool["weight_missing"] = pool["weight"].isna()
for col in ["dist_bucket", "equipment", "weight_missing"]:
    print(pool.groupby(col, observed=True).agg(n=("ape", "size"), bias=("bias", "mean"),
          abs_err=("abs_err", "mean"), ape=("ape", "mean")).round(2), "\n")

# quote_signal regime detection without labels; C derived from training-period observations
print("quote_signal regime backtest (diagnostic only):")
C = 4.15
rows = []
for m in MONTHS:
    tr, te = fold_data(m)
    rpm_hat = fit_predict(tr, te, BASE) / te["distance"].values
    qs = te["quote_signal"].values
    d_plus = np.median(np.abs(qs - rpm_hat))
    d_minus = np.median(np.abs((C - qs) - rpm_hat))
    detected = "noise" if min(d_plus, d_minus) > 0.15 else ("+" if d_plus < d_minus else "-")
    ok = ~te["is_bad"].values
    t = te[ok]
    sp, sm = (t["quote_signal"] - t["rpm"]).std(), (t["quote_signal"] + t["rpm"]).std()
    truth = "noise" if min(sp, sm) > 0.15 else ("+" if sp < sm else "-")
    y = te["posted_rate"].values
    dec_rpm = {"+": qs, "-": C - qs, "noise": rpm_hat}[detected]
    rows.append({"month": str(m), "detected": detected, "truth": truth,
                 "MAE_base": np.abs(rpm_hat * te["distance"].values - y)[ok].mean(),
                 "MAE_decoded": np.abs(dec_rpm * te["distance"].values - y)[ok].mean()})
print(pd.DataFrame(rows).round(2).to_string(index=False))
