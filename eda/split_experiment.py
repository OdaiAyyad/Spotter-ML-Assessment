import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

train = pd.read_csv("data/train_test.csv", parse_dates=["date"])
train["rate_per_mile"] = train["posted_rate"] / train["distance"]
# labels outside 1.2-4.0 $/mi treated as corrupted; excluded from training only
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
    model = HistGradientBoostingRegressor(random_state=0)
    model.fit(make_features(tr_fit, drop_cols), tr_fit["posted_rate"])
    pred = model.predict(make_features(te, drop_cols))
    ok = ~te["is_bad"].values
    y = te["posted_rate"].values
    report(y[ok], pred[ok], label + " [clean test]")
    report(y, pred, label + " [all test]")


print("random vs time split:")
tr_a, te_a = train_test_split(train, test_size=0.2, random_state=42)
cutoff = pd.Timestamp("2025-09-01")
tr_b, te_b = train[train["date"] < cutoff], train[train["date"] >= cutoff]
for drop in (False, True):
    fit_eval(tr_a, te_a, f"random 80/20 | drop_bad_train={drop}", drop_bad_train=drop)
    fit_eval(tr_b, te_b, f"time split   | drop_bad_train={drop}", drop_bad_train=drop)

# time split is honest; random split leaks future market conditions
print("\nmarket feature availability (December has none):")
fit_eval(tr_b, te_b, "real market features")
fit_eval(tr_b, te_b, "no market features", drop_cols=("market_index", "quote_signal"))
recent = tr_b[tr_b["date"] >= tr_b["date"].max() - pd.Timedelta(days=6)]
te_c = te_b.assign(market_index=recent["market_index"].mean(),
                   quote_signal=recent["quote_signal"].mean())
fit_eval(tr_b, te_c, "frozen last-7-day values")
