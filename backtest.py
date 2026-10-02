"""Rolling monthly backtest, candidate comparison, and unseen-city simulation."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from common import (
    RPM_LOW, RPM_HIGH, build_features, make_coord_lookup,
    make_model, feature_cols,
)

train_raw = pd.read_csv("data/train_test.csv", parse_dates=["date"])
val_raw   = pd.read_csv("data/validation.csv",  parse_dates=["date"])
dec_raw   = pd.read_csv("data/december_chart_inputs.csv")

train_raw["rpm"] = train_raw["posted_rate"] / train_raw["distance"]
train_raw["is_bad"] = (train_raw["rpm"] < RPM_LOW) | (train_raw["rpm"] > RPM_HIGH)

EQ = sorted(train_raw["equipment"].unique())
coord_lookup = make_coord_lookup(train_raw)
train = build_features(train_raw, EQ, coord_lookup=coord_lookup)

MONTHS = pd.period_range("2025-04", "2025-10", freq="M")
FCOLS = feature_cols(EQ)
CHOSEN_COLS = FCOLS["+dow+geo"]


def folds():
    for m in MONTHS:
        tr = train[(train["date"] < m.start_time) & ~train["is_bad"]]
        te = train[(train["date"] >= m.start_time) & (train["date"] < (m + 1).start_time) & ~train["is_bad"]]
        yield str(m), tr, te


def eval_model(make_fn, cols, train_cap=None):
    rows = []
    for name, tr, te in folds():
        if train_cap and len(tr) > train_cap:
            tr = tr.sample(train_cap, random_state=0)
        m = make_fn()
        m.fit(tr[cols], tr["rpm"])
        pred = m.predict(te[cols]) * te["distance"].values
        err = np.abs(pred - te["posted_rate"].values)
        rows.append({"month": name, "MAE": err.mean(),
                     "MAPE": (err / te["posted_rate"].values).mean() * 100})
    return pd.DataFrame(rows).set_index("month")


def unseen_simulation(seeds=range(5)):
    """Simulate unseen cities across random held-out city draws."""
    clean = train[~train["is_bad"]].copy()
    all_cities = sorted(set(clean["pickup"]) | set(clean["delivery"]))
    all_city_dt = pd.CategoricalDtype(categories=all_cities, ordered=False)
    clean["pickup_cat"] = clean["pickup"].astype(all_city_dt)
    clean["delivery_cat"] = clean["delivery"].astype(all_city_dt)
    clean["pickup_c"] = clean["pickup_cat"].cat.codes
    clean["delivery_c"] = clean["delivery_cat"].cat.codes

    base = ["distance", "weight"] + EQ + ["dow"]
    coords = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
    variants = {
        "(a) no_geo": (base, None),
        "(b) int_codes": (base + ["pickup_c", "delivery_c"], None),
        "(c) cat_dtype": (base + ["pickup_cat", "delivery_cat"], "from_dtype"),
        "(d) coords": (base + coords, None),
        "(e) coords+cat": (base + coords + ["pickup_cat", "delivery_cat"], "from_dtype"),
    }

    records = {v: {"MAE": [], "MAPE": []} for v in variants}
    for s in seeds:
        rng = np.random.RandomState(s)
        held = set(rng.choice(all_cities, 8, replace=False))
        tr = clean[(clean["date"] < "2025-09-01") & ~clean["pickup"].isin(held) & ~clean["delivery"].isin(held)]
        te = clean[(clean["date"] >= "2025-09-01") & (clean["pickup"].isin(held) | clean["delivery"].isin(held))]
        y = te["posted_rate"].values
        d = te["distance"].values

        for name, (cols, cat_mode) in variants.items():
            kwargs = {"random_state": 0}
            if cat_mode:
                kwargs["categorical_features"] = cat_mode
            m = HistGradientBoostingRegressor(**kwargs).fit(tr[cols], tr["rpm"])
            err = np.abs(m.predict(te[cols]) * d - y)
            records[name]["MAE"].append(err.mean())
            records[name]["MAPE"].append((err / y).mean() * 100)
    return records


def split_experiment(seeds=range(5)):
    """Compare time split vs random split for FINAL and OLD feature sets on Sep-Oct."""
    clean = train[~train["is_bad"]].copy()
    before = clean[clean["date"] < "2025-09-01"]
    sepoct = clean[(clean["date"] >= "2025-09-01") & (clean["date"] <= "2025-10-31")]

    final_cols = ["distance", "weight"] + EQ + ["dow", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
    old_cols = ["distance", "weight"] + EQ + ["market_index", "quote_signal"]
    fsets = {"FINAL": final_cols, "OLD": old_cols}

    from sklearn.model_selection import train_test_split

    results = {"time_split": {}, "random_split": {}}
    # (a) time split
    for fname, cols in fsets.items():
        m = HistGradientBoostingRegressor(random_state=0).fit(before[cols], before["rpm"])
        pred = m.predict(sepoct[cols]) * sepoct["distance"].values
        y = sepoct["posted_rate"].values
        err = np.abs(pred - y)
        results["time_split"][fname] = {
            "MAE": float(err.mean()),
            "MAPE": float((err / y).mean() * 100),
        }

    # (b) random split (seeds 0-4)
    for fname, cols in fsets.items():
        maes, mapes = [], []
        for s in seeds:
            tr_so, te_so = train_test_split(sepoct, test_size=0.20, random_state=s)
            tr = pd.concat([before, tr_so])
            m = HistGradientBoostingRegressor(random_state=0).fit(tr[cols], tr["rpm"])
            pred = m.predict(te_so[cols]) * te_so["distance"].values
            y = te_so["posted_rate"].values
            err = np.abs(pred - y)
            maes.append(err.mean())
            mapes.append((err / y).mean() * 100)
        results["random_split"][fname] = {
            "MAE": float(np.mean(maes)),
            "MAPE": float(np.mean(mapes)),
        }
    return results


def ridge(): return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
def rf():    return make_pipeline(SimpleImputer(strategy="median"),
                                  RandomForestRegressor(n_estimators=100, min_samples_leaf=3,
                                                        n_jobs=-1, random_state=0))
def svr():   return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVR(C=3.0))


class MedianRPM:
    def fit(self, X, y): self.m = float(np.median(y)); return self
    def predict(self, X): return np.full(len(X), self.m)


if __name__ == "__main__":
    print("rolling backtest, MAE by month — feature sets:")
    mae = pd.DataFrame({k: eval_model(make_model, c)["MAE"] for k, c in FCOLS.items()})
    print(mae.round(1))
    print("mean:", mae.mean().round(1).to_dict())

    print("\nrolling backtest, MAE by month — candidate models (+dow+geo features):")
    candidates = {
        "naive median": (MedianRPM, CHOSEN_COLS, None),
        "ridge":        (ridge,     CHOSEN_COLS, None),
        "random forest":(rf,        CHOSEN_COLS, None),
        "hgb":          (make_model,CHOSEN_COLS, None),
        "svr (8k)":     (svr,       CHOSEN_COLS, 8000),
    }
    results = {name: eval_model(fn, cols, cap) for name, (fn, cols, cap) in candidates.items()}
    mae_c = pd.DataFrame({k: v["MAE"] for k, v in results.items()})
    mape_c = pd.DataFrame({k: v["MAPE"] for k, v in results.items()})
    print(mae_c.round(1))
    print("MAE mean: ", mae_c.mean().round(1).to_dict())
    print("MAPE mean:", mape_c.mean().round(2).to_dict())
