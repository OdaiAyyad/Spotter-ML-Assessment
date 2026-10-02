"""Fits the chosen model on all clean training rows and writes submission files."""
import pandas as pd
from common import RPM_LOW, RPM_HIGH, build_features, make_coord_lookup, make_model, feature_cols

CHOSEN = "+dow+geo"

train_raw = pd.read_csv("data/train_test.csv", parse_dates=["date"])
val_raw   = pd.read_csv("data/validation.csv",  parse_dates=["date"])
dec_raw   = pd.read_csv("data/december_chart_inputs.csv")

dec_dates = pd.to_datetime(dec_raw["date"])

train_raw["rpm"] = train_raw["posted_rate"] / train_raw["distance"]
train_raw["is_bad"] = (train_raw["rpm"] < RPM_LOW) | (train_raw["rpm"] > RPM_HIGH)

EQ = sorted(train_raw["equipment"].unique())
coord_lookup = make_coord_lookup(train_raw)

dec_tmp = dec_raw.copy()
dec_tmp["date"] = dec_dates

train = build_features(train_raw, EQ, coord_lookup=coord_lookup)
val   = build_features(val_raw,   EQ, coord_lookup=coord_lookup)
dec   = build_features(dec_tmp,   EQ, coord_lookup=coord_lookup)

COLS = feature_cols(EQ)[CHOSEN]

clean = train[~train["is_bad"]]
model = make_model()
model.fit(clean[COLS], clean["rpm"])

val["pred"] = model.predict(val[COLS]) * val["distance"]
assert val["pred"].notna().all() and (val["pred"] > 0).all() and len(val) == 12000

pd.DataFrame({
    "load_id": val_raw["load_id"],
    "predicted_rate": val["pred"].round(2),
}).to_csv("validation_predictions.csv", index=False)

dec_out = dec_raw.copy()
dec_out["predicted_rate"] = (model.predict(dec[COLS]) * dec["distance"]).round(2)
dec_out.to_csv("december_predictions.csv", index=False)

print("validation_predictions.csv written:", len(val), "rows")
print("december_predictions.csv written:", len(dec_out), "rows")
print("December rates:")
print(dec_out.assign(weekday=dec_dates.dt.day_name())[["date", "weekday", "predicted_rate"]].to_string(index=False))
