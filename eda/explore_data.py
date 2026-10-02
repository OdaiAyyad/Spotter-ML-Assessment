import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)

train = pd.read_csv("data/train_test.csv")
val = pd.read_csv("data/validation.csv")
dec = pd.read_csv("data/december_chart_inputs.csv")

for name, df in [("train", train), ("validation", val), ("december", dec)]:
    print(f"\n{name}")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head())
    print(df.dtypes)
    print("missing:\n", df.isna().sum())

for df in (train, val, dec):
    df["date"] = pd.to_datetime(df["date"])

print(train["date"].min(), train["date"].max())
print(val["date"].min(), val["date"].max())

print(train["posted_rate"].describe())
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(train["posted_rate"], bins=60, ax=ax[0])
sns.boxplot(x=train["posted_rate"], ax=ax[1])
sns.scatterplot(data=train.sample(3000, random_state=0), x="distance", y="posted_rate",
                hue="equipment", ax=ax[2], s=10)
import os
os.makedirs("eda/output", exist_ok=True)
plt.savefig("eda/output/eda_target.png")

monthly = train.groupby(train["date"].dt.to_period("M"))[["market_index", "quote_signal", "posted_rate"]].mean()
print(monthly)

train["rate_per_mile"] = train["posted_rate"] / train["distance"]

print(train["rate_per_mile"].describe())
print(train["rate_per_mile"].quantile([0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]))
print(train[["rate_per_mile", "quote_signal", "market_index", "weight", "distance"]].corr())

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(train["rate_per_mile"], bins=80, ax=ax[0])
sns.scatterplot(data=train.sample(3000, random_state=0), x="quote_signal", y="rate_per_mile", s=8, ax=ax[1])
sns.boxplot(data=train, x="equipment", y="rate_per_mile", ax=ax[2])
ax[2].set_ylim(0, 10)
plt.tight_layout()
plt.savefig("eda/output/eda_rate_per_mile.png")

# each pickup->delivery pair has varying distances due to zip-level routing
print(train.groupby(["pickup", "delivery"])["distance"].nunique().describe())

# outlier plateau check — count stabilises near 675 between 1.2/4.0
for lo, hi in [(1.0, 4.0), (1.2, 3.5), (1.5, 3.3), (1.6, 3.2)]:
    n = ((train["rate_per_mile"] < lo) | (train["rate_per_mile"] > hi)).sum()
    print(lo, hi, n)

# bad-row profile
mask_bad = (train["rate_per_mile"] < 1.2) | (train["rate_per_mile"] > 4)
bad, clean = train[mask_bad], train[~mask_bad]
print(len(bad))
print(bad["equipment"].value_counts(normalize=True))
print(bad["date"].dt.month.value_counts().sort_index())
print(bad[["distance", "weight", "market_index", "quote_signal"]].describe())

cols = ["rate_per_mile", "quote_signal", "market_index", "weight", "distance"]
print(clean[cols].corr())
print(clean[cols].corr(method="spearman"))
print(clean.groupby("equipment")["rate_per_mile"].agg(["mean", "std"]))


def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


train["geo_dist"] = haversine(train.pickup_lat, train.pickup_lon, train.delivery_lat, train.delivery_lon)
print((train["distance"] / train["geo_dist"]).describe())
print(train.groupby("pickup")[["pickup_lat", "pickup_lon"]].nunique().describe())
print(train.groupby(["pickup", "delivery"])["distance"].agg(["min", "max", "std"]).describe())

clean = train[~((train["rate_per_mile"] < 1.2) | (train["rate_per_mile"] > 4.0))].copy()

print(train.head(20)[["date", "distance", "rate_per_mile", "quote_signal"]])
print((clean["quote_signal"] - clean["rate_per_mile"]).describe())
# quote_signal flips sign across months; unstable over time
print(clean.groupby(clean["date"].dt.month).apply(lambda g: g["quote_signal"].corr(g["rate_per_mile"])))
print(clean.groupby("equipment")["rate_per_mile"].agg(["mean", "std"]))

circ = train["distance"] / train["geo_dist"]
print(pd.crosstab(train["rate_per_mile"].between(1.2, 4.0), circ > 1.5))

daily = train.groupby("date")["market_index"].agg(["mean", "std"])
print(daily.describe())

val["date"] = pd.to_datetime(val["date"])
print(val.loc[val["date"] >= "2025-12-01", "market_index"].describe())
