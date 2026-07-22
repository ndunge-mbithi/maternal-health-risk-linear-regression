import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1014  # matches published UCI Maternal Health Risk dataset size

# ---- Age: right-skewed, 10-70 (documented range) ----
age = rng.gamma(shape=4.0, scale=6.5, size=n) + 10
age = np.clip(age, 10, 70).round().astype(int)

# ---- Blood pressure: correlated with age, Systolic & Diastolic highly correlated with each other ----
# base systolic influenced by age
systolic_base = 90 + 0.55 * (age - 10) + rng.normal(0, 9, n)
systolic = np.clip(systolic_base, 70, 160).round().astype(int)

# diastolic strongly correlated with systolic (documented as the strongest pair correlation)
diastolic = np.clip(0.62 * systolic + rng.normal(15, 6, n), 49, 100).round().astype(int)

# ---- Blood Sugar (BS): right-skewed, mmol/L, weakly tied to age ----
bs = rng.gamma(shape=2.2, scale=2.6, size=n) + 6 + 0.01 * (age - 30)
bs = np.clip(bs, 6.0, 19.0).round(1)

# ---- Body temperature (F): mostly normal, weak signal (documented as not very informative) ----
body_temp = rng.normal(98.4, 1.0, n)
# small fraction of fevers
fever_idx = rng.choice(n, size=int(0.08 * n), replace=False)
body_temp[fever_idx] += rng.uniform(1.5, 3.5, len(fever_idx))
body_temp = np.clip(body_temp, 98.0, 103.0).round(1)

# ---- Heart rate: weak signal, mostly normal range ----
heart_rate = rng.normal(74, 8, n)
heart_rate = np.clip(heart_rate, 7, 90).round().astype(int)
# UCI dataset has a documented artifact of a few very-low (7 bpm) erroneous sensor readings
low_idx = rng.choice(n, size=3, replace=False)
heart_rate[low_idx] = 7

df = pd.DataFrame({
    "Age": age,
    "SystolicBP": systolic,
    "DiastolicBP": diastolic,
    "BS": bs,
    "BodyTemp": body_temp,
    "HeartRate": heart_rate,
})

# ---- Derive RiskLevel (categorical, as in the original clinically-labeled dataset) ----
# Clinical scoring loosely based on published risk factor literature (hypertension, hyperglycemia, fever, age extremes)
risk_score = (
    (df.SystolicBP >= 140).astype(int) * 2 +
    (df.SystolicBP.between(130, 139)).astype(int) * 1 +
    (df.DiastolicBP >= 90).astype(int) * 2 +
    (df.DiastolicBP.between(85, 89)).astype(int) * 1 +
    (df.BS >= 11).astype(int) * 3 +
    (df.BS.between(7.8, 10.9)).astype(int) * 1 +
    (df.BodyTemp >= 100.4).astype(int) * 2 +
    (df.Age >= 40).astype(int) * 1 +
    (df.Age <= 17).astype(int) * 1 +
    (df.HeartRate <= 10).astype(int) * 1 +
    rng.normal(0, 0.6, n)
)

df["RiskLevel"] = pd.cut(risk_score, bins=[-np.inf, 1.5, 3.5, np.inf], labels=["low risk", "mid risk", "high risk"])

out_path = "summative/linear_regression/maternal_health_risk.csv"
df.to_csv(out_path, index=False)
print(df.shape)
print(df["RiskLevel"].value_counts())
print(df.describe())
