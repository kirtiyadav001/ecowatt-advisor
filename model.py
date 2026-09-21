import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(42)

# ---- 1. Synthetic household electricity usage dataset ----
# Simulates monthly readings for 300 households over 12 months
n_households = 300
months = 12
rows = []
for h in range(n_households):
    household_size = np.random.randint(1, 6)
    ac_units = np.random.randint(0, 3)
    has_geyser = np.random.choice([0, 1], p=[0.4, 0.6])
    base_load = 60 + household_size * 25 + ac_units * 55 + has_geyser * 20
    for m in range(1, months + 1):
        # seasonal effect: summer (Apr-Jun) higher AC use
        seasonal = 1.35 if m in [4, 5, 6] else (1.15 if m in [3, 7, 8, 9] else 0.9)
        avg_temp = 22 + 10 * np.sin((m - 3) / 12 * 2 * np.pi) + np.random.normal(0, 1.5)
        noise = np.random.normal(0, 12)
        consumption = base_load * seasonal + ac_units * avg_temp * 0.8 + noise
        rows.append({
            "household_id": h,
            "month": m,
            "household_size": household_size,
            "ac_units": ac_units,
            "has_geyser": has_geyser,
            "avg_temp_c": round(avg_temp, 1),
            "consumption_kwh": max(20, round(consumption, 1)),
        })

df = pd.DataFrame(rows)
df.to_csv("household_energy_data.csv", index=False)

# ---- 2. Forecasting model: predict next month's consumption ----
df_sorted = df.sort_values(["household_id", "month"])
df_sorted["prev_month_kwh"] = df_sorted.groupby("household_id")["consumption_kwh"].shift(1)
df_model = df_sorted.dropna(subset=["prev_month_kwh"])

features = ["household_size", "ac_units", "has_geyser", "avg_temp_c", "prev_month_kwh"]
X = df_model[features]
y = df_model["consumption_kwh"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"MAE: {mae:.2f} kWh")
print(f"R2: {r2:.3f}")

# feature importance
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importance:")
print(importance)

# ---- 3. Chart: Actual vs Predicted ----
plt.figure(figsize=(7, 5))
plt.scatter(y_test, preds, alpha=0.5, color="#2E7D32", edgecolors="white", linewidth=0.3, s=45)
lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
plt.plot(lims, lims, "--", color="#888888", linewidth=1)
plt.xlabel("Actual Monthly Consumption (kWh)")
plt.ylabel("Predicted Monthly Consumption (kWh)")
plt.title("EcoWatt Advisor: Predicted vs Actual Household Electricity Use")
plt.text(lims[0], lims[1]*0.95, f"MAE = {mae:.1f} kWh   |   R² = {r2:.2f}", fontsize=10, color="#333333")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=200)
plt.close()

# ---- 4. Feature importance chart ----
plt.figure(figsize=(7, 4.5))
importance_sorted = importance.sort_values()
labels = {
    "prev_month_kwh": "Previous month's usage",
    "avg_temp_c": "Average temperature",
    "ac_units": "Number of AC units",
    "household_size": "Household size",
    "has_geyser": "Has water geyser",
}
plt.barh([labels[i] for i in importance_sorted.index], importance_sorted.values, color="#2E7D32")
plt.xlabel("Relative importance")
plt.title("What Drives Predicted Electricity Consumption")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=200)
plt.close()

# ---- 5. Sample forecast output for 3 households (for advisory demo) ----
sample = df_model.groupby("household_id").tail(1).sample(3, random_state=7)
sample_preds = model.predict(sample[features])
sample_out = sample[["household_id", "household_size", "ac_units", "has_geyser", "avg_temp_c", "prev_month_kwh"]].copy()
sample_out["predicted_next_month_kwh"] = np.round(sample_preds, 1)
sample_out.to_csv("sample_forecasts.csv", index=False)
print("\nSample forecasts:")
print(sample_out)
