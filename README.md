# EcoWatt Advisor

**AI-Powered Electricity Forecasting & Energy-Saving Advisor for Households**

Submitted for the 1M1B AI for Sustainability Virtual Internship (July–Sep 2026), in collaboration with IBM SkillsBuild & AICTE.

**SDG Alignment:** SDG 7 (Affordable and Clean Energy), secondary SDG 13 (Climate Action)

## Problem

Most households receive a single bimonthly electricity bill with no breakdown of what drives their usage, so overconsumption is only discovered after a high bill arrives. There is no early-warning system that helps families understand or reduce avoidable energy use before it happens.

## Solution

EcoWatt Advisor combines two AI components:

1. **Forecasting model** — a Random Forest regression model (`model.py`) trained on household usage data (household size, appliances, local temperature, prior month's consumption) that predicts next month's electricity consumption. On a simulated dataset of 300 households over 12 months, the model achieves **R² = 0.91** with a mean absolute error of ~21 kWh.
2. **Advisory layer** — a prompt-engineered template that turns the forecast into a plain-language, personalized recommendation (one-sentence explanation + two concrete, low-effort actions).

## Repository Contents

| File | Description |
|---|---|
| `model.py` | Full pipeline: synthetic data generation, model training, evaluation, chart generation |
| `household_energy_data.csv` | Simulated dataset: 300 households × 12 months of usage data |
| `sample_forecasts.csv` | Example model outputs for 3 sample households |
| `charts/actual_vs_predicted.png` | Model performance: predicted vs. actual consumption |
| `charts/feature_importance.png` | What drives predicted electricity consumption |
| `EcoWatt_Advisor_1M1B_Project.pptx` | Full project presentation (problem, solution, prototype, responsible AI, impact) |

## Tech Stack

Python (pandas, scikit-learn, matplotlib) for the forecasting model; prompt engineering for the advisory layer.

## Responsible AI

- **Fairness:** model uses only usage-pattern and appliance data — no demographic or income proxies
- **Transparency:** feature importance is surfaced alongside every forecast
- **Ethics:** advisory tips are optional suggestions, never automated appliance control
- **Privacy:** only aggregate usage metrics are used; no personal identity or billing data required

## Author

Kirti Yadav — MSc Economics, Gokhale Institute of Politics and Economics (GIPE), Pune
