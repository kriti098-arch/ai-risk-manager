"""SHAP explainability report -- which features actually drive the model's
fraud calls, matching ATDS's SHAP dashboard widget."""
from pathlib import Path
import joblib
import json
import shap
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import FEATURES

BASE = Path(__file__).resolve().parent.parent  # project root, works on Windows/Mac/Linux

df = pd.read_csv(BASE / "data" / "transactions.csv")
rf = joblib.load(BASE / "models" / "rf_classifier.pkl")

sample = df[FEATURES].sample(1500, random_state=42)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(sample)

# shap_values for binary RF: list [class0, class1] or (n,features,2) depending on version
sv = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]

mean_abs = pd.Series(abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False)
print(mean_abs)

mean_abs.to_json(BASE / "reports" / "shap_feature_importance.json", indent=2)

plt.figure()
shap.summary_plot(sv, sample, feature_names=FEATURES, show=False)
plt.tight_layout()
plt.savefig(BASE / "reports" / "shap_summary.png", dpi=140)
print("saved reports/shap_summary.png")
