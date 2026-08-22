import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent  # project root, works on Windows/Mac/Linux

m = json.load(open(BASE / "reports" / "metrics.json"))
curve = m["validation_cost_curve"]
thresholds = [c["threshold"] for c in curve]
costs = [c["cost_inr"] for c in curve]
fn = [c["missed_fraud"] for c in curve]
fp = [c["false_positives"] for c in curve]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

ax1.plot(thresholds, costs, color="#c0392b", linewidth=2)
ax1.axvline(m["chosen_threshold"], color="#2c3e50", linestyle="--", linewidth=1,
            label=f"chosen threshold = {m['chosen_threshold']}")
ax1.axvline(0.5, color="#7f8c8d", linestyle=":", linewidth=1, label="naive 0.5 cutoff")
ax1.set_ylabel("Estimated cost (₹) on validation set")
ax1.set_title("Cost vs. decision threshold")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(thresholds, fn, label="Missed fraud (FN)", color="#e74c3c")
ax2.plot(thresholds, fp, label="Wrongly blocked (FP)", color="#3498db")
ax2.axvline(m["chosen_threshold"], color="#2c3e50", linestyle="--", linewidth=1)
ax2.set_xlabel("Decision threshold")
ax2.set_ylabel("Count")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(BASE / "reports" / "cost_curve.png", dpi=140)
print("saved reports/cost_curve.png")
