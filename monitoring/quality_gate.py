
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MONITOR_DIR = BASE_DIR / "monitoring"

# Load model comparison results
with open(MONITOR_DIR / "model_comparison.json", "r") as file:
    results = json.load(file)

champion = results["champion"]
candidate = results["candidate"]

# Quality gate: candidate must improve MAE
# by at least 1% to be eligible for promotion
required_improvement = 0.01

improvement = (
    (champion["MAE"] - candidate["MAE"])
    / champion["MAE"]
)

print("QUALITY GATE EVALUATION")
print("-----------------------")
print(f"Champion MAE: {champion['MAE']}")
print(f"Candidate MAE: {candidate['MAE']}")
print(f"Required improvement: {required_improvement * 100:.1f}%")
print(f"Actual improvement: {improvement * 100:.2f}%")

if improvement >= required_improvement:
    decision = "APPROVED"
    reason = "Candidate achieved at least 1% MAE improvement."
else:
    decision = "REJECTED"
    reason = "Candidate did not achieve the required MAE improvement."

# Record decision without overwriting champion model
decision_report = {
    "decision": decision,
    "reason": reason,
    "champion_MAE": champion["MAE"],
    "candidate_MAE": candidate["MAE"],
    "required_improvement_percent": 1.0,
    "actual_improvement_percent": round(improvement * 100, 2),
    "champion_retained": decision == "REJECTED"
}

with open(MONITOR_DIR / "quality_gate_report.json", "w") as file:
    json.dump(decision_report, file, indent=4)

print("\nQUALITY GATE DECISION:", decision)
print("Reason:", reason)
print("Existing champion model retained.")
print("\nDecision report saved successfully!")
print("Task D completed successfully!")
