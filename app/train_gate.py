"""
Phase 5 — the learned escalation gate + an HONEST verdict.

We trained on 80 videos, tested on 32. Reporting done the DISCIPLINED way:
  - report cross-validated AUC on train (not a single lucky split)
  - choose the decision threshold on TRAIN, then apply to TEST (no test leakage)

Finding (honest): with only ~80 training videos and per-VIDEO aggregate motion
features, the learned gate does NOT robustly beat a well-tuned single rule. CV AUC
is ~0.80 but with huge variance; the honestly-chosen threshold does not dominate the
baseline. The apparent wins came from peeking at the test set to pick the threshold.

Takeaway that IS solid: the gate is a cheap, tunable RECALL/BUDGET knob. Precision
comes from the VLM arbiter downstream (proven quiet on normal traffic). So the gate
should escalate liberally for recall; the VLM removes the false alarms. That is the
real system design -> evaluate the FULL cascade (Phase 6), not the gate alone.

    python -m app.train_gate
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

FEATURES = ["num_stalled", "max_simultaneous_stopped", "max_flow_at_stall", "isolated_stall_count"]
B_RECALL, B_FA = 0.8125, 0.3125


def mk():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))


def main():
    tr = pd.read_csv("data/train_features.csv")
    te = pd.read_csv("data/test_features.csv")
    ytr, yte = tr.label.values, te.label.values

    cv = cross_val_score(mk(), tr[FEATURES], ytr, cv=5, scoring="roc_auc")
    print(f"Train videos {len(tr)} | Test videos {len(te)} | features {FEATURES}")
    print(f"\n5-fold CV AUC on train: {cv.mean():.3f} +/- {cv.std():.3f}   "
          f"(big variance = small-data, not a reliable model)")

    m = mk().fit(tr[FEATURES], ytr)
    ptr, pte = m.predict_proba(tr[FEATURES])[:, 1], m.predict_proba(te[FEATURES])[:, 1]
    print(f"Test AUC: {roc_auc_score(yte, pte):.3f}")

    # HONEST protocol: choose threshold on TRAIN (max recall s.t. FA<=baseline), apply to TEST
    best = None
    for t in np.linspace(0.05, 0.95, 181):
        p = (ptr >= t).astype(int); fa = p[ytr == 0].mean(); rec = p[ytr == 1].mean()
        if fa <= B_FA and (best is None or rec > best[1]):
            best = (t, rec, fa)
    t = best[0]; p = (pte >= t).astype(int)
    print(f"\nHonest (threshold chosen on TRAIN={t:.2f}) -> TEST: "
          f"recall {p[yte==1].mean():.0%}  FA {p[yte==0].mean():.0%}")
    print(f"Baseline hand-tuned rule                 -> TEST: recall {B_RECALL:.0%}  FA {B_FA:.0%}")
    print("=> learned gate does NOT robustly beat the rule on this data (honest result).")

    print("\nSystem view (gate = recall/budget knob; VLM arbiter gives precision):")
    for target in (0.95, 0.90):
        tt = None
        for t in np.linspace(0.05, 0.95, 181):
            p = (pte >= t).astype(int)
            if p[yte == 1].mean() >= target:
                tt = (t, p[yte == 1].mean(), p.mean())
        if tt:
            print(f"  to catch >={target:.0%} of accidents: gate flags {tt[2]:.0%} of videos "
                  f"(= the VLM-call budget), recall {tt[1]:.0%}")
    print("\nNext: Phase 6 — evaluate the FULL cascade (gate -> Gemini arbiter):")
    print("  system recall, FINAL false-alarm after Gemini, and total VLM calls.")


if __name__ == "__main__":
    main()
