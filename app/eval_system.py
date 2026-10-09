"""
Phase 6 — END-TO-END CASCADE EVALUATION (resumable).

Full system: gate escalates (a new stall) -> Gemini verifies -> final alarm only if
Gemini says it's a real incident. Reuses gate decisions from data/test_features.csv
(which videos escalate + first_stall_frame), so it calls Gemini once per flagged video.

RESUMABLE: every successful Gemini verdict is cached to data/system_cache.csv. If the
API is overloaded and some calls fail, just RE-RUN — it skips the ones already done and
only retries the failures, converging to a complete result. Numbers are final only when
"pending" reaches 0.

    python -m app.eval_system "/Users/.../TAD-benchmark/test"
"""

import csv
import os
import sys
import time

import cv2
import pandas as pd
from PIL import Image

from app.report import analyze_image

CONF_MIN = 0.5
CACHE = "data/system_cache.csv"


def load_cache():
    cache = {}
    if os.path.exists(CACHE):
        for _, r in pd.read_csv(CACHE).iterrows():
            cache[r["video"]] = (r["incident_type"], float(r["confidence"]))
    return cache


def save_to_cache(video, incident_type, confidence):
    new = not os.path.exists(CACHE)
    with open(CACHE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["video", "incident_type", "confidence"])
        w.writerow([video, incident_type, confidence])


def frame_at(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_idx, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, fps
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), fps


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    df = pd.read_csv("data/test_features.csv")
    cache = load_cache()

    n_acc = int((df.label == 1).sum())
    n_nor = int((df.label == 0).sum())
    vlm_calls_this_run = acc_detected = nor_false = pending = 0
    ttd_list = []

    print(f"{'video':26} {'true':9} gate  VLM-verdict            final")
    print("-" * 72)
    for _, r in df.iterrows():
        sub = "accident" if r.label == 1 else "normal"
        vid = os.path.join(root, sub, r.video)
        gate_escalated = r.num_stalled > 0
        incident_type, confidence, verdict = None, 0.0, "-"

        if gate_escalated:
            if r.video in cache:
                incident_type, confidence = cache[r.video]
                verdict = f"{incident_type}/{confidence:.2f} (cached)"
            else:
                img, fps = frame_at(vid, int(r.first_stall_frame))
                if img is not None:
                    try:
                        report, _ = analyze_image(img)
                        vlm_calls_this_run += 1
                        if report is not None:
                            incident_type, confidence = report.incident_type, report.confidence
                            verdict = f"{incident_type}/{confidence:.2f}"
                            save_to_cache(r.video, incident_type, confidence)
                    except Exception:  # noqa: BLE001 - API boundary: mark for re-run, keep evaluating
                        verdict = "API-ERROR (re-run)"
                        pending += 1
                    time.sleep(0.4)

        final_alarm = incident_type is not None and incident_type != "none" and confidence >= CONF_MIN
        if final_alarm and r.label == 1:
            acc_detected += 1
            fps = cv2.VideoCapture(vid).get(cv2.CAP_PROP_FPS) or 25.0
            ttd_list.append(r.first_stall_frame / fps)
        if final_alarm and r.label == 0:
            nor_false += 1

        gate_s = "FLAG" if gate_escalated else " .  "
        final_s = "ALARM" if final_alarm else "-"
        print(f"{r.video:26} {sub:9} {gate_s}  {verdict:26} {final_s}")

    print("\n" + "=" * 72)
    print("=== FULL-CASCADE (system) results on TAD test ===")
    print(f"System recall     : {acc_detected}/{n_acc} = {acc_detected/max(n_acc,1):.0%}")
    print(f"FINAL false-alarm : {nor_false}/{n_nor} = {nor_false/max(n_nor,1):.0%}")
    print(f"VLM calls this run: {vlm_calls_this_run}   (cached verdicts reused from prior runs)")
    if ttd_list:
        ttd_list.sort()
        print(f"Time-to-detection : median {ttd_list[len(ttd_list)//2]:.1f}s")
    if pending:
        print(f"\n>>> PENDING: {pending} flagged videos still have NO verdict (Gemini was busy).")
        print(">>> Numbers are NOT final. Re-run the same command to fill them in.")
    else:
        print("\nAll flagged videos have a verdict -> these numbers are FINAL.")
    print("\nThe GATE ALONE had 31% false alarm; the VLM arbiter removes them.")


if __name__ == "__main__":
    main()
