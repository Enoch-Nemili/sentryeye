"""
Phase 4 — the BASELINE CASCADE: all three stages as one system.

  Stage 1 (YOLO+track)  -> Stage 2 (cheap signals)  -> Stage 3 (Gemini VLM)
  runs on EVERY frame       runs on EVERY frame          runs ONLY on escalation

The whole point: the cheap stages watch every frame almost for free; the expensive
VLM is called only when a cheap signal fires ("a vehicle stalled in a lane"). At the
end we print HOW RARELY the VLM ran -- that efficiency is the headline the learned
gate (Phase 5) will try to beat.

Run from the project root (venv active):
    python -m app.cascade "/Users/.../TAD-benchmark/test/accident/videox_test_T1.mp4"
"""

import os
import sys
from collections import defaultdict

import cv2
from PIL import Image
from ultralytics import YOLO

from app.report import analyze_image

VEHICLE_CLASSES = {2, 3, 5, 7}    # car, motorcycle, bus, truck
STOP_SPEED = 1.5                  # px/frame below this = "not moving"
MOVE_SPEED = 4.0                  # px/frame above this = "was genuinely moving"
STOP_FRAMES = 15                  # this many slow frames in a row = "stalled"

COOLDOWN_FRAMES = 100             # after escalating, wait this long before escalating again
MAX_ESCALATIONS = 5              # hard budget: never call the VLM more than this per video


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/car-detection.mp4"

    model = YOLO("yolo11s.pt")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Could not open {src}")
        return

    os.makedirs("data/escalations", exist_ok=True)

    last_center = {}
    slow_count = defaultdict(int)
    has_moved = defaultdict(bool)
    stalled_ids = set()

    frame_idx = 0
    vlm_calls = 0
    last_escalation_frame = -10 ** 9
    escalation_log = []

    print("Cascade running: cheap detector on every frame, VLM only on escalation...\n")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        results = model.track(
            frame, persist=True, conf=0.20,
            classes=list(VEHICLE_CLASSES), tracker="bytetrack.yaml", verbose=False,
        )
        r = results[0]

        newly_stalled = False
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            xywh = r.boxes.xywh.tolist()
            for tid, (cx, cy, bw, bh) in zip(ids, xywh):
                if tid in last_center:
                    px, py = last_center[tid]
                    speed = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    if speed > MOVE_SPEED:
                        has_moved[tid] = True
                    slow_count[tid] = slow_count[tid] + 1 if speed < STOP_SPEED else 0
                    if has_moved[tid] and slow_count[tid] >= STOP_FRAMES and tid not in stalled_ids:
                        stalled_ids.add(tid)
                        newly_stalled = True
                last_center[tid] = (cx, cy)

        # ESCALATE: a new stall fired, and we're within budget + past the cooldown
        if (newly_stalled
                and vlm_calls < MAX_ESCALATIONS
                and (frame_idx - last_escalation_frame) >= COOLDOWN_FRAMES):
            last_escalation_frame = frame_idx
            fp = f"data/escalations/frame{frame_idx:05d}.jpg"
            cv2.imwrite(fp, frame)                              # save the raw frame we escalated on
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            print(f"[frame {frame_idx}] cheap signal fired -> escalating to the VLM...")
            report, text = analyze_image(pil)
            vlm_calls += 1
            if report is not None:
                print(f"    VLM => {report.incident_type} / {report.severity} / "
                      f"lane_blocked={report.lane_blocked} / conf={report.confidence:.2f}")
                print(f"           {report.description}\n")
                escalation_log.append((frame_idx, report.incident_type, report.confidence))
            else:
                print(f"    VLM(raw) => {text}\n")

    cap.release()

    pct = 100 * vlm_calls / max(frame_idx, 1)
    print("=== CASCADE SUMMARY ===")
    print(f"Frames processed         : {frame_idx}")
    print(f"Cheap stage ran on       : {frame_idx} frames (100%)")
    print(f"Expensive VLM ran on     : {vlm_calls} frames ({pct:.2f}% of frames)")
    print(f"Vehicles that stalled    : {sorted(stalled_ids)}")
    for fi, kind, conf in escalation_log:
        print(f"   - frame {fi}: {kind} (conf {conf:.2f})")


if __name__ == "__main__":
    main()
