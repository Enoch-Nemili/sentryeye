"""
Phase 2 (part 2) — cheap ANOMALY SIGNALS from the tracked vehicles.

Speed of each vehicle (px/frame) + a STALL flag.

Key refinement: a parked car and a stalled car both have speed ~0, but only one is
an incident. So we flag STOPPED **only if the vehicle was moving first, then stopped**
(a real stall in traffic) — not a car that was parked the whole time.

Run from the project root (venv active):
    python -m app.signals "/Users/.../MVI_20011/img%05d.jpg"
"""

import sys
from collections import defaultdict

import cv2
from ultralytics import YOLO

VEHICLE_CLASSES = {2, 3, 5, 7}   # car, motorcycle, bus, truck
STOP_SPEED = 1.5                 # px/frame below this = "not moving"
MOVE_SPEED = 4.0                 # px/frame above this = "was genuinely moving"
STOP_FRAMES = 15                 # this many slow frames in a row = "stopped"


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/car-detection.mp4"

    model = YOLO("yolo11s.pt")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Could not open {src}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter("data/signals.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    last_center = {}
    slow_count = defaultdict(int)
    has_moved = defaultdict(bool)     # did this vehicle ever actually move?
    stalled_ids = set()
    frame_idx = 0

    print("Detecting, tracking, and computing anomaly signals...")
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
        annotated = r.plot()

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
                    # a STALL = was moving before, and is now stopped for a while
                    if has_moved[tid] and slow_count[tid] >= STOP_FRAMES:
                        stalled_ids.add(tid)
                        cv2.putText(annotated, "STALLED",
                                    (int(cx - bw / 2), int(cy - bh / 2) - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                last_center[tid] = (cx, cy)

        writer.write(annotated)

    cap.release()
    writer.release()

    print(f"\nDone. Processed {frame_idx} frames.")
    print(f"Vehicles that MOVED then STALLED: {len(stalled_ids)}  ids={sorted(stalled_ids)}")
    print("Saved: data/signals.mp4")


if __name__ == "__main__":
    main()
