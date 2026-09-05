"""
Phase 4->5 bridge — run the cheap stages over a whole split, score the single-rule
baseline, and dump a per-video FEATURES table (training data for the learned gate).

v2: adds CONTEXT-AWARE features to separate incidents from ordinary congestion:
  - max_flow_at_stall     : fraction of OTHER vehicles moving at the moment a vehicle stalls
                            (high = stopped while traffic flows = incident-like)
  - isolated_stall_count  : # of stalls that happened while the majority kept moving
  - max_abrupt_decel      : sharpest single-frame speed drop (collision = sudden)

Works on both TAD splits (accident* -> 1, normal* -> 0).

  python -m app.eval_gate ".../TAD-benchmark/test"  --out data/test_features.csv
  python -m app.eval_gate ".../TAD-benchmark/train" --limit 40 --out data/train_features.csv
"""

import sys, os, glob, csv, argparse
from collections import defaultdict
import cv2
from ultralytics import YOLO

VEHICLE_CLASSES = {2, 3, 5, 7}
STOP_SPEED = 1.5
MOVE_SPEED = 4.0
STOP_FRAMES = 15

FEATURE_COLS = ["video", "label", "frames", "unique_vehicles", "max_in_frame",
                "num_stalled", "max_simultaneous_stopped", "first_stall_frame",
                "max_flow_at_stall", "isolated_stall_count", "max_abrupt_decel"]


def features_for_video(model, path):
    cap = cv2.VideoCapture(path)
    last_center, last_speed = {}, {}
    slow_count = defaultdict(int)
    has_moved = defaultdict(bool)
    stalled_ids, seen_ids = set(), set()
    frame_idx = max_in_frame = max_simul_stopped = 0
    first_stall_frame = -1
    max_flow_at_stall = 0.0
    isolated_stall_count = 0
    max_abrupt_decel = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        r = model.track(frame, persist=True, conf=0.20,
                        classes=list(VEHICLE_CLASSES), tracker="bytetrack.yaml", verbose=False)[0]

        if r.boxes is None or r.boxes.id is None:
            continue
        ids = r.boxes.id.int().tolist()
        xywh = r.boxes.xywh.tolist()
        max_in_frame = max(max_in_frame, len(ids))

        # pass 1: speeds for vehicles we have a previous position for + this frame's flow
        speeds = {}
        moving = defined = 0
        for tid, (cx, cy, bw, bh) in zip(ids, xywh):
            if tid in last_center:
                px, py = last_center[tid]
                s = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                speeds[tid] = s
                defined += 1
                if s > MOVE_SPEED:
                    moving += 1
        flow_frac = moving / defined if defined else 0.0

        # pass 2: update stall logic + context features
        stopped_now = 0
        for tid, (cx, cy, bw, bh) in zip(ids, xywh):
            seen_ids.add(tid)
            if tid in speeds:
                s = speeds[tid]
                if tid in last_speed:
                    decel = last_speed[tid] - s
                    if decel > max_abrupt_decel:
                        max_abrupt_decel = decel
                if s > MOVE_SPEED:
                    has_moved[tid] = True
                slow_count[tid] = slow_count[tid] + 1 if s < STOP_SPEED else 0
                if slow_count[tid] >= STOP_FRAMES:
                    stopped_now += 1
                if has_moved[tid] and slow_count[tid] >= STOP_FRAMES and tid not in stalled_ids:
                    stalled_ids.add(tid)
                    if first_stall_frame < 0:
                        first_stall_frame = frame_idx
                    if flow_frac > max_flow_at_stall:
                        max_flow_at_stall = flow_frac
                    if flow_frac >= 0.5:
                        isolated_stall_count += 1
                last_speed[tid] = s
            last_center[tid] = (cx, cy)
        max_simul_stopped = max(max_simul_stopped, stopped_now)
    cap.release()

    return {
        "frames": frame_idx,
        "unique_vehicles": len(seen_ids),
        "max_in_frame": max_in_frame,
        "num_stalled": len(stalled_ids),
        "max_simultaneous_stopped": max_simul_stopped,
        "first_stall_frame": first_stall_frame,
        "max_flow_at_stall": round(max_flow_at_stall, 3),
        "isolated_stall_count": isolated_stall_count,
        "max_abrupt_decel": round(max_abrupt_decel, 2),
    }


def collect_videos(root, limit):
    groups = {1: [], 0: []}
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub):
            continue
        label = 1 if name.startswith("accident") else 0 if name.startswith("normal") else None
        if label is None:
            continue
        groups[label].extend(sorted(glob.glob(os.path.join(sub, "*.mp4"))))
    out = []
    for label in (1, 0):
        vids = groups[label][:limit] if limit else groups[label]
        out.extend((v, label) for v in vids)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--limit", type=int, default=None, help="max videos PER CLASS")
    ap.add_argument("--out", default="data/gate_features.csv")
    args = ap.parse_args()

    model = YOLO("yolo11s.pt")
    os.makedirs("data", exist_ok=True)
    videos = collect_videos(args.root, args.limit)
    print(f"{len(videos)} videos to process -> {args.out}\n")

    rows = []
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLS)
        writer.writeheader()
        for i, (v, label) in enumerate(videos, 1):
            feats = features_for_video(model, v)
            row = {"video": os.path.basename(v), "label": label, **feats}
            writer.writerow(row); f.flush(); rows.append(row)
            tag = "accident" if label == 1 else "normal"
            print(f"[{i:3d}/{len(videos)}] [{tag:8}] {os.path.basename(v):24} "
                  f"stalled={feats['num_stalled']:2d} flow@stall={feats['max_flow_at_stall']:.2f} "
                  f"isol={feats['isolated_stall_count']} decel={feats['max_abrupt_decel']:.1f}")

    acc = [r for r in rows if r["label"] == 1]; nor = [r for r in rows if r["label"] == 0]
    tp = sum(1 for r in acc if r["num_stalled"] > 0); fp = sum(1 for r in nor if r["num_stalled"] > 0)
    print("\n=== BASELINE GATE (num_stalled>0) ===")
    print(f"Accident: {len(acc)} -> detected {tp} (recall {tp/max(len(acc),1):.0%})")
    print(f"Normal  : {len(nor)} -> false alarm {fp} (FA {fp/max(len(nor),1):.0%})")
    print(f"Saved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
