"""
Phase 2 — Stage 1: detect + track vehicles with YOLO (the cheap, always-on loop).

For each frame we run YOLO to find vehicles, and a tracker (ByteTrack) to give each
vehicle a stable ID across frames (so "car #3" stays #3 as it moves). We save an
annotated video so you can watch it work.

Run from the project root (venv active):
    python -m app.detect data/car-detection.mp4
"""

import sys
import cv2
from ultralytics import YOLO

# COCO class IDs for vehicles (YOLO was trained on these categories)
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/car-detection.mp4"

    # yolo11s = the "small" model: more accurate than nano (better on tricky/overhead
    # angles), still fast. Downloads automatically the first time.
    model = YOLO("yolo11s.pt")
    conf_threshold = 0.20  # lower = catches fainter/overhead vehicles (a few more false hits)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # a writer to save the annotated output video
    out_path = "data/annotated.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    seen_ids = set()
    max_in_frame = 0
    frame_idx = 0

    print("Running YOLO + tracking on every frame...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # detect + track vehicles; persist=True keeps track IDs stable across frames
        results = model.track(
            frame,
            persist=True,
            conf=conf_threshold,
            classes=list(VEHICLE_CLASSES.keys()),
            tracker="bytetrack.yaml",
            verbose=False,
        )
        r = results[0]

        # count vehicles in this frame + remember their track IDs
        n = 0
        if r.boxes is not None and r.boxes.id is not None:
            n = len(r.boxes.id)
            for tid in r.boxes.id.tolist():
                seen_ids.add(int(tid))
        max_in_frame = max(max_in_frame, n)

        # r.plot() draws the boxes + IDs onto the frame; save it to the output video
        writer.write(r.plot())

        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx}: {n} vehicles")

    cap.release()
    writer.release()

    print(f"\nDone. Processed {frame_idx} frames.")
    print(f"Most vehicles in one frame   : {max_in_frame}")
    print(f"Total unique vehicles tracked: {len(seen_ids)}")
    print(f"Annotated video saved to     : {out_path}  (open it to watch the detections)")


if __name__ == "__main__":
    main()
