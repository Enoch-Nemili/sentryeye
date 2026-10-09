"""
Phase 1 — open a video with OpenCV, print its properties, and save one frame.

OpenCV reads video as a sequence of frames (images). This confirms we can load
our footage before we start detecting anything in it.

Run from the project root (venv active):
    python -m app.read_video data/car-detection.mp4
"""

import sys

import cv2


def main():
    # which video? default to the highway clip if none given
    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/car-detection.mp4"

    # a VideoCapture is OpenCV's "video reader"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path} — is the path right?")
        return

    # read the video's basic properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps else 0

    print(f"Video      : {video_path}")
    print(f"Resolution : {width} x {height}")
    print(f"FPS        : {fps:.1f}")
    print(f"Frames     : {frame_count}")
    print(f"Duration   : {duration:.1f} seconds")

    # jump to the middle of the video and grab that frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
    ok, frame = cap.read()
    if ok:
        out_path = "data/sample_frame.jpg"
        cv2.imwrite(out_path, frame)  # save the frame as an image
        print(f"\nSaved a sample frame to {out_path} — open it to see the footage.")
    else:
        print("Could not read a frame from the video.")

    cap.release()  # always release the reader when done


if __name__ == "__main__":
    main()
