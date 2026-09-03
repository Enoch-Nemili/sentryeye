# SentryEye

Real-time AI **incident detection** for traffic cameras. Watches a video stream,
detects accidents/incidents within seconds, and produces a **structured, responder-ready
report** — spending an expensive vision-LLM only when a **learned, budget-aware gate**
says it's worth it.

## Why it's different
Not the cascade itself (that exists) — the contribution is a **learned budget-aware
escalation policy** (extending SkipComp), responder-ready **structured reports**, and an
honest metric: **time-to-detection + false-alarm rate on rare events**.

## Stack
OpenCV · YOLO/Ultralytics + ByteTrack (Stage 1) · a vision-LLM (Stage 2) · PyTorch
(the learned gate) · Pydantic (report schema) · FastAPI + WebSockets (demo).

## Dev setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.check_env
```
