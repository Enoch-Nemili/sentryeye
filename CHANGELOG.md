# Changelog

All notable changes to SentryEye. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-09

First release: the full budget-aware cascade, evaluated on real accident footage.

### Added
- **3-stage cascade**: YOLOv11 + ByteTrack detection and tracking on every frame, cheap motion-anomaly signals (stall, flow-at-stall, abrupt deceleration), and a vision-LLM verifier that runs only on escalation (~0.1% of frames).
- **Typed incident reports**: Pydantic `IncidentReport` (type, severity, lane_blocked, recommended action, confidence).
- **Provider-agnostic verifier**: Gemini (cloud) or Ollama (local), with retry and backoff on transient errors.
- **Evaluation on the TAD benchmark**: cheap gate at 81% recall / 31% false-alarm on 32 real test videos; resumable, cached end-to-end evaluation.
- **Learned-gate experiment** with train/test discipline: feature selection lifted ROC-AUC from 0.70 to 0.82, but the learned gate did not robustly beat the tuned rule, a documented feature-ceiling finding.
- CI: lint (ruff) and a compile check on every push.

[1.0.0]: https://github.com/Enoch-Nemili/sentryeye/releases/tag/v1.0.0
