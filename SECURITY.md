# Security Policy

## Reporting a vulnerability

Please **don't open a public issue** for security problems. Email **enoch.das@gmail.com** with what you found, how to reproduce it, and the commit you tested. You'll get an acknowledgement within a few days.

## Things to know when running SentryEye

| Area | Guidance |
|------|----------|
| API keys | `GOOGLE_API_KEY` lives in a git-ignored `.env`. Never commit it; rotate it if it leaks. Ollama mode needs no key. |
| Video privacy | CCTV footage shows people and license plates. Datasets are never committed (`data/` is git-ignored). With Gemini, escalated frames are sent to a third-party API; use Ollama to keep all frames on your machine. |
| Model output | Vision-LLM responses are validated against the `IncidentReport` schema before use; malformed output is rejected, not trusted. |
| Decisions | SentryEye is a research prototype. Its reports are suggestions for a human dispatcher, not automated emergency actions. |
