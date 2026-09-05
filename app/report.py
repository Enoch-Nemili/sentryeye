"""
Stage 3 — the vision-LLM VERIFIER. Provider-agnostic: Gemini (cloud) or Ollama (local).

Pick the backend with an env var (in .env or the shell):
    VLM_PROVIDER=ollama        # local, no API, no outages (default here)
    VLM_PROVIDER=gemini        # cloud
    OLLAMA_VLM_MODEL=llama3.2-vision   # any local vision model you've pulled

    python -m app.report data/sample_frame.jpg
"""

import os
import io
import sys
import time
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

VLM_PROVIDER = os.getenv("VLM_PROVIDER", "gemini").lower()
OLLAMA_VLM_MODEL = os.getenv("OLLAMA_VLM_MODEL", "llama3.2-vision")
GEMINI_MODEL = "gemini-3.6-flash"


class IncidentReport(BaseModel):
    incident_type: str = Field(description="none, stalled_vehicle, collision, debris, wrong_way, or congestion")
    severity: str = Field(description="none, low, medium, or high")
    vehicles_involved: int = Field(description="how many vehicles are part of the incident")
    lane_blocked: bool = Field(description="is a travel lane blocked?")
    description: str = Field(description="one or two sentences on what is happening")
    recommended_action: str = Field(description="what a dispatcher should do")
    confidence: float = Field(description="0.0 to 1.0 that this is a real incident")


PROMPT = ("You are a traffic-incident analyst reviewing a single traffic-camera frame. "
    "Decide whether there is an incident (accident/collision, stalled vehicle, debris, "
    "wrong-way driver, or other dangerous situation). If it looks like normal traffic, "
    "set incident_type to 'none'. Fill in the report accurately and concisely.")


def _analyze_gemini(image, retries=4):
    from google import genai
    from google.genai import types
    client = genai.Client()
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL, contents=[PROMPT, image],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", response_schema=IncidentReport),
            )
            return resp.parsed, resp.text
        except Exception as e:
            if any(s in str(e) for s in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded")) and attempt < retries - 1:
                wait = 2 * (2 ** attempt)
                print(f"    (transient API error, retrying in {wait}s...)")
                time.sleep(wait); continue
            raise


def _analyze_ollama(image):
    import ollama
    buf = io.BytesIO(); image.convert("RGB").save(buf, format="JPEG"); img_bytes = buf.getvalue()
    msgs = [{"role": "user", "content": PROMPT, "images": [img_bytes]}]
    try:
        resp = ollama.chat(model=OLLAMA_VLM_MODEL, messages=msgs,
                           format=IncidentReport.model_json_schema(),  # schema-guided JSON
                           options={"temperature": 0})
    except Exception:
        resp = ollama.chat(model=OLLAMA_VLM_MODEL, messages=msgs,
                           format="json", options={"temperature": 0})   # fallback: plain JSON mode
    text = resp["message"]["content"]
    return IncidentReport.model_validate_json(text), text


def analyze_image(image, retries=4):
    """Return (IncidentReport | None, raw_text). Backend chosen by VLM_PROVIDER."""
    if VLM_PROVIDER == "ollama":
        return _analyze_ollama(image)
    return _analyze_gemini(image, retries)


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_frame.jpg"
    image = Image.open(img_path)
    print(f"(VLM provider: {VLM_PROVIDER})")
    report, text = analyze_image(image)
    print(f"\nFrame: {img_path}\n--- Incident report ---")
    print(report.model_dump_json(indent=2) if report is not None else text)


if __name__ == "__main__":
    main()
