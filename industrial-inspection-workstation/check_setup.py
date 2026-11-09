from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

REQUIRED = ["cv2", "numpy", "google.genai", "nicegui", "dotenv"]
OPTIONAL = ["ultralytics"]
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
VLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")

print(f"Python: {sys.version.split()[0]}")
if sys.version_info < (3, 10):
    print("ERROR: Python 3.10+ is required.")

all_required = True
for name in REQUIRED + OPTIONAL:
    try:
        mod = importlib.import_module(name)
        print(f"OK: {name} {getattr(mod, '__version__', '')}")
    except Exception as exc:
        label = "ERROR" if name in REQUIRED else "OPTIONAL"
        print(f"{label}: {name}: {exc}")
        if name in REQUIRED:
            all_required = False

configured_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
has_env_key = bool(configured_key.strip() and configured_key.strip() != "your_api_key_here")
print("OK: Gemini API key configured" if has_env_key else "WARNING: Gemini API key missing (set it in .env or environment)")
print(f"VLM model: {VLM_MODEL}")
print(f"Embedding model: {EMBEDDING_MODEL}")

for folder in ("runtime", "reference_images", "memory_archive"):
    path = ROOT / folder
    path.mkdir(parents=True, exist_ok=True)
    print(f"OK: data folder {folder}/")

if "--api" in sys.argv:
    if not all_required:
        raise SystemExit("Install requirements before running the live API check.")
    from google import genai
    from google.genai import types
    from industrial_inspection.vlm_client import GEMINI_API_KEY

    if not GEMINI_API_KEY:
        raise SystemExit("Set GEMINI_API_KEY in .env before running --api")
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=VLM_MODEL,
        contents="Return exactly: GEMINI_OK",
        config=types.GenerateContentConfig(max_output_tokens=20),
    )
    print(f"VLM API TEST: {(response.text or '').strip()}")
    emb = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents="industrial inspection setup check",
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    count = len(getattr(emb.embeddings[0], "values", []) or []) if emb.embeddings else 0
    print(f"EMBEDDING API TEST: {count} dimensions")

if not all_required:
    raise SystemExit("Required packages are missing. Run: pip install -r requirements.txt")
