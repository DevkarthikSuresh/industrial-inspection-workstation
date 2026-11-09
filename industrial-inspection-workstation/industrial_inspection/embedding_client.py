"""Gemini Embedding 2 adapter for Industrial Memory.

Embeds the archived inspection image and the VLM rationale independently into
Gemini Embedding 2's shared multimodal space.  Keeping both vectors allows:
- visual drift against known-good / prior-good images,
- semantic drift against prior rationales,
- direct image<->rationale concordance checks.
"""
from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from google import genai
from google.genai import types

from .vlm_client import GEMINI_API_KEY

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIMENSION = 768


def _first_vector(response) -> list[float]:
    embeddings = getattr(response, "embeddings", None) or []
    if not embeddings:
        raise RuntimeError("Gemini Embedding 2 returned no embedding.")
    values = getattr(embeddings[0], "values", None)
    if values is None:
        raise RuntimeError("Gemini Embedding 2 returned an embedding without values.")
    return [float(v) for v in values]


def _image_part(image_path: str) -> types.Part:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding image not found: {image_path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    if mime_type not in {"image/jpeg", "image/png"}:
        mime_type = "image/jpeg"
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)


def embed_episode(image_path: str, rationale: str) -> dict:
    """Return image and rationale embeddings in the same Gemini space."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION)

    image_response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[_image_part(image_path)],
        config=config,
    )
    rationale_response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=rationale or "No explicit visual rationale was produced.",
        config=config,
    )

    return {
        "model": EMBEDDING_MODEL,
        "dimension": EMBEDDING_DIMENSION,
        "image_embedding": _first_vector(image_response),
        "rationale_embedding": _first_vector(rationale_response),
    }
