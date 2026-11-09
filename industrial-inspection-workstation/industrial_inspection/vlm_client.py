"""Gemini VLM adapter shared by all inspection process modules."""
from __future__ import annotations

import json
import mimetypes
import os

from dotenv import load_dotenv
from pathlib import Path

from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

SYSTEM_PROMPT = """You are an industrial evidence-gated visual inspection assistant.
Evaluate only evidence visible in the supplied component images and supplied reference context.
If evidence is insufficient, explicitly say so and state exactly what additional view, lighting,
cleaning, or measurement would resolve the uncertainty. Do not invent hidden defects or
measurements. Confidence must reflect the actual evidence quality. Maintain a concise technical
industrial tone."""

INSPECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "visual_evidence": {"type": "string"},
        "uncertainty": {"type": "string"},
        "assessment": {"type": "string", "enum": ["ACCEPT", "REJECT", "INDETERMINATE"]},
        "action": {"type": "string"},
    },
    "required": ["confidence", "visual_evidence", "uncertainty", "assessment", "action"],
    "additionalProperties": False,
}

EVIDENCE_ADEQUACY_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence_adequate": {"type": "boolean"},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "claim_area_visible": {"type": "boolean"},
        "same_physical_feature": {"type": "boolean"},
        "adequacy_reason": {"type": "string"},
        "missing_evidence": {"type": "string"},
    },
    "required": [
        "evidence_adequate", "confidence", "claim_area_visible",
        "same_physical_feature", "adequacy_reason", "missing_evidence"
    ],
    "additionalProperties": False,
}

CLAIM_ADJUDICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "adjudication": {"type": "string", "enum": ["SUPPORT", "REFUTE", "INSUFFICIENT"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "evidence_summary": {"type": "string"},
        "reasoning": {"type": "string"},
        "required_next_evidence": {"type": "string"},
    },
    "required": ["adjudication", "confidence", "evidence_summary", "reasoning", "required_next_evidence"],
    "additionalProperties": False,
}


def _require_api_key() -> None:
    if not GEMINI_API_KEY.strip() or GEMINI_API_KEY.strip() == "your_api_key_here":
        raise RuntimeError("Set GEMINI_API_KEY in your environment or .env file before calling Gemini.")


def _image_part(image_path: str) -> types.Part:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    if not mime_type.startswith("image/"):
        mime_type = "image/jpeg"
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)


def _contents(images: str | list[str] | None, prompt: str) -> list[object]:
    content: list[object] = [prompt]
    if images:
        paths = [images] if isinstance(images, str) else list(images)
        for index, path in enumerate(paths, start=1):
            content.append(f"Image {index}:")
            content.append(_image_part(path))
    return content


def _generate_structured(images: str | list[str] | None, prompt: str, schema: dict) -> dict:
    _require_api_key()
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=_contents(images, prompt),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_json_schema=schema,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                max_output_tokens=768,
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini API request failed: {exc}") from exc

    raw_text = (response.text or "").strip()
    if not raw_text:
        raise RuntimeError("Gemini returned an empty structured response.")
    try:
        return json.loads(raw_text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(f"Gemini returned invalid structured JSON: {exc}") from exc


def _fallback(raw_text: str, uncertainty: str) -> dict:
    return {
        "confidence": 0,
        "visual_evidence": raw_text,
        "uncertainty": uncertainty,
        "assessment": "INDETERMINATE",
        "action": "Re-capture the image and retry inspection.",
    }


def query_vlm(image_path: str | list[str], operator_prompt: str, extra_context: str = "") -> dict:
    """Standard ACCEPT/REJECT/INDETERMINATE inspection call."""
    prompt = operator_prompt
    if extra_context:
        prompt += f"\n\nAdditional context: {extra_context}"
    try:
        parsed = _generate_structured(image_path, prompt, INSPECTION_SCHEMA)
    except RuntimeError as exc:
        # Preserve the original adapter's safe failure contract for malformed output,
        # but allow transport/API failures to be visible to the operator.
        if "invalid structured JSON" in str(exc) or "empty structured response" in str(exc):
            return _fallback("", str(exc))
        raise

    parsed["confidence"] = max(0, min(100, int(parsed.get("confidence", 0))))
    parsed.setdefault("visual_evidence", "")
    parsed.setdefault("uncertainty", "")
    parsed.setdefault("assessment", "INDETERMINATE")
    parsed.setdefault("action", "")
    if parsed["assessment"] not in {"ACCEPT", "REJECT", "INDETERMINATE"}:
        parsed["assessment"] = "INDETERMINATE"
    return parsed


def query_evidence_adequacy(images: list[str], claim: str, operator_justification: str) -> dict:
    """Gate an override on whether the new physical evidence actually addresses the old claim."""
    prompt = f"""EVIDENCE ADEQUACY GATE — do not reclassify the component.

Original inspection claim:
{claim}

Operator justification:
{operator_justification}

Image order is documented by the caller as reference/original/new-evidence where available.
Judge ONLY whether the NEW physical evidence makes the specific feature/condition in the claim
observable well enough to test that claim. It is not adequate merely because it is a new image.
Check viewpoint, occlusion, lighting, focus, scale, and whether it actually shows the same physical
feature. Return evidence_adequate=false if the claim cannot be directly addressed."""
    result = _generate_structured(images, prompt, EVIDENCE_ADEQUACY_SCHEMA)
    result["confidence"] = max(0, min(100, int(result.get("confidence", 0))))
    return result


def query_claim_adjudication(images: list[str], claim: str, operator_justification: str,
                             adequacy_context: dict) -> dict:
    """Adjudicate the original claim as SUPPORT, REFUTE, or INSUFFICIENT.

This deliberately does not ask for a fresh ACCEPT/REJECT classification.
"""
    prompt = f"""CLAIM ADJUDICATION — do not perform a fresh whole-part classification.

Original inspection claim:
{claim}

Operator justification:
{operator_justification}

Evidence adequacy gate result:
{json.dumps(adequacy_context, ensure_ascii=False)}

Use the newly provided physical evidence to adjudicate the ORIGINAL CLAIM only:
- SUPPORT: new evidence supports the original claim.
- REFUTE: new evidence directly contradicts the original claim.
- INSUFFICIENT: even after the adequacy gate, the evidence is not strong enough to decide.

Be conservative. A changed camera angle alone is not a refutation. State the visible evidence that
connects the adjudication to the original claimed feature."""
    result = _generate_structured(images, prompt, CLAIM_ADJUDICATION_SCHEMA)
    result["confidence"] = max(0, min(100, int(result.get("confidence", 0))))
    if result.get("adjudication") not in {"SUPPORT", "REFUTE", "INSUFFICIENT"}:
        result["adjudication"] = "INSUFFICIENT"
    return result
