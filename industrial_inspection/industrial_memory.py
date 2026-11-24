"""Persistent multilayer Industrial Memory.

Layers:
1. Semantic/reference memory: taught known-good parts and reference embeddings.
2. Immutable episodic memory: append-only inspection episodes + image archives.
3. Batch memory: active production-batch scope per part.
4. Embedding/drift memory: Gemini Embedding 2 image + rationale vectors,
   visual drift, semantic drift, and cross-modal concordance/disagreement.
5. Audit memory: explainability and evidence-gated override events.

The store is deliberately JSON-backed so it remains edge-friendly.  The public
functions preserve the interfaces used by the original process modules.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path

from . import embedding_client

ROOT_DIR = Path(os.getenv("INDUSTRIAL_MEMORY_ROOT", Path(__file__).resolve().parents[1])).resolve()
DB_PATH = str(ROOT_DIR / "industrial_memory.json")
REFERENCE_DIR = ROOT_DIR / "reference_images"
ARCHIVE_DIR = ROOT_DIR / "memory_archive"

VISUAL_DRIFT_THRESHOLD = 0.25
SEMANTIC_DRIFT_THRESHOLD = 0.25
CROSS_MODAL_CONCORDANCE_THRESHOLD = 0.35

_DEFAULT_DB = {
    "schema_version": 2,
    "learned_parts": {},
    "batches": [],
    "active_batches": {},
    "inspection_results": [],  # legacy-compatible name; records are immutable episodes
    "explainability_logs": [],
    "override_history": [],
}


def _fresh_db() -> dict:
    return copy.deepcopy(_DEFAULT_DB)


def _migrate(db: dict) -> dict:
    merged = _fresh_db()
    for key, default in merged.items():
        if key not in db:
            db[key] = copy.deepcopy(default)
    db["schema_version"] = 2
    return db


def _load() -> dict:
    if not os.path.exists(DB_PATH):
        return _fresh_db()
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return _migrate(json.load(f))


def _save(db: dict) -> None:
    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ROOT_DIR / f".industrial_memory.{uuid.uuid4().hex}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    os.replace(tmp, DB_PATH)


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value) or "unnamed"


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cosine(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return None
    return max(-1.0, min(1.0, dot / (na * nb)))


def _centroid(vectors: list[list[float]]) -> list[float] | None:
    valid = [v for v in vectors if v]
    if not valid:
        return None
    dim = len(valid[0])
    valid = [v for v in valid if len(v) == dim]
    if not valid:
        return None
    return [sum(v[i] for v in valid) / len(valid) for i in range(dim)]


def _archive_image(part_name: str, batch_id: str, episode_id: str,
                   image_path: str, assessment: str) -> str:
    source = Path(image_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Inspection image not found: {image_path}")
    category = {"ACCEPT": "good", "REJECT": "bad"}.get(assessment, "indeterminate")
    suffix = source.suffix.lower() if source.suffix.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
    target_dir = ARCHIVE_DIR / _safe_name(part_name) / _safe_name(batch_id) / category
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{episode_id}{suffix}"
    shutil.copy2(source, target)
    return str(target.resolve())


def _episode_hash(record_without_hash: dict) -> str:
    payload = json.dumps(record_without_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------- Batch memory ----------

def start_new_batch(part_name: str, label: str = "") -> dict:
    db = _load()
    batch_id = f"batch-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    record = {
        "batch_id": batch_id,
        "part_name": part_name,
        "label": label or f"{part_name} production batch",
        "started_at": time.time(),
    }
    db["batches"].append(record)
    db["active_batches"][part_name] = batch_id
    _save(db)
    return copy.deepcopy(record)


def get_active_batch_id(part_name: str) -> str | None:
    return _load()["active_batches"].get(part_name)


def get_or_create_active_batch(part_name: str) -> str:
    existing = get_active_batch_id(part_name)
    if existing:
        return existing
    return start_new_batch(part_name)["batch_id"]


def get_batches(part_name: str | None = None) -> list:
    rows = _load()["batches"]
    if part_name is not None:
        rows = [r for r in rows if r.get("part_name") == part_name]
    return copy.deepcopy(rows)


# ---------- Learn Engine / semantic reference memory ----------

def save_reference(part_name: str, reference_image: str, description: str) -> None:
    db = _load()
    source = Path(reference_image).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Reference image not found: {reference_image}")

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower() or ".jpg"
    stored = REFERENCE_DIR / f"{_safe_name(part_name)}{suffix}"
    if source != stored.resolve():
        shutil.copy2(source, stored)

    embedding = None
    embedding_error = ""
    try:
        embedding = embedding_client.embed_episode(str(stored), description)
    except Exception as exc:  # embedding failure must never erase a usable taught reference
        embedding_error = str(exc)

    db["learned_parts"][part_name] = {
        "reference_image": str(stored.resolve()),
        "description": description,
        "taught_at": time.time(),
        "image_sha256": _sha256_file(stored),
        "embedding": embedding,
        "embedding_status": "READY" if embedding else "UNAVAILABLE",
        "embedding_error": embedding_error,
    }
    _save(db)


def get_reference(part_name: str) -> dict | None:
    ref = _load()["learned_parts"].get(part_name)
    return copy.deepcopy(ref) if ref else None


def has_seen_before(part_name: str) -> bool:
    return get_reference(part_name) is not None


# ---------- Immutable episodic inspection memory ----------

def _baseline_vectors(db: dict, part_name: str, batch_id: str) -> tuple[list[float] | None, list[float] | None, str]:
    accepted = [
        r for r in db["inspection_results"]
        if r.get("part_name") == part_name
        and r.get("batch_id") == batch_id
        and r.get("assessment") == "ACCEPT"
        and r.get("embeddings", {}).get("status") == "READY"
    ]
    if accepted:
        image_base = _centroid([r["embeddings"].get("image_embedding", []) for r in accepted])
        semantic_base = _centroid([r["embeddings"].get("rationale_embedding", []) for r in accepted])
        return image_base, semantic_base, "accepted episodes in current batch"

    ref = db["learned_parts"].get(part_name) or {}
    ref_emb = ref.get("embedding") or {}
    if ref_emb:
        return ref_emb.get("image_embedding"), ref_emb.get("rationale_embedding"), "known-good reference"
    return None, None, "no embedding baseline available"


def _drift_metrics(db: dict, part_name: str, batch_id: str, embeddings: dict | None) -> dict:
    if not embeddings:
        return {
            "visual_drift": None,
            "semantic_drift": None,
            "cross_modal_similarity": None,
            "concordance": "UNAVAILABLE",
            "baseline": "embedding unavailable",
        }

    image_base, semantic_base, baseline = _baseline_vectors(db, part_name, batch_id)
    image_vec = embeddings.get("image_embedding")
    rationale_vec = embeddings.get("rationale_embedding")

    visual_similarity = _cosine(image_vec, image_base)
    semantic_similarity = _cosine(rationale_vec, semantic_base)
    cross_modal_similarity = _cosine(image_vec, rationale_vec)

    visual_drift = None if visual_similarity is None else max(0.0, 1.0 - visual_similarity)
    semantic_drift = None if semantic_similarity is None else max(0.0, 1.0 - semantic_similarity)

    if cross_modal_similarity is None:
        concordance = "UNAVAILABLE"
    elif cross_modal_similarity >= CROSS_MODAL_CONCORDANCE_THRESHOLD:
        concordance = "CONCORDANT"
    else:
        concordance = "DISAGREEMENT"

    if visual_drift is not None and semantic_drift is not None:
        visual_state = visual_drift >= VISUAL_DRIFT_THRESHOLD
        semantic_state = semantic_drift >= SEMANTIC_DRIFT_THRESHOLD
        if visual_state != semantic_state:
            concordance = "DISAGREEMENT"

    return {
        "visual_drift": round(visual_drift, 6) if visual_drift is not None else None,
        "semantic_drift": round(semantic_drift, 6) if semantic_drift is not None else None,
        "cross_modal_similarity": round(cross_modal_similarity, 6) if cross_modal_similarity is not None else None,
        "concordance": concordance,
        "baseline": baseline,
        "visual_drift_threshold": VISUAL_DRIFT_THRESHOLD,
        "semantic_drift_threshold": SEMANTIC_DRIFT_THRESHOLD,
        "cross_modal_concordance_threshold": CROSS_MODAL_CONCORDANCE_THRESHOLD,
    }


def log_inspection(part_name: str, image_path: str, vlm_result: dict,
                   parent_inspection_id: str | None = None, batch_id: str | None = None) -> str:
    """Append one immutable inspection episode and archive its physical image."""
    db = _load()
    inspection_id = str(uuid.uuid4())
    batch_id = batch_id or db["active_batches"].get(part_name)
    if not batch_id:
        batch = start_new_batch(part_name)
        db = _load()
        batch_id = batch["batch_id"]

    assessment = vlm_result.get("assessment", "INDETERMINATE")
    archived_image = _archive_image(part_name, batch_id, inspection_id, image_path, assessment)
    rationale = vlm_result.get("visual_evidence", "") or ""

    embedding_payload = None
    embedding_error = ""
    try:
        embedding_payload = embedding_client.embed_episode(archived_image, rationale)
    except Exception as exc:
        embedding_error = str(exc)

    metrics = _drift_metrics(db, part_name, batch_id, embedding_payload)
    timestamp = time.time()
    embedding_record = {
        "status": "READY" if embedding_payload else "UNAVAILABLE",
        "error": embedding_error,
        **(embedding_payload or {}),
    }

    record = {
        "episode_version": 1,
        "inspection_id": inspection_id,
        "parent_inspection_id": parent_inspection_id,
        "batch_id": batch_id,
        "part_name": part_name,
        "source_image_path": str(Path(image_path).resolve()),
        "image_path": archived_image,
        "archive_class": {"ACCEPT": "good", "REJECT": "bad"}.get(assessment, "indeterminate"),
        "image_sha256": _sha256_file(archived_image),
        "confidence": vlm_result.get("confidence"),
        "assessment": assessment,
        "visual_evidence": rationale,
        "uncertainty": vlm_result.get("uncertainty"),
        "action": vlm_result.get("action"),
        "embeddings": embedding_record,
        "memory_metrics": metrics,
        "timestamp": timestamp,
    }
    record["episode_hash"] = _episode_hash(record)

    db["inspection_results"].append(record)
    db["explainability_logs"].append({
        "inspection_id": inspection_id,
        "batch_id": batch_id,
        "rationale": rationale,
        "timestamp": timestamp,
    })
    _save(db)
    return inspection_id


def get_inspection(inspection_id: str) -> dict | None:
    for record in _load()["inspection_results"]:
        if record.get("inspection_id") == inspection_id:
            return copy.deepcopy(record)
    return None


def get_inspection_history(part_name: str, batch_id: str | None = None) -> list:
    rows = [r for r in _load()["inspection_results"] if r.get("part_name") == part_name]
    if batch_id is not None:
        rows = [r for r in rows if r.get("batch_id") == batch_id]
    return copy.deepcopy(rows)


def verify_episode_integrity(inspection_id: str) -> bool:
    record = get_inspection(inspection_id)
    if not record:
        return False
    expected = record.pop("episode_hash", None)
    image_path = record.get("image_path")
    if not expected or not image_path or not Path(image_path).exists():
        return False
    if _sha256_file(image_path) != record.get("image_sha256"):
        return False
    return expected == _episode_hash(record)


def get_episode_analytics(inspection_id: str) -> dict:
    record = get_inspection(inspection_id) or {}
    return {
        "batch_id": record.get("batch_id"),
        "archive_class": record.get("archive_class"),
        "embedding_status": (record.get("embeddings") or {}).get("status"),
        **(record.get("memory_metrics") or {}),
    }


def get_memory_snapshot(part_name: str) -> dict:
    db = _load()
    batch_id = db["active_batches"].get(part_name)
    episodes = [r for r in db["inspection_results"] if r.get("part_name") == part_name]
    batch_episodes = [r for r in episodes if r.get("batch_id") == batch_id] if batch_id else []
    latest = batch_episodes[-1] if batch_episodes else None
    return {
        "part_name": part_name,
        "active_batch_id": batch_id,
        "total_episodes": len(episodes),
        "batch_episodes": len(batch_episodes),
        "latest_metrics": copy.deepcopy((latest or {}).get("memory_metrics") or {}),
        "latest_embedding_status": ((latest or {}).get("embeddings") or {}).get("status"),
    }


# ---------- Evidence-gated override audit memory ----------

def log_override(inspection_id: str, operator_id: str, new_evidence_image: str | None,
                 justification: str, accepted: bool, **details) -> dict:
    db = _load()
    record = {
        "override_id": str(uuid.uuid4()),
        "inspection_id": inspection_id,
        "operator_id": operator_id,
        "new_evidence_image": new_evidence_image,
        "justification": justification,
        "accepted": accepted,
        "timestamp": time.time(),
        **details,
    }
    db["override_history"].append(record)
    _save(db)
    return copy.deepcopy(record)


def get_override_history(inspection_id: str) -> list:
    return copy.deepcopy([
        r for r in _load()["override_history"] if r.get("inspection_id") == inspection_id
    ])


# ---------- Destructive maintenance ----------

def wipe_all_memory() -> dict:
    """Delete all learned references, episodes, embeddings, archives, batches and audits."""
    for path in (REFERENCE_DIR, ARCHIVE_DIR):
        if path.exists():
            shutil.rmtree(path)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    _save(_fresh_db())
    return {"status": "WIPED", "db_path": DB_PATH}
