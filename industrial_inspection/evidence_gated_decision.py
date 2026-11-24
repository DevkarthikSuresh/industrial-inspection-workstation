"""Enhanced Evidence-Gated Decision Protocol.

An operator override never triggers a generic reclassification.  The original
inspection rationale becomes an explicit claim.  New physical evidence must
first be adequate to address that claim, after which Gemini adjudicates the
claim as SUPPORT / REFUTE / INSUFFICIENT.  Only a confident REFUTE can change
the original ACCEPT/REJECT decision.
"""
from __future__ import annotations

import hashlib
import shutil
import time
import uuid
from pathlib import Path

from . import industrial_memory as memory
from .vlm_client import query_claim_adjudication, query_evidence_adequacy

EVIDENCE_ADEQUACY_CONFIDENCE_THRESHOLD = 70
ADJUDICATION_CONFIDENCE_THRESHOLD = 80


def _claim_from_prior(prior: dict) -> str:
    assessment = prior.get("assessment", "INDETERMINATE")
    rationale = prior.get("visual_evidence") or "No explicit rationale was recorded."
    if assessment == "REJECT":
        return f"The component is defective/rejectable because: {rationale}"
    if assessment == "ACCEPT":
        return f"The component is acceptable with no rejectable defect in the inspected feature because: {rationale}"
    return f"The original inspection was indeterminate because: {rationale}"


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _archive_override_evidence(prior: dict, new_evidence_image: str) -> str:
    source = Path(new_evidence_image).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Override evidence image not found: {new_evidence_image}")
    batch_id = prior.get("batch_id") or memory.get_or_create_active_batch(prior["part_name"])
    root = Path(memory.ARCHIVE_DIR) / memory._safe_name(prior["part_name"]) / memory._safe_name(batch_id) / "override_evidence"
    root.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower() if source.suffix.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
    target = root / f"override-{uuid.uuid4().hex}{suffix}"
    shutil.copy2(source, target)
    return str(target.resolve())


class EvidenceGatedDecision:
    def request_override(self, inspection_id: str, operator_id: str,
                         justification: str, new_evidence_image: str | None = None) -> dict:
        prior = memory.get_inspection(inspection_id)
        if prior is None:
            raise ValueError(f"No inspection found for id {inspection_id}")

        claim = _claim_from_prior(prior)
        if not new_evidence_image:
            record = memory.log_override(
                inspection_id=inspection_id,
                operator_id=operator_id,
                new_evidence_image=None,
                justification=justification,
                accepted=False,
                original_claim=claim,
                evidence_adequacy={"evidence_adequate": False, "adequacy_reason": "No new physical evidence supplied."},
                adjudication="INSUFFICIENT",
                evidence_adequacy_threshold=EVIDENCE_ADEQUACY_CONFIDENCE_THRESHOLD,
                final_assessment=prior["assessment"],
            )
            return {
                "status": "OVERRIDE_BLOCKED_NO_NEW_EVIDENCE",
                "reason": "Override blocked: a justification alone cannot overturn an inspection claim.",
                "original_claim": claim,
                "adjudication": "INSUFFICIENT",
                "override_record": record,
                "final_assessment": prior["assessment"],
            }

        source_evidence = Path(new_evidence_image).resolve()
        if prior.get("image_sha256") and _sha256(source_evidence) == prior.get("image_sha256"):
            record = memory.log_override(
                inspection_id=inspection_id,
                operator_id=operator_id,
                new_evidence_image=str(source_evidence),
                justification=justification,
                accepted=False,
                original_claim=claim,
                evidence_adequacy={
                    "evidence_adequate": False,
                    "adequacy_reason": "The supplied file is byte-identical to the original inspection evidence.",
                },
                adjudication="INSUFFICIENT",
                final_assessment=prior["assessment"],
            )
            return {
                "status": "OVERRIDE_BLOCKED_NOT_NEW_EVIDENCE",
                "reason": "Override blocked: the supplied image is identical to the original inspection evidence.",
                "original_claim": claim,
                "adjudication": "INSUFFICIENT",
                "override_record": record,
                "final_assessment": prior["assessment"],
            }

        archived_new_evidence = _archive_override_evidence(prior, new_evidence_image)
        reference = memory.get_reference(prior["part_name"])
        images = []
        if reference and reference.get("reference_image"):
            images.append(reference["reference_image"])
        if prior.get("image_path"):
            images.append(prior["image_path"])
        images.append(archived_new_evidence)

        adequacy = query_evidence_adequacy(images, claim, justification)
        adequate = (
            bool(adequacy.get("evidence_adequate"))
            and int(adequacy.get("confidence", 0)) >= EVIDENCE_ADEQUACY_CONFIDENCE_THRESHOLD
        )
        if not adequate:
            record = memory.log_override(
                inspection_id=inspection_id,
                operator_id=operator_id,
                new_evidence_image=archived_new_evidence,
                justification=justification,
                accepted=False,
                original_claim=claim,
                evidence_adequacy=adequacy,
                adjudication="INSUFFICIENT",
                final_assessment=prior["assessment"],
            )
            return {
                "status": "OVERRIDE_BLOCKED_INADEQUATE_EVIDENCE",
                "reason": adequacy.get("adequacy_reason") or "New evidence does not adequately address the original claim.",
                "original_claim": claim,
                "evidence_adequacy": adequacy,
                "adjudication": "INSUFFICIENT",
                "override_record": record,
                "final_assessment": prior["assessment"],
            }

        adjudication = query_claim_adjudication(images, claim, justification, adequacy)
        verdict = adjudication.get("adjudication", "INSUFFICIENT")
        confidence = int(adjudication.get("confidence", 0))

        can_change = (
            verdict == "REFUTE"
            and confidence >= ADJUDICATION_CONFIDENCE_THRESHOLD
            and prior.get("assessment") in {"ACCEPT", "REJECT"}
        )
        final_assessment = prior["assessment"]
        if can_change:
            final_assessment = "ACCEPT" if prior["assessment"] == "REJECT" else "REJECT"

        status = (
            "OVERRIDE_ACCEPTED_CLAIM_REFUTED" if can_change
            else "OVERRIDE_REJECTED_CLAIM_SUPPORTED" if verdict == "SUPPORT"
            else "OVERRIDE_BLOCKED_INSUFFICIENT_ADJUDICATION"
        )
        record = memory.log_override(
            inspection_id=inspection_id,
            operator_id=operator_id,
            new_evidence_image=archived_new_evidence,
            justification=justification,
            accepted=can_change,
            original_claim=claim,
            evidence_adequacy=adequacy,
            adjudication=verdict,
            adjudication_detail=adjudication,
            adjudication_threshold=ADJUDICATION_CONFIDENCE_THRESHOLD,
            final_assessment=final_assessment,
            adjudicated_at=time.time(),
        )
        return {
            "status": status,
            "original_claim": claim,
            "evidence_adequacy": adequacy,
            "adjudication": verdict,
            "adjudication_detail": adjudication,
            "override_record": record,
            "final_assessment": final_assessment,
        }

    def get_audit_trail(self, inspection_id: str) -> dict:
        return {
            "inspection": memory.get_inspection(inspection_id),
            "overrides": memory.get_override_history(inspection_id),
        }
