"""Orchestration entry point for the industrial inspection process modules."""
from __future__ import annotations

from . import industrial_memory as memory
from .learn_engine import LearnEngine
from .inspection_engine import InspectionEngine
from .self_verifying_inspection import SelfVerifyingInspection
from .evidence_gated_decision import EvidenceGatedDecision

learn_engine = LearnEngine()
inspection_engine = InspectionEngine()
self_verifier = SelfVerifyingInspection()
evidence_gate = EvidenceGatedDecision()


def _analytics(inspection_id: str | None) -> dict:
    return memory.get_episode_analytics(inspection_id) if inspection_id else {}


def run_inspection_pipeline(roi_image_path: str, part_name: str, operator_query: str = "") -> dict:
    if not learn_engine.has_reference(part_name):
        return {
            "decision": "PENDING_TEACHING",
            "message": f"'{part_name}' has no reference yet. Use LearnEngine.teach_new_part() first.",
        }

    first_pass = inspection_engine.inspect(roi_image_path, part_name, operator_query)
    if first_pass["routed_to"] == "LEARN_ENGINE":
        return {"decision": "PENDING_TEACHING", "message": first_pass["result"]["uncertainty"]}

    if first_pass["routed_to"] == "FINAL":
        return _finalize(first_pass["inspection_id"], first_pass["result"])

    return {
        "decision": "AWAITING_OPERATOR_EVIDENCE",
        "inspection_id": first_pass["inspection_id"],
        "requested_evidence": self_verifier.request_more_evidence(first_pass["result"]),
        "preliminary_result": first_pass["result"],
        "memory_analytics": _analytics(first_pass["inspection_id"]),
    }


def continue_with_new_evidence(inspection_id: str, new_image_path: str) -> dict:
    reeval = self_verifier.reevaluate(inspection_id, new_image_path)
    if reeval["status"] == "RESOLVED":
        return _finalize(reeval["inspection_id"], reeval["result"])
    if reeval["status"] == "STILL_UNCERTAIN":
        return {
            "decision": "AWAITING_OPERATOR_EVIDENCE",
            "inspection_id": reeval["inspection_id"],
            "requested_evidence": reeval["next_action"],
            "preliminary_result": reeval["result"],
            "memory_analytics": _analytics(reeval["inspection_id"]),
        }
    return {
        "decision": "ESCALATE_TO_OPERATOR",
        "inspection_id": reeval["inspection_id"],
        "result": reeval["result"],
        "memory_analytics": _analytics(reeval["inspection_id"]),
    }


def request_operator_override(inspection_id: str, operator_id: str,
                              justification: str, new_evidence_image: str | None = None) -> dict:
    outcome = evidence_gate.request_override(inspection_id, operator_id, justification, new_evidence_image)
    return {
        "decision": outcome["final_assessment"],
        "inspection_id": inspection_id,
        "override_status": outcome["status"],
        "adjudication": outcome.get("adjudication"),
        "original_claim": outcome.get("original_claim"),
        "evidence_adequacy": outcome.get("evidence_adequacy"),
        "adjudication_detail": outcome.get("adjudication_detail"),
        "memory_analytics": _analytics(inspection_id),
        "details": outcome,
    }


def _finalize(inspection_id: str, result: dict) -> dict:
    decision = "ACCEPT" if result["assessment"] == "ACCEPT" else (
        "REJECT" if result["assessment"] == "REJECT" else "ESCALATE_TO_OPERATOR"
    )
    return {
        "decision": decision,
        "inspection_id": inspection_id,
        "confidence": result["confidence"],
        "reasoning": result["visual_evidence"],
        "recommended_action": result["action"],
        "memory_analytics": _analytics(inspection_id),
    }
