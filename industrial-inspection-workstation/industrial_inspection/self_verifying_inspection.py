"""
self_verifying_inspection.py
------------------------------
Process Module C: Self-Verifying Inspection (Section I.II, point 3).
"Handles uncertain inspections by requesting additional information/evidence
such as different viewing angle or improved lighting, then performs
re-evaluation."

This directly addresses two of your "Challenges Faced" (II.I): high VLM
latency (by only triggering the extra round-trip when confidence is
genuinely low) and hallucinations/overconfidence (by forcing a second,
evidence-anchored pass instead of trusting a single low-confidence read).
"""

from .vlm_client import query_vlm
from . import industrial_memory as memory

MAX_REVERIFY_ATTEMPTS = 2

REVERIFY_PROMPT_TEMPLATE = """This is a RE-EVALUATION. The initial inspection
of "{part_name}" was uncertain:

  Previous confidence: {prev_confidence}
  Previous uncertainty: {prev_uncertainty}
  Reference description: {reference_description}

Two images are attached in this order: (1) known-good reference and
(2) newly captured evidence ({evidence_note}). Re-assess the new evidence
against the known-good image. If confidence is still low, state precisely what additional evidence
would resolve it next.
"""


class SelfVerifyingInspection:
    def request_more_evidence(self, prior_result: dict) -> str:
        """
        Builds the operator-facing prompt for what new evidence is needed,
        taken straight from the VLM's own 'action' field (e.g. "Rotate bolt
        90 degrees" / "improve lighting") so the request stays grounded in
        what the model actually flagged, not a generic template.
        """
        return prior_result.get("action") or "Please provide a clearer image or different angle."

    def reevaluate(self, inspection_id: str, new_image_path: str,
                    evidence_note: str = "operator provided new angle/lighting") -> dict:
        """
        Re-runs inspection using new evidence, chained to the original
        inspection_id in memory. Will recurse into another self-verification
        round if still uncertain, up to MAX_REVERIFY_ATTEMPTS, to avoid an
        infinite low-confidence loop eating into real-time throughput.
        """
        prior = memory.get_inspection(inspection_id)
        if prior is None:
            raise ValueError(f"No prior inspection found for id {inspection_id}")

        reference = memory.get_reference(prior["part_name"])
        prompt = REVERIFY_PROMPT_TEMPLATE.format(
            part_name=prior["part_name"],
            prev_confidence=prior["confidence"],
            prev_uncertainty=prior["uncertainty"],
            reference_description=reference["description"] if reference else "N/A",
            evidence_note=evidence_note,
        )

        images = [reference["reference_image"], new_image_path] if reference else new_image_path
        result = query_vlm(images, prompt)
        new_inspection_id = memory.log_inspection(
            prior["part_name"], new_image_path, result,
            parent_inspection_id=inspection_id,
            batch_id=prior.get("batch_id"),
        )

        attempt_count = self._count_chain_depth(new_inspection_id)
        if result["confidence"] < 80 and attempt_count < MAX_REVERIFY_ATTEMPTS:
            return {
                "inspection_id": new_inspection_id,
                "status": "STILL_UNCERTAIN",
                "result": result,
                "next_action": self.request_more_evidence(result),
            }

        status = "RESOLVED" if result["confidence"] >= 80 else "ESCALATE_TO_OPERATOR"
        return {"inspection_id": new_inspection_id, "status": status, "result": result}

    def _count_chain_depth(self, inspection_id: str) -> int:
        depth = 0
        current = memory.get_inspection(inspection_id)
        while current and current.get("parent_inspection_id"):
            depth += 1
            current = memory.get_inspection(current["parent_inspection_id"])
        return depth
