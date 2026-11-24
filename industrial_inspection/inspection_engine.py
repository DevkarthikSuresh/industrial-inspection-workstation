"""
inspection_engine.py
---------------------
Process Module B: Inspection and Explainability Engine (Section I.II, point 3).
"Compares captured parts with stored references, detects defects, and
provides confidence scores and explanations."

This is the module your current VLMtest.py prototype most closely maps to
(Gemini 3.5 Flash-Lite + structured system prompt). Here it's formalized to always
compare against the saved reference from the Learn Engine, and to hand off
low-confidence results to the Self-Verifying Inspection module rather than
returning them directly.
"""

from .vlm_client import query_vlm
from . import industrial_memory as memory

CONFIDENCE_THRESHOLD = 80  # below this, route to Self-Verifying Inspection

INSPECT_PROMPT_TEMPLATE = """Compare the current component image against this
known-good reference description for "{part_name}":

REFERENCE: {reference_description}

Query from operator: {operator_query}

Two images are attached in this order:
1. KNOWN-GOOD reference image.
2. CURRENT part under inspection.

Assess whether the CURRENT part matches the known-good reference or shows a defect
(damaged threads, corrosion, cracks, geometry inconsistency, etc). Report
your confidence (0-100), the visual evidence you observed, any uncertainty,
your assessment, and a recommended action.
"""


class InspectionEngine:
    def inspect(self, image_path: str, part_name: str, operator_query: str = "") -> dict:
        """
        Runs a full inspection against the stored reference.
        Returns {inspection_id, routed_to, result}.
        `routed_to` is "FINAL" if confidence is high enough to go straight
        to the Decision Engine, or "SELF_VERIFYING" if it needs re-evaluation.
        """
        reference = memory.get_reference(part_name)
        if reference is None:
            # No baseline yet -- this should be routed to Learn Engine by
            # the Task Planner, but guard here too.
            return {
                "inspection_id": None,
                "routed_to": "LEARN_ENGINE",
                "result": {
                    "confidence": 0,
                    "assessment": "INDETERMINATE",
                    "visual_evidence": "",
                    "uncertainty": f"No reference exists yet for '{part_name}'.",
                    "action": "Teach this part via the Learn Engine first.",
                },
            }

        prompt = INSPECT_PROMPT_TEMPLATE.format(
            part_name=part_name,
            reference_description=reference["description"],
            operator_query=operator_query or "General inspection.",
        )
        result = query_vlm([reference["reference_image"], image_path], prompt)

        inspection_id = memory.log_inspection(part_name, image_path, result)

        routed_to = "FINAL" if result["confidence"] >= CONFIDENCE_THRESHOLD else "SELF_VERIFYING"
        return {"inspection_id": inspection_id, "routed_to": routed_to, "result": result}
