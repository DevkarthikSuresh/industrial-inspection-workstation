"""
learn_engine.py
----------------
Process Module A: Learn Engine (Section I.II, point 3).
"Enables interactive teaching of new parts and inspection references."

Workflow (matches the architecture diagram's Learn Engine box):
  1. Operator shows a new part + optionally gives a voice/text instruction.
  2. VLM is asked to describe the part in structured, reference-grade detail
     (not defect-checking yet -- this is establishing the "normal" baseline).
  3. If the VLM needs a better view ("Rotate bolt, need underside view"),
     the engine reports that back so the UI can prompt the operator, rather
     than silently saving a bad reference.
  4. Once a usable description is produced, it's stored via industrial_memory
     so the Inspection Engine can compare future captures against it.
"""

from .vlm_client import query_vlm
from . import industrial_memory as memory

TEACH_PROMPT_TEMPLATE = """You are teaching mode, not defect-checking mode.
The operator is showing you a KNOWN-GOOD reference sample of a component
named "{part_name}". Produce a reference description covering: shape,
material finish, thread/edge condition, color, and any identifying marks.
If the current view is insufficient to fully characterize the part (e.g.
underside/threads not visible), set assessment to "INDETERMINATE" and put
the specific view you need in the "action" field (e.g. "Rotate bolt,
need underside view").
"""


class LearnEngine:
    def teach_new_part(self, image_path: str, part_name: str,
                        operator_note: str = "") -> dict:
        """
        Attempts to learn a new part from one image. Returns a dict:
          {status: "SAVED" | "NEEDS_MORE_VIEWS", detail: <vlm result>}
        """
        prompt = TEACH_PROMPT_TEMPLATE.format(part_name=part_name)
        if operator_note:
            prompt += f"\nOperator note: {operator_note}"

        result = query_vlm(image_path, prompt)

        if result["assessment"] == "INDETERMINATE":
            return {"status": "NEEDS_MORE_VIEWS", "detail": result}

        memory.save_reference(
            part_name=part_name,
            reference_image=image_path,
            description=result["visual_evidence"],
        )
        return {"status": "SAVED", "detail": result}

    def add_alternate_view(self, image_path: str, part_name: str) -> dict:
        """
        Adds a supplementary view (e.g. after the VLM requested a rotation)
        by re-running teach_new_part with the follow-up image, so multi-turn
        teaching (as shown in the diagram's infographic loop) converges on
        one saved reference.
        """
        return self.teach_new_part(
            image_path, part_name,
            operator_note="This is a follow-up view requested for the same part."
        )

    def has_reference(self, part_name: str) -> bool:
        return memory.has_seen_before(part_name)
