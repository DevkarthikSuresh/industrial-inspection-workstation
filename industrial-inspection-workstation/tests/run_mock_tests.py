"""Offline smoke tests for the packaged project.

No network/Gemini calls are made.  google.genai is stubbed and the embedding /
adjudication functions are mocked so the persistence and control logic can be
validated deterministically.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import types as pytypes
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def install_google_stub():
    google = pytypes.ModuleType("google")
    genai = pytypes.ModuleType("google.genai")
    types_mod = pytypes.ModuleType("google.genai.types")
    genai.types = types_mod
    google.genai = genai
    sys.modules.setdefault("google", google)
    sys.modules.setdefault("google.genai", genai)
    sys.modules.setdefault("google.genai.types", types_mod)


def write_img(path: Path, value: int):
    image = np.full((32, 32, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def run():
    install_google_stub()
    with tempfile.TemporaryDirectory(prefix="industrial-memory-test-") as td:
        root = Path(td)
        os.environ["INDUSTRIAL_MEMORY_ROOT"] = str(root / "memory")

        from industrial_inspection import embedding_client

        def fake_embed(image_path: str, rationale: str):
            mean = float(cv2.imread(image_path).mean())
            image_vec = [1.0, 0.0, 0.0] if mean > 100 else [0.0, 1.0, 0.0]
            text = (rationale or "").lower()
            rationale_vec = [0.0, 1.0, 0.0] if any(w in text for w in ("defect", "crack", "damage")) else [1.0, 0.0, 0.0]
            return {
                "model": "gemini-embedding-2",
                "dimension": 3,
                "image_embedding": image_vec,
                "rationale_embedding": rationale_vec,
            }

        embedding_client.embed_episode = fake_embed
        from industrial_inspection import industrial_memory as memory
        importlib.reload(memory)
        memory.embedding_client.embed_episode = fake_embed

        good = root / "good.jpg"
        bad = root / "bad.jpg"
        write_img(good, 220)
        write_img(bad, 20)

        memory.save_reference("bolt", str(good), "clean intact normal bolt")
        batch = memory.start_new_batch("bolt", "mock batch")

        good_id = memory.log_inspection("bolt", str(good), {
            "confidence": 95,
            "assessment": "ACCEPT",
            "visual_evidence": "clean intact normal bolt",
            "uncertainty": "",
            "action": "continue",
        })
        bad_id = memory.log_inspection("bolt", str(bad), {
            "confidence": 97,
            "assessment": "REJECT",
            "visual_evidence": "visible crack defect damage",
            "uncertainty": "",
            "action": "quarantine",
        })

        good_rec = memory.get_inspection(good_id)
        bad_rec = memory.get_inspection(bad_id)
        assert good_rec["batch_id"] == batch["batch_id"]
        assert good_rec["archive_class"] == "good"
        assert bad_rec["archive_class"] == "bad"
        assert "/good/" in good_rec["image_path"].replace("\\", "/")
        assert "/bad/" in bad_rec["image_path"].replace("\\", "/")
        assert memory.verify_episode_integrity(good_id)
        assert memory.verify_episode_integrity(bad_id)
        assert bad_rec["memory_metrics"]["visual_drift"] is not None
        assert bad_rec["memory_metrics"]["semantic_drift"] is not None
        assert bad_rec["memory_metrics"]["concordance"] == "CONCORDANT"

        # Returned records are copies; callers cannot mutate stored episodes.
        good_rec["assessment"] = "REJECT"
        assert memory.get_inspection(good_id)["assessment"] == "ACCEPT"

        # Intentional image/rationale mismatch should be marked disagreement.
        disagree_id = memory.log_inspection("bolt", str(bad), {
            "confidence": 70,
            "assessment": "INDETERMINATE",
            "visual_evidence": "clean intact normal bolt",
            "uncertainty": "view conflict",
            "action": "recheck",
        })
        assert memory.get_inspection(disagree_id)["memory_metrics"]["concordance"] == "DISAGREEMENT"

        snapshot = memory.get_memory_snapshot("bolt")
        assert snapshot["active_batch_id"] == batch["batch_id"]
        assert snapshot["batch_episodes"] == 3

        # Evidence-gated decision tests: no generic reclassification is involved.
        from industrial_inspection import evidence_gated_decision as egd
        importlib.reload(egd)
        prior_id = bad_id
        egd.query_evidence_adequacy = lambda images, claim, justification: {
            "evidence_adequate": True,
            "confidence": 94,
            "claim_area_visible": True,
            "same_physical_feature": True,
            "adequacy_reason": "claimed area is directly visible",
            "missing_evidence": "",
        }
        egd.query_claim_adjudication = lambda images, claim, justification, adequacy: {
            "adjudication": "REFUTE",
            "confidence": 93,
            "evidence_summary": "new view contradicts the apparent crack",
            "reasoning": "the line is a reflection, not a discontinuity",
            "required_next_evidence": "",
        }
        gate = egd.EvidenceGatedDecision()
        identical = gate.request_override(prior_id, "tester", "same frame should not count", str(bad))
        assert identical["status"] == "OVERRIDE_BLOCKED_NOT_NEW_EVIDENCE"
        assert identical["adjudication"] == "INSUFFICIENT"

        refuted = gate.request_override(prior_id, "tester", "new angle shows reflection", str(good))
        assert refuted["status"] == "OVERRIDE_ACCEPTED_CLAIM_REFUTED"
        assert refuted["final_assessment"] == "ACCEPT"
        assert refuted["adjudication"] == "REFUTE"

        egd.query_claim_adjudication = lambda images, claim, justification, adequacy: {
            "adjudication": "SUPPORT",
            "confidence": 96,
            "evidence_summary": "crack remains visible",
            "reasoning": "same discontinuity persists",
            "required_next_evidence": "",
        }
        supported = gate.request_override(prior_id, "tester", "please reconsider", str(good))
        assert supported["status"] == "OVERRIDE_REJECTED_CLAIM_SUPPORTED"
        assert supported["final_assessment"] == "REJECT"

        egd.query_evidence_adequacy = lambda images, claim, justification: {
            "evidence_adequate": False,
            "confidence": 90,
            "claim_area_visible": False,
            "same_physical_feature": False,
            "adequacy_reason": "claimed threads are occluded",
            "missing_evidence": "close view of the same threads",
        }
        inadequate = gate.request_override(prior_id, "tester", "different photo", str(good))
        assert inadequate["status"] == "OVERRIDE_BLOCKED_INADEQUATE_EVIDENCE"
        assert inadequate["adjudication"] == "INSUFFICIENT"
        assert inadequate["final_assessment"] == "REJECT"

        # Core module import smoke after the stubs are installed.
        for module_name in [
            "learn_engine", "inspection_engine", "self_verifying_inspection",
            "decision_engine", "vision_localizer"
        ]:
            importlib.import_module("industrial_inspection." + module_name)

        importlib.import_module("main")
        result = memory.wipe_all_memory()
        assert result["status"] == "WIPED"
        assert memory.get_inspection_history("bolt") == []
        assert memory.get_reference("bolt") is None
        assert memory.get_batches("bolt") == []

        print(json.dumps({
            "status": "PASS",
            "checks": [
                "append-only episode storage + integrity hash",
                "good/bad image archiving",
                "batch-scoped memory",
                "mock Gemini image/rationale embeddings",
                "visual + semantic drift",
                "cross-modal concordance/disagreement",
                "new-evidence identity gate + claim adequacy gate",
                "SUPPORT/REFUTE/INSUFFICIENT override protocol",
                "core import smoke",
                "wipe-all-memory",
            ],
        }, indent=2))


if __name__ == "__main__":
    run()
