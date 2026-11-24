<!-- Historical notes from original supplied ZIP. File locations and setup commands may have changed. See root README for current instructions. -->

# Industrial Inspection Workstation — Multilayer Memory + Evidence-Gated Overrides

This package continues the existing Gemini/NiceGUI industrial inspection prototype while preserving browser upload, ROI localization, teaching, self-verification, and the existing operator UI.

## What changed

### A. Multilayer Industrial Memory

`industrial_memory.py` now maintains five cooperating layers:

1. **Reference / semantic memory** — taught known-good images and descriptions.
2. **Immutable inspection episodes** — every inspection attempt is appended as a new episode. Its image is copied into a permanent archive and the episode receives an integrity hash. Existing episodes are never updated in place by the application.
3. **Batch-scoped memory** — each part has an active production batch. Drift baselines prefer accepted episodes from the current batch and fall back to the taught known-good reference.
4. **Embedding / drift memory** — `gemini-embedding-2` embeds the archived image and VLM rationale into the same multimodal space (768 dimensions by default). The memory layer computes:
   - visual drift,
   - semantic drift,
   - image↔rationale cross-modal similarity,
   - `CONCORDANT`, `DISAGREEMENT`, or `UNAVAILABLE` status.
5. **Audit memory** — explainability logs and evidence-gated override events.

Inspection images are archived under:

```text
memory_archive/<part>/<batch>/good/
memory_archive/<part>/<batch>/bad/
memory_archive/<part>/<batch>/indeterminate/
memory_archive/<part>/<batch>/override_evidence/
```

An embedding API failure is **non-fatal**. The inspection episode is still archived and logged; its embedding status is recorded as `UNAVAILABLE`.

### B. Enhanced Evidence-Gated Decision Protocol

An override is no longer a generic second classification pass.

The protocol is now:

```text
Original ACCEPT/REJECT rationale
        ↓ becomes an explicit claim
Operator requests override
        ↓
New physical evidence present?
        ├─ no  → block / INSUFFICIENT
        ↓ yes
Does the new evidence actually address the claim?
        ├─ no  → block / INSUFFICIENT
        ↓ yes
Gemini claim adjudication
        ├─ SUPPORT      → keep original decision
        ├─ INSUFFICIENT → keep original decision
        └─ REFUTE       → change decision only if confidence ≥ 80
```

The VLM is explicitly instructed **not** to perform a fresh whole-part classification during override adjudication.

## UI controls

The existing NiceGUI application now includes an **Industrial Memory** panel containing the active batch, episode count, latest visual drift, semantic drift, and cross-modal state.

- **New Batch** starts a fresh batch scope for the active part and clears pending per-session inspection IDs.
- **Wipe All Memory** opens a confirmation dialog. Confirming permanently removes taught references, inspection episodes, image archives, embeddings, batches, and override audit history.

The existing browser upload and query interface is preserved.

Useful chat commands:

```text
/part bolt       set active part
/teach bolt      teach the uploaded image as a known-good reference
/evidence        use the uploaded image as requested follow-up evidence
/override reason request an evidence-gated override using the uploaded image
/newbatch        start a fresh batch for the active part
```

## Setup

Python 3.10+ is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python check_setup.py
```

Run an optional live Gemini + embedding connectivity test:

```powershell
python check_setup.py --api
```

For this public packaging, API keys must be provided through the environment or a local `.env` file.

## Run the UI

```bash
python VLMtest_integrated.py
```

Open `http://localhost:8880` if the browser does not open automatically.

## CLI examples

Teach and inspect:

```bash
python main.py teach --image good_bolt.jpg --part bolt --no-yolo
python main.py inspect --image test_bolt.jpg --part bolt --query "check the threads" --no-yolo
```

Start a new batch and inspect memory status:

```bash
python main.py new-batch --part bolt --label "Shift B lot 14"
python main.py memory-status --part bolt
```

Evidence-gated override:

```bash
python main.py override \
  --inspection-id YOUR_ID \
  --operator operator_1 \
  --justification "new close view shows the apparent crack is a reflection" \
  --image new_close_view.jpg \
  --no-yolo
```

Destructive CLI wipe requires an explicit token:

```bash
python main.py wipe-memory --confirm WIPE
```

## Tests

Offline tests do not call Gemini. They mock embeddings and claim adjudication while exercising the real persistence and decision-control code.

```bash
python -m py_compile *.py
python tests/run_mock_tests.py
```

The mock test covers immutable episode behavior, image archiving, batch scope, drift metrics, cross-modal disagreement, the new override protocol, import smoke, and wipe-all-memory.

## Files

- `VLMtest_integrated.py` — NiceGUI frontend and browser upload flow
- `vision_localizer.py` — YOLO / contour ROI localization
- `learn_engine.py` — known-good teaching
- `inspection_engine.py` — reference-vs-current inspection
- `self_verifying_inspection.py` — low-confidence evidence loop
- `decision_engine.py` — process orchestration
- `evidence_gated_decision.py` — claim adequacy + SUPPORT/REFUTE/INSUFFICIENT adjudication
- `industrial_memory.py` — multilayer persistent memory
- `embedding_client.py` — Gemini Embedding 2 adapter
- `vlm_client.py` — Gemini structured VLM adapter
- `main.py` — CLI harness
- `check_setup.py` — dependency / model / folder check
- `tests/run_mock_tests.py` — offline mock tests

## Notes

The bundled YOLOv8n model is a generic COCO detector, not a component-specific industrial detector. During simple bench testing, contour localization can be more reliable; use `--no-yolo` from the CLI when appropriate.

Images analyzed by Gemini or Gemini Embedding 2 leave the local machine and are sent to the configured API. Do not use confidential industrial imagery unless your deployment and data terms permit it.

### Calibration note

The packaged drift/concordance thresholds are prototype defaults, not plant-qualified tolerances. Calibrate `VISUAL_DRIFT_THRESHOLD`, `SEMANTIC_DRIFT_THRESHOLD`, and `CROSS_MODAL_CONCORDANCE_THRESHOLD` on labeled data from the actual camera, part family, lighting, and defect taxonomy before using them as production alarms.
