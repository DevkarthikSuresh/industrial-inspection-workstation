# Validation and evidence

## What is verified offline

`tests/run_mock_tests.py` executes without sending any API requests. It uses fake embedding vectors and stubbed adjudication responses to exercise the real episode store and override control flow.

The test suite covers:

- known-good teaching and batch creation;
- image archive classes and SHA-256 episode integrity;
- non-mutable returned inspection objects;
- visual and semantic drift calculations;
- image/text disagreement signals;
- rejection of byte-identical follow-up imagery;
- confidence / adequacy gates and `SUPPORT`, `REFUTE`, `INSUFFICIENT` paths;
- wiping the local memory store.

Run with:

```bash
python -m compileall -q app.py main.py check_setup.py industrial_inspection tests
python tests/run_mock_tests.py
```

## What is **not** claimed

The project archive includes two saved inference records (preserved in `examples/recorded_inspections.json`) but does not contain a labeled benchmark split, a confusion matrix, calibration results, or verified model-vs-model comparative performance. Accordingly, this repository does **not** claim production accuracy, precision, recall, throughput, or numerical superiority for the VLM. Images under `assets/samples/` are supplied example inputs, **not** a benchmark dataset, and the reference/defect gallery is not a screenshot of an inference result.

`check_setup.py --api` optionally performs live API calls and may incur usage costs. This is separate from the deterministic offline tests.
