# Recorded inspection snapshots

This project archive contained **two real saved inspection episodes** for a bolt. The table below preserves their reported outputs while removing internal Windows paths, raw embedding arrays, unique IDs, and machine-specific state. These are **observations**, not benchmark measurements or ground-truth defect labels.

| Archived input | Recorded assessment | VLM-reported confidence | Visual drift | Semantic drift | Memory concordance |
| --- | --- | ---: | ---: | ---: | --- |
| [Corroded bolt](../assets/samples/corroded-bolt.jpg) | `REJECT` | `95/100` | `0.173883` | `0.447238` | `DISAGREEMENT` |
| [Fractured bolt](../assets/samples/fractured-bolt.jpg) | `REJECT` | `100/100` | `0.192296` | `0.495509` | `DISAGREEMENT` |

The model's stored rationales identify severe corrosion in the first image and a fully sheared bolt in the second. The `DISAGREEMENT` memory states come from **mismatched drift-trigger states**: semantic drift exceeds its prototype threshold (`0.25`) while visual drift is below its threshold (`0.25`), even though the image/rationale cross-modal similarity is above its configured cutoff. Consequently, `DISAGREEMENT` does **not** mean either recorded decision has been proven incorrect.

These examples are interesting precisely because high model-reported confidence and a memory disagreement flag can coexist. Whether the decisions are *accurate* cannot be established from two unlabelled stored examples.

The sanitized structured records, including the complete original rationales, are in [`examples/recorded_inspections.json`](../examples/recorded_inspections.json). The raw memory database is intentionally excluded from Git because it includes local paths and mutable runtime/inspection state.
