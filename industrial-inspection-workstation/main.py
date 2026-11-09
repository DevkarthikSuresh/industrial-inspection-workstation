"""Command-line entry point for the complete industrial inspection project."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from industrial_inspection.decision_engine import run_inspection_pipeline, continue_with_new_evidence, request_operator_override
from industrial_inspection.learn_engine import LearnEngine
from industrial_inspection.vision_localizer import localize_and_crop_object
from industrial_inspection import industrial_memory as memory


def _print(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _roi(image: str, no_yolo: bool = False) -> tuple[str, dict]:
    stem = Path(image).stem
    crop = str(Path(__file__).resolve().parent / "runtime" / f"{stem}_roi.jpg")
    return localize_and_crop_object(image, crop, use_yolo=not no_yolo)


def cmd_teach(args):
    crop, localization = _roi(args.image, args.no_yolo)
    result = LearnEngine().teach_new_part(crop, args.part, args.note)
    _print({"localization": localization, "teaching": result})


def cmd_inspect(args):
    crop, localization = _roi(args.image, args.no_yolo)
    result = run_inspection_pipeline(crop, args.part, args.query)
    _print({"localization": localization, "pipeline": result})


def cmd_continue(args):
    crop, localization = _roi(args.image, args.no_yolo)
    result = continue_with_new_evidence(args.inspection_id, crop)
    _print({"localization": localization, "pipeline": result})


def cmd_override(args):
    evidence = None
    localization = None
    if args.image:
        evidence, localization = _roi(args.image, args.no_yolo)
    result = request_operator_override(args.inspection_id, args.operator, args.justification, evidence)
    _print({"localization": localization, "pipeline": result})


def cmd_new_batch(args):
    _print(memory.start_new_batch(args.part, args.label))


def cmd_memory_status(args):
    _print(memory.get_memory_snapshot(args.part))


def cmd_wipe_memory(args):
    if args.confirm != "WIPE":
        raise SystemExit('Refusing destructive wipe. Re-run with: wipe-memory --confirm WIPE')
    _print(memory.wipe_all_memory())


def build_parser():
    p = argparse.ArgumentParser(description="Industrial VLM inspection pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--image", required=True, help="Input image path")
    common.add_argument("--no-yolo", action="store_true", help="Use contour localization only")

    teach = sub.add_parser("teach", parents=[common], help="Teach a known-good part")
    teach.add_argument("--part", required=True)
    teach.add_argument("--note", default="")
    teach.set_defaults(func=cmd_teach)

    inspect = sub.add_parser("inspect", parents=[common], help="Inspect a part")
    inspect.add_argument("--part", required=True)
    inspect.add_argument("--query", default="General inspection")
    inspect.set_defaults(func=cmd_inspect)

    cont = sub.add_parser("continue", parents=[common], help="Provide requested new evidence")
    cont.add_argument("--inspection-id", required=True)
    cont.set_defaults(func=cmd_continue)

    override = sub.add_parser("override", help="Request a claim-adjudicated operator override")
    override.add_argument("--inspection-id", required=True)
    override.add_argument("--operator", required=True)
    override.add_argument("--justification", required=True)
    override.add_argument("--image", help="New physical evidence image")
    override.add_argument("--no-yolo", action="store_true")
    override.set_defaults(func=cmd_override)

    batch = sub.add_parser("new-batch", help="Start a fresh batch-scoped memory window")
    batch.add_argument("--part", required=True)
    batch.add_argument("--label", default="")
    batch.set_defaults(func=cmd_new_batch)

    status = sub.add_parser("memory-status", help="Show active batch and latest drift metrics")
    status.add_argument("--part", required=True)
    status.set_defaults(func=cmd_memory_status)

    wipe = sub.add_parser("wipe-memory", help="Delete all Industrial Memory")
    wipe.add_argument("--confirm", default="", help='Must be exactly "WIPE"')
    wipe.set_defaults(func=cmd_wipe_memory)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
