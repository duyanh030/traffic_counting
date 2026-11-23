import argparse
import json
import os
import sys
from pathlib import Path
from ultralytics import YOLO

import config
from track_loss_diagnoser import diagnose_candidate_frame


def main():
    parser = argparse.ArgumentParser(
        description="Automated Root-Cause Diagnostic CLI for Track Loss and Low Confidence Candidates"
    )
    parser.add_argument(
        "target",
        help="Path to candidate metadata .json file (or .jpg image file with matching metadata)",
    )
    parser.add_argument(
        "-m", "--model",
        default=config.MODEL_NAME,
        help=f"YOLO weights path (default: {config.MODEL_NAME})",
    )
    args = parser.parse_args()

    target_path = Path(args.target)
    if not target_path.exists():
        print(f"❌ Error: File not found at '{args.target}'!")
        sys.exit(1)

    if target_path.suffix == ".json":
        meta_path = target_path
    else:
        meta_path = target_path.with_suffix(".json")
        if not meta_path.exists():
            meta_path = target_path.parent.parent / "metadata" / f"{target_path.stem}.json"

    if not meta_path.exists():
        print(f"❌ Error: Could not locate corresponding metadata JSON for '{args.target}'!")
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    run_dir = meta_path.parent.parent
    img_rel = meta.get("image_file") or f"images/{meta.get('candidate_id')}.jpg"
    img_path = run_dir / img_rel
    if not img_path.exists():
        prev_rel = meta.get("preview_file") or f"previews/{meta.get('candidate_id')}.jpg"
        img_path = run_dir / prev_rel

    if not img_path.exists():
        print(f"❌ Error: Raw image file not found at '{img_path}'!")
        sys.exit(1)

    print(f"🔍 Diagnosing Candidate: {meta.get('candidate_id')}")
    print(f"📄 Initial Alert Reason: {meta.get('reason')} (Frame #{meta.get('frame_idx')})")
    print(f"📷 Image File: {img_path}")

    model = YOLO(args.model)
    diag = diagnose_candidate_frame(
        frame_input=str(img_path),
        details=meta.get("details", {}),
        model=model,
    )

    print("\n" + "=" * 65)
    print(f"🩺 DIAGNOSTIC VERDICT: [ {diag.get('verdict')} ]")
    rec = diag.get("recommendation")
    rec_icon = "🟢" if rec == "APPROVE" else "⚪" if rec == "REJECT" else "🟠"
    print(f"👉 ACTION RECOMMENDATION: {rec_icon} {rec}")
    print(f"📌 Title: {diag.get('title')}")
    print(f"💡 Root Cause Analysis: {diag.get('explanation')}")
    if diag.get("overlap_ratio") is not None:
        print(f"📊 Occlusion Coverage: {diag.get('overlap_ratio')*100:.1f}% (By: {diag.get('occluder')})")
    if diag.get("detected_conf") is not None:
        print(f"🎯 Re-detected Confidence: {diag.get('detected_conf'):.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
