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
        description="CLI Chẩn đoán tự động nguyên nhân Track Loss / Low Confidence của Candidate"
    )
    parser.add_argument(
        "target",
        help="Đường dẫn đến file metadata .json của Candidate (hoặc file ảnh .jpg kèm file json cùng tên)",
    )
    parser.add_argument(
        "-m", "--model",
        default=config.MODEL_NAME,
        help=f"File trọng số YOLO (Mặc định: {config.MODEL_NAME})",
    )
    args = parser.parse_args()

    target_path = Path(args.target)
    if not target_path.exists():
        print(f"❌ Lỗi: Không tìm thấy file tại '{args.target}'!")
        sys.exit(1)

    if target_path.suffix == ".json":
        meta_path = target_path
    else:
        # Tìm file json tương ứng
        meta_path = target_path.with_suffix(".json")
        if not meta_path.exists():
            meta_path = target_path.parent.parent / "metadata" / f"{target_path.stem}.json"

    if not meta_path.exists():
        print(f"❌ Lỗi: Không tìm thấy file metadata JSON tương ứng với '{args.target}'!")
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Tìm file ảnh
    run_dir = meta_path.parent.parent
    img_rel = meta.get("image_file") or f"images/{meta.get('candidate_id')}.jpg"
    img_path = run_dir / img_rel
    if not img_path.exists():
        # Thử tìm trực tiếp ở preview
        prev_rel = meta.get("preview_file") or f"previews/{meta.get('candidate_id')}.jpg"
        img_path = run_dir / prev_rel

    if not img_path.exists():
        print(f"❌ Lỗi: Không tìm thấy file ảnh gốc tại '{img_path}'!")
        sys.exit(1)

    print(f"🔍 Đang chẩn đoán Candidate: {meta.get('candidate_id')}")
    print(f"📄 Lý do ghi nhận ban đầu: {meta.get('reason')} (Frame #{meta.get('frame_idx')})")
    print(f"📷 File ảnh: {img_path}")

    model = YOLO(args.model)
    diag = diagnose_candidate_frame(
        frame_input=str(img_path),
        details=meta.get("details", {}),
        model=model,
    )

    print("\n" + "=" * 65)
    print(f"🩺 KẾT QUẢ CHẨN ĐOÁN: [ {diag.get('verdict')} ]")
    rec = diag.get("recommendation")
    rec_icon = "🟢" if rec == "APPROVE" else "⚪" if rec == "REJECT" else "🟠"
    print(f"👉 KHUYẾN NGHỊ HÀNH ĐỘNG: {rec_icon} {rec}")
    print(f"📌 Tiêu đề: {diag.get('title')}")
    print(f"💡 Giải thích chi tiết: {diag.get('explanation')}")
    if diag.get("overlap_ratio") is not None:
        print(f"📊 Tỷ lệ che phủ: {diag.get('overlap_ratio')*100:.1f}% (Bởi: {diag.get('occluder')})")
    if diag.get("detected_conf") is not None:
        print(f"🎯 Điểm tin cậy phát hiện lại: {diag.get('detected_conf'):.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
