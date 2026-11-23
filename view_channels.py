import argparse
import math
import os
import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from view_heatmap import get_model_layers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bóc tách và hiển thị toàn bộ các Channels con của một Layer trong YOLOv8"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="candidates/previews/cand_20261119_095701_f000185_track_loss.jpg",
        help="Đường dẫn đến file ảnh",
    )
    parser.add_argument(
        "-l", "--layer",
        type=int,
        default=15,
        help="Chỉ số Layer muốn bóc tách (Mặc định: 15 - P3 có 64 channels; có thể thử 2, 4, 6, 9, 18, 21)",
    )
    parser.add_argument(
        "-m", "--model",
        default="yolov8n.pt",
        help="File trọng số YOLO (Mặc định: yolov8n.pt)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Tên file ảnh đầu ra (Mặc định: channels_layer_<L>.jpg)",
    )
    parser.add_argument(
        "--cmap",
        choices=["gray", "jet", "inferno"],
        default="gray",
        help="Bản đồ màu cho channel: gray (đen trắng gốc), jet (nhiệt xanh-đỏ), inferno (lửa)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = args.image

    if not os.path.exists(image_path):
        print(f"❌ Lỗi: Không tìm thấy file ảnh tại '{image_path}'!")
        sys.exit(1)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Lỗi: Không thể đọc ảnh '{image_path}'!")
        sys.exit(1)

    # 1. Nạp mô hình YOLO
    model = YOLO(args.model)
    active_layers = get_model_layers(model)
    total_layers = len(active_layers)
    if args.layer < 0 or args.layer >= total_layers:
        print(f"❌ Lỗi: Layer {args.layer} không hợp lệ (mô hình có {total_layers} layers: 0 đến {total_layers-1})")
        sys.exit(1)

    target_layer = active_layers[args.layer]
    layer_name = target_layer.__class__.__name__

    # 2. Đăng ký Hook lấy Feature Map
    activations = {}
    def forward_hook(module, inp, out):
        if isinstance(out, (list, tuple)):
            activations["feat"] = out[0]
        else:
            activations["feat"] = out

    hooks = [target_layer.register_forward_hook(forward_hook)]
    if hasattr(model, "model") and hasattr(model.model, "model"):
        base_mod = model.model.model[args.layer]
        if base_mod is not target_layer:
            hooks.append(base_mod.register_forward_hook(forward_hook))

    # 3. Chạy qua mô hình
    print(f"🧠 Đang bóc tách Layer {args.layer} ({layer_name}) từ ảnh: {os.path.basename(image_path)}...")
    try:
        model(frame, verbose=False)
    finally:
        for h in hooks:
            h.remove()

    feat = activations.get("feat")
    if feat is None:
        print("❌ Lỗi: Không bắt được Feature Map từ layer này!")
        sys.exit(1)

    # feat shape: (1, C, H, W)
    if feat.ndim == 4:
        feat = feat.squeeze(0)  # (C, H, W)
    elif feat.ndim != 3:
        print(f"⚠️ Layer này có định dạng tensor không phù hợp để vẽ dạng 2D channel: shape = {feat.shape}")
        sys.exit(1)

    C, H, W = feat.shape
    print(f"📦 Layer {args.layer} ({layer_name}) có tổng cộng: {C} Channels (kích thước mỗi channel: {W}x{H})")

    feat_np = feat.detach().cpu().float().numpy()

    # 4. Tính toán số hàng và số cột cho lưới ảnh (Grid)
    cols = int(math.ceil(math.sqrt(C)))
    # Cân đối tỷ lệ khung hình
    if C == 64:
        rows, cols = 8, 8
    elif C == 32:
        rows, cols = 4, 8
    elif C == 128:
        rows, cols = 8, 16
    elif C == 256:
        rows, cols = 16, 16
    else:
        rows = int(math.ceil(C / cols))

    # Kích thước thumbnail của mỗi channel trên lưới
    thumb_w, thumb_h = 120, 80
    border = 2

    grid_h = rows * thumb_h + (rows + 1) * border
    grid_w = cols * thumb_w + (cols + 1) * border

    # Tạo canvas
    if args.cmap == "gray":
        canvas = np.zeros((grid_h, grid_w), dtype=np.uint8)
    else:
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    # 5. Điền từng channel vào lưới
    for idx in range(C):
        ch = feat_np[idx]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            ch_norm = ((ch - ch_min) / (ch_max - ch_min) * 255).astype(np.uint8)
        else:
            ch_norm = np.zeros_like(ch, dtype=np.uint8)

        ch_resized = cv2.resize(ch_norm, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)

        r = idx // cols
        c = idx % cols
        y = r * thumb_h + (r + 1) * border
        x = c * thumb_w + (c + 1) * border

        if args.cmap == "gray":
            # Ghi số thứ tự channel góc trên trái
            cv2.putText(ch_resized, f"Ch.{idx}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,), 1)
            canvas[y : y + thumb_h, x : x + thumb_w] = ch_resized
        else:
            colormap_code = cv2.COLORMAP_JET if args.cmap == "jet" else cv2.COLORMAP_INFERNO
            ch_color = cv2.applyColorMap(ch_resized, colormap_code)
            cv2.putText(ch_color, f"Ch.{idx}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            canvas[y : y + thumb_h, x : x + thumb_w] = ch_color

    # 6. Lưu file
    output_path = args.output or f"channels_layer_{args.layer}.jpg"
    cv2.imwrite(output_path, canvas)
    print(f"✅ Đã tạo xong lưới {C} channels: [ {output_path} ]")
    print(f"👉 Mở file '{output_path}' để ngắm toàn bộ {C} 'kính lọc' con của Layer {args.layer}!")


if __name__ == "__main__":
    main()
