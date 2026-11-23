import argparse
import os
import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trực quan hóa não bộ AI (Eigen-CAM) cho YOLOv8 (Hỗ trợ chạy nhiều Layer cùng lúc)"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Đường dẫn đến bức ảnh cần phân tích (ví dụ: car.jpg)",
    )
    parser.add_argument(
        "-i", "--image-path",
        dest="image_flag",
        default=None,
        help="Đường dẫn ảnh (dùng cờ -i nếu thích)",
    )
    parser.add_argument(
        "-m", "--model",
        default="yolov8n.pt",
        help="File trọng số YOLO (mặc định: yolov8n.pt)",
    )
    parser.add_argument(
        "-l", "--layer", "--layers",
        dest="layers",
        type=int,
        nargs="+",
        default=[15],
        help="Danh sách các Layer cần soi (ví dụ: -l 9 15 18 21; mặc định: 15)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Đường dẫn file ảnh đầu ra (mặc định: heatmap_<tên_ảnh>_layer_<số_layer>.jpg)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold cho YOLO detection (mặc định: 0.25)",
    )
    return parser.parse_args()


def compute_eigen_cam(feature_tensor, orig_w, orig_h):
    """Tính Eigen-CAM từ feature map tensor (1, C, H, W)."""
    feat = feature_tensor
    if feat.ndim == 4:
        feat = feat.squeeze(0)
    C, H, W = feat.shape

    # Chuẩn hóa tâm 0 trên chiều không gian
    X = feat.view(C, -1)
    X = X - X.mean(dim=1, keepdim=True)

    # SVD tìm Principal Component đầu tiên
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    first_eigenvector = U[:, 0]

    # Chiếu và chuyển đổi thành Heatmap 2D
    projection = torch.matmul(first_eigenvector, X).view(H, W)
    heatmap = torch.relu(projection)
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_np = heatmap.detach().cpu().float().numpy()

    # Phóng to về kích thước ảnh gốc và tô màu JET
    heatmap_resized = cv2.resize(heatmap_np, (orig_w, orig_h))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    return heatmap_color


def get_model_layers(model):
    """
    Lấy danh sách các layers thực sự hoạt động trong YOLO.
    Khi Ultralytics đã chạy inference, nó chuyển luồng tính toán vào
    model.predictor.model (AutoBackend), do đó cần hook đúng vào đối tượng đang chạy.
    """
    if hasattr(model, "predictor") and model.predictor is not None:
        p_model = getattr(model.predictor, "model", None)
        if p_model is not None:
            if hasattr(p_model, "model") and hasattr(p_model.model, "model"):
                return p_model.model.model
            elif hasattr(p_model, "model"):
                return p_model.model
    return model.model.model


def generate_heatmap_for_image(image_input, layer_idx=15, model=None, model_path="yolov8n.pt", conf=0.25):
    """
    Sinh Heatmap Eigen-CAM cho một ảnh (đường dẫn hoặc np.ndarray).
    Trả về: (cam_overlay_rgb, comparison_rgb, layer_name)
    """
    if isinstance(image_input, (str, os.PathLike)):
        frame = cv2.imread(str(image_input))
        if frame is None:
            raise ValueError(f"Không thể đọc ảnh từ: {image_input}")
    else:
        frame = image_input.copy()

    orig_h, orig_w = frame.shape[:2]

    if model is None:
        model = YOLO(model_path)

    # Đảm bảo lấy đúng layer đang active trong pipeline
    active_layers = get_model_layers(model)
    total_layers = len(active_layers)
    if layer_idx < 0 or layer_idx >= total_layers:
        raise ValueError(f"Layer {layer_idx} không hợp lệ (mô hình có {total_layers} layers: 0..{total_layers-1})")

    target_mod = active_layers[layer_idx]
    layer_name = target_mod.__class__.__name__

    activations = {}
    def hook_fn(mod, inp, out):
        activations["feat"] = out[0] if isinstance(out, (list, tuple)) else out

    # Đăng ký hook trên cả active_layers và model.model.model để đảm bảo luôn bắt được
    hooks = []
    hooks.append(target_mod.register_forward_hook(hook_fn))
    if hasattr(model, "model") and hasattr(model.model, "model"):
        base_mod = model.model.model[layer_idx]
        if base_mod is not target_mod:
            hooks.append(base_mod.register_forward_hook(hook_fn))

    # Inference
    try:
        results = model(frame, conf=conf, verbose=False)
        boxes_plot = results[0].plot()
    finally:
        for h in hooks:
            h.remove()

    feat = activations.get("feat")
    if feat is None:
        raise RuntimeError(f"Không bắt được Feature Map từ Layer {layer_idx}")

    heatmap_bgr = compute_eigen_cam(feat, orig_w, orig_h)
    cam_overlay_bgr = cv2.addWeighted(frame, 0.6, heatmap_bgr, 0.4, 0)

    # Nhãn tên Layer lên góc trên của ảnh Heatmap
    label_text = f"Layer {layer_idx}: {layer_name}"
    cv2.rectangle(cam_overlay_bgr, (10, 10), (320, 50), (0, 0, 0), -1)
    cv2.putText(cam_overlay_bgr, label_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    comparison_bgr = np.hstack((boxes_plot, cam_overlay_bgr))

    cam_overlay_rgb = cv2.cvtColor(cam_overlay_bgr, cv2.COLOR_BGR2RGB)
    comparison_rgb = cv2.cvtColor(comparison_bgr, cv2.COLOR_BGR2RGB)

    return cam_overlay_rgb, comparison_rgb, layer_name


def main():
    args = parse_args()

    # 1. Xác định đường dẫn ảnh
    image_path = args.image or args.image_flag
    if not image_path:
        print("💡 Gợi ý: Bạn có thể chạy nhanh: python view_heatmap.py <đường_dẫn_ảnh> -l 9 15 18")
        image_path = input("👉 Nhập đường dẫn bức ảnh cần phân tích: ").strip()

    if not image_path or not os.path.exists(image_path):
        print(f"❌ Lỗi: Không tìm thấy file ảnh tại '{image_path}'!")
        sys.exit(1)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Lỗi: Không thể đọc được file ảnh '{image_path}'!")
        sys.exit(1)

    orig_h, orig_w = frame.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"📷 Đã tải ảnh: '{image_path}' ({orig_w}x{orig_h})")

    # 2. Khởi tạo mô hình YOLO
    print(f"🧠 Đang nạp mô hình: '{args.model}'...")
    model = YOLO(args.model)
    total_layers = len(model.model.model)

    # Kiểm tra các layer hợp lệ
    valid_layers = []
    for l in args.layers:
        if 0 <= l < total_layers:
            valid_layers.append(l)
        else:
            print(f"⚠️ Cảnh báo: Bỏ qua Layer {l} vì mô hình chỉ có từ Layer 0 đến {total_layers-1}!")

    if not valid_layers:
        print("❌ Lỗi: Không có layer nào hợp lệ để phân tích!")
        sys.exit(1)

    print(f"🔍 Sẽ phân tích {len(valid_layers)} layer: {valid_layers}")

    # 3. Đăng ký Hook cho tất cả các layer được chọn
    activations = {}
    hooks = []
    for layer_idx in valid_layers:
        target_mod = model.model.model[layer_idx]
        def make_hook(idx):
            def hook(mod, inp, out):
                activations[idx] = out[0] if isinstance(out, (list, tuple)) else out
            return hook
        hooks.append(target_mod.register_forward_hook(make_hook(layer_idx)))

    # 4. Chạy dự đoán YOLO 1 lần duy nhất (Inference)
    results = model(frame, conf=args.conf, verbose=False)
    boxes_plot = results[0].plot()

    # Tháo hooks sau khi inference xong
    for h in hooks:
        h.remove()

    # 5. Xử lý xuất ảnh cho từng layer
    saved_files = []
    for layer_idx in valid_layers:
        target_mod = model.model.model[layer_idx]
        layer_name = target_mod.__class__.__name__

        feat = activations.get(layer_idx)
        if feat is None:
            print(f"⚠️ Không lấy được kích hoạt từ Layer {layer_idx}, bỏ qua.")
            continue

        heatmap_color = compute_eigen_cam(feat, orig_w, orig_h)
        cam_overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

        # Ghi nhãn tên Layer lên góc trên của ảnh Heatmap
        label_text = f"Layer {layer_idx}: {layer_name}"
        cv2.rectangle(cam_overlay, (10, 10), (320, 50), (0, 0, 0), -1)
        cv2.putText(cam_overlay, label_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Ghép so sánh song song: [Bounding Box] | [Heatmap Layer X]
        comparison = np.hstack((boxes_plot, cam_overlay))

        # Định dạng tên file đầu ra có chứa số layer:
        # Ví dụ: heatmap_<tên_ảnh>_layer_15.jpg
        if args.output and len(valid_layers) == 1:
            out_file = args.output
        elif args.output and len(valid_layers) > 1:
            out_dir = os.path.dirname(args.output) or "."
            out_stem = os.path.splitext(os.path.basename(args.output))[0]
            out_file = os.path.join(out_dir, f"{out_stem}_layer_{layer_idx}.jpg")
        else:
            out_file = f"heatmap_{base_name}_layer_{layer_idx}.jpg"

        cv2.imwrite(out_file, comparison)
        saved_files.append((layer_idx, layer_name, out_file))

    # In kết quả tổng kết
    print("\n" + "=" * 65)
    print("✅ ĐÃ XUẤT THÀNH CÔNG TẤT CẢ CÁC HEATMAP:")
    for l_idx, l_name, f_path in saved_files:
        print(f"  • Layer {l_idx:2d} ({l_name:<8}): [ {f_path} ]")
    print("=" * 65)
    print("👉 Mở các file trên để so sánh sự khác biệt góc nhìn giữa các tầng!")


if __name__ == "__main__":
    main()
