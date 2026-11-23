import argparse
import os
import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize AI attention heatmaps (Eigen-CAM) for YOLOv8 (Supports multiple layers simultaneously)"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to input image for analysis (e.g. car.jpg)",
    )
    parser.add_argument(
        "-i", "--image-path",
        dest="image_flag",
        default=None,
        help="Path to input image (using flag)",
    )
    parser.add_argument(
        "-m", "--model",
        default="yolov8n.pt",
        help="YOLO model weights path (default: yolov8n.pt)",
    )
    parser.add_argument(
        "-l", "--layer", "--layers",
        dest="layers",
        type=int,
        nargs="+",
        default=[15],
        help="List of layer indices to inspect (e.g., -l 9 15 18 21; default: 15)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output image path (default: heatmap_<name>_layer_<L>.jpg)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO detection (default: 0.25)",
    )
    return parser.parse_args()


def compute_eigen_cam(feature_tensor, orig_w, orig_h):
    """Compute Eigen-CAM from feature map tensor (1, C, H, W)."""
    feat = feature_tensor
    if feat.ndim == 4:
        feat = feat.squeeze(0)
    C, H, W = feat.shape

    # Zero-center along spatial dimension
    X = feat.view(C, -1)
    X = X - X.mean(dim=1, keepdim=True)

    # SVD to find first principal component
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    first_eigenvector = U[:, 0]

    # Project and reshape into 2D heatmap
    projection = torch.matmul(first_eigenvector, X).view(H, W)
    heatmap = torch.relu(projection)
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_np = heatmap.detach().cpu().float().numpy()

    # Resize to original resolution and apply JET colormap
    heatmap_resized = cv2.resize(heatmap_np, (orig_w, orig_h))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    return heatmap_color


def get_model_layers(model):
    """
    Retrieve currently active layers in the YOLO pipeline.
    After Ultralytics executes an initial inference, computation moves to
    model.predictor.model (AutoBackend), so hooks must attach to active modules.
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
    Generate Eigen-CAM Heatmap for an image (path or np.ndarray).
    Returns: (cam_overlay_rgb, comparison_rgb, layer_name)
    """
    if isinstance(image_input, (str, os.PathLike)):
        frame = cv2.imread(str(image_input))
        if frame is None:
            raise ValueError(f"Could not read image from: {image_input}")
    else:
        frame = image_input.copy()

    orig_h, orig_w = frame.shape[:2]

    if model is None:
        model = YOLO(model_path)

    active_layers = get_model_layers(model)
    total_layers = len(active_layers)
    if layer_idx < 0 or layer_idx >= total_layers:
        raise ValueError(f"Layer {layer_idx} invalid (model has {total_layers} layers: 0..{total_layers-1})")

    target_mod = active_layers[layer_idx]
    layer_name = target_mod.__class__.__name__

    activations = {}
    def hook_fn(mod, inp, out):
        activations["feat"] = out[0] if isinstance(out, (list, tuple)) else out

    hooks = [target_mod.register_forward_hook(hook_fn)]
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
        raise RuntimeError(f"Could not capture Feature Map from Layer {layer_idx}")

    heatmap_bgr = compute_eigen_cam(feat, orig_w, orig_h)
    cam_overlay_bgr = cv2.addWeighted(frame, 0.6, heatmap_bgr, 0.4, 0)

    # Layer label in top corner
    label_text = f"Layer {layer_idx}: {layer_name}"
    cv2.rectangle(cam_overlay_bgr, (10, 10), (320, 50), (0, 0, 0), -1)
    cv2.putText(cam_overlay_bgr, label_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    comparison_bgr = np.hstack((boxes_plot, cam_overlay_bgr))

    cam_overlay_rgb = cv2.cvtColor(cam_overlay_bgr, cv2.COLOR_BGR2RGB)
    comparison_rgb = cv2.cvtColor(comparison_bgr, cv2.COLOR_BGR2RGB)

    return cam_overlay_rgb, comparison_rgb, layer_name


def main():
    args = parse_args()

    image_path = args.image or args.image_flag
    if not image_path:
        print("💡 Tip: Quick run: python view_heatmap.py <image_path> -l 9 15 18")
        image_path = input("👉 Enter image path to analyze: ").strip()

    if not image_path or not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at '{image_path}'!")
        sys.exit(1)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Error: Unable to read image '{image_path}'!")
        sys.exit(1)

    orig_h, orig_w = frame.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"📷 Loaded image: '{image_path}' ({orig_w}x{orig_h})")

    print(f"🧠 Loading model: '{args.model}'...")
    model = YOLO(args.model)
    total_layers = len(model.model.model)

    valid_layers = []
    for l in args.layers:
        if 0 <= l < total_layers:
            valid_layers.append(l)
        else:
            print(f"⚠️ Warning: Skipping layer {l}; model only has layers 0 to {total_layers-1}!")

    if not valid_layers:
        print("❌ Error: No valid layers selected!")
        sys.exit(1)

    print(f"🔍 Analyzing {len(valid_layers)} layer(s): {valid_layers}")

    activations = {}
    hooks = []
    for layer_idx in valid_layers:
        target_mod = model.model.model[layer_idx]
        def make_hook(idx):
            def hook(mod, inp, out):
                activations[idx] = out[0] if isinstance(out, (list, tuple)) else out
            return hook
        hooks.append(target_mod.register_forward_hook(make_hook(layer_idx)))

    results = model(frame, conf=args.conf, verbose=False)
    boxes_plot = results[0].plot()

    for h in hooks:
        h.remove()

    saved_files = []
    for layer_idx in valid_layers:
        target_mod = model.model.model[layer_idx]
        layer_name = target_mod.__class__.__name__

        feat = activations.get(layer_idx)
        if feat is None:
            print(f"⚠️ Could not capture activations from Layer {layer_idx}, skipping.")
            continue

        heatmap_color = compute_eigen_cam(feat, orig_w, orig_h)
        cam_overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

        label_text = f"Layer {layer_idx}: {layer_name}"
        cv2.rectangle(cam_overlay, (10, 10), (320, 50), (0, 0, 0), -1)
        cv2.putText(cam_overlay, label_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        comparison = np.hstack((boxes_plot, cam_overlay))

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

    print("\n" + "=" * 65)
    print("✅ SUCCESSFULLY GENERATED HEATMAPS:")
    for l_idx, l_name, f_path in saved_files:
        print(f"  • Layer {l_idx:2d} ({l_name:<8}): [ {f_path} ]")
    print("=" * 65)
    print("👉 Open the files above to inspect activations across layers!")


if __name__ == "__main__":
    main()
