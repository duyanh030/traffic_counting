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
        description="Dissect and display all individual feature channels of a YOLOv8 layer"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="candidates/previews/cand_20261119_095701_f000185_track_loss.jpg",
        help="Path to input image file",
    )
    parser.add_argument(
        "-l", "--layer",
        type=int,
        default=15,
        help="Layer index to dissect (default: 15 - P3 with 64 channels; options: 2, 4, 6, 9, 18, 21)",
    )
    parser.add_argument(
        "-m", "--model",
        default="yolov8n.pt",
        help="YOLO model weights path (default: yolov8n.pt)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output image path (default: channels_layer_<L>.jpg)",
    )
    parser.add_argument(
        "--cmap",
        choices=["gray", "jet", "inferno"],
        default="gray",
        help="Channel colormap: gray (grayscale), jet (blue-red heat), inferno (fire)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = args.image

    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at '{image_path}'!")
        sys.exit(1)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Error: Could not read image '{image_path}'!")
        sys.exit(1)

    model = YOLO(args.model)
    active_layers = get_model_layers(model)
    total_layers = len(active_layers)
    if args.layer < 0 or args.layer >= total_layers:
        print(f"❌ Error: Layer {args.layer} invalid (model has {total_layers} layers: 0 to {total_layers-1})")
        sys.exit(1)

    target_layer = active_layers[args.layer]
    layer_name = target_layer.__class__.__name__

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

    print(f"🧠 Dissecting Layer {args.layer} ({layer_name}) from image: {os.path.basename(image_path)}...")
    try:
        model(frame, verbose=False)
    finally:
        for h in hooks:
            h.remove()

    feat = activations.get("feat")
    if feat is None:
        print("❌ Error: Could not capture Feature Map from this layer!")
        sys.exit(1)

    if feat.ndim == 4:
        feat = feat.squeeze(0)  # (C, H, W)
    elif feat.ndim != 3:
        print(f"⚠️ Layer output tensor format not compatible with 2D channel grid: shape = {feat.shape}")
        sys.exit(1)

    C, H, W = feat.shape
    print(f"📦 Layer {args.layer} ({layer_name}) has {C} Channels (each channel size: {W}x{H})")

    feat_np = feat.detach().cpu().float().numpy()

    cols = int(math.ceil(math.sqrt(C)))
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

    thumb_w, thumb_h = 120, 80
    border = 2

    grid_h = rows * thumb_h + (rows + 1) * border
    grid_w = cols * thumb_w + (cols + 1) * border

    if args.cmap == "gray":
        canvas = np.zeros((grid_h, grid_w), dtype=np.uint8)
    else:
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

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
            cv2.putText(ch_resized, f"Ch.{idx}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,), 1)
            canvas[y : y + thumb_h, x : x + thumb_w] = ch_resized
        else:
            colormap_code = cv2.COLORMAP_JET if args.cmap == "jet" else cv2.COLORMAP_INFERNO
            ch_color = cv2.applyColorMap(ch_resized, colormap_code)
            cv2.putText(ch_color, f"Ch.{idx}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            canvas[y : y + thumb_h, x : x + thumb_w] = ch_color

    output_path = args.output or f"channels_layer_{args.layer}.jpg"
    cv2.imwrite(output_path, canvas)
    print(f"✅ Successfully created {C}-channel grid: [ {output_path} ]")
    print(f"👉 Open '{output_path}' to view all {C} sub-channel feature filters of Layer {args.layer}!")


if __name__ == "__main__":
    main()
