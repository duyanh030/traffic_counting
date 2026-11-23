import os
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

import config


def get_scenario_tag(img_path: Path, label_path: Optional[Path]) -> str:
    """
    Determine scenario/use-case tag for stratified sampling.
    Derives from candidate filename (track_loss, low_confidence)
    or from annotated YOLO classes inside the .txt label.
    """
    name_lower = img_path.name.lower()
    if "track_loss" in name_lower:
        return "track_loss"
    elif "low_confidence" in name_lower or "low_conf" in name_lower:
        return "low_confidence"

    if label_path and label_path.is_file():
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                classes = {line.strip().split()[0] for line in f if line.strip()}
            if not classes:
                return "empty_background"
            if len(classes) == 1:
                class_map = {"0": "car", "1": "motorcycle", "2": "bus", "3": "truck"}
                return f"{class_map.get(list(classes)[0], 'class_' + list(classes)[0])}_dominant"
            return "mixed_traffic"
        except Exception:
            pass

    return "general_traffic"


def stratified_split_by_scenario(
    pairs: List[Tuple[Path, Optional[Path]]],
    val_ratio: float = 0.2,
) -> Tuple[List[Tuple[Path, Optional[Path]]], List[Tuple[Path, Optional[Path]]]]:
    """
    Perform stratified sampling to ensure all scenarios/classes
    are proportionally represented across both train and validation splits.
    """
    total = len(pairs)
    if total <= 3:
        # Insufficient samples to split; retain all in training
        return pairs, []

    # Group by scenario
    scenario_buckets = defaultdict(list)
    for img, lbl in pairs:
        tag = get_scenario_tag(img, lbl)
        scenario_buckets[tag].append((img, lbl))

    final_train = []
    final_val = []

    random.seed(42)
    print(f"\n📊 Stratified Dataset Splitting ({len(scenario_buckets)} scenarios):")
    for scenario, items in sorted(scenario_buckets.items()):
        random.shuffle(items)
        n_items = len(items)

        if n_items > 3:
            val_count = max(1, int(round(n_items * val_ratio)))
        elif n_items >= 2 and total > 5:
            val_count = 1
        else:
            val_count = 0

        val_items = items[:val_count]
        train_items = items[val_count:]

        final_val.extend(val_items)
        final_train.extend(train_items)
        print(f"   • Scenario '{scenario}': {len(train_items)} train | {len(val_items)} val (total: {n_items})")

    return final_train, final_val


class DatasetManager:
    """
    Manages dataset ingestion from CVAT exports, train/val splitting,
    and YAML configuration generation for YOLO training.
    """

    def __init__(self, dataset_root: Path = None):
        self.root = Path(dataset_root or (config.BASE_DIR / "dataset"))
        self.train_img_dir = self.root / "train" / "images"
        self.train_lbl_dir = self.root / "train" / "labels"
        self.val_img_dir = self.root / "val" / "images"
        self.val_lbl_dir = self.root / "val" / "labels"
        self.imports_dir = self.root / "imports"

        # Create standard directory structure
        for d in [
            self.train_img_dir,
            self.train_lbl_dir,
            self.val_img_dir,
            self.val_lbl_dir,
            self.imports_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def extract_and_ingest_zip(self, zip_path: Path, val_ratio: float = 0.2) -> int:
        """
        Extract CVAT YOLO 1.1 exported zip and organize into train/val splits.
        """
        if not zip_path.is_file():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        temp_extract = self.root / "_temp_extract"
        if temp_extract.exists():
            shutil.rmtree(temp_extract)
        temp_extract.mkdir(parents=True, exist_ok=True)

        print(f"📦 Extracting CVAT export: {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_extract)

        # Discover all image files and their matching .txt label files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        pairs: List[Tuple[Path, Path]] = []

        for p in temp_extract.rglob("*"):
            if p.is_file() and p.suffix.lower() in image_extensions:
                matching_txt = p.with_suffix(".txt")
                if matching_txt.is_file():
                    pairs.append((p, matching_txt))
                else:
                    # Search in a sibling labels directory
                    alt_txt = p.parent.parent / "labels" / f"{p.stem}.txt"
                    if alt_txt.is_file():
                        pairs.append((p, alt_txt))
                    else:
                        # Empty label file for background/negative sample
                        pairs.append((p, None))

        if not pairs:
            shutil.rmtree(temp_extract, ignore_errors=True)
            raise ValueError(f"No valid image-label pairs found inside {zip_path.name}")

        print(f"🔍 Discovered {len(pairs)} labeled images inside the zip archive.")
        train_pairs, val_pairs = stratified_split_by_scenario(pairs, val_ratio=val_ratio)

        # Ingest train pairs
        for img, lbl in train_pairs:
            dest_img = self.train_img_dir / img.name
            shutil.copy2(img, dest_img)
            dest_lbl = self.train_lbl_dir / f"{img.stem}.txt"
            if lbl and lbl.is_file():
                shutil.copy2(lbl, dest_lbl)
            else:
                dest_lbl.touch()

        # Ingest val pairs
        for img, lbl in val_pairs:
            dest_img = self.val_img_dir / img.name
            shutil.copy2(img, dest_img)
            dest_lbl = self.val_lbl_dir / f"{img.stem}.txt"
            if lbl and lbl.is_file():
                shutil.copy2(lbl, dest_lbl)
            else:
                dest_lbl.touch()

        # Cleanup temp directory
        shutil.rmtree(temp_extract, ignore_errors=True)

        print(
            f"✅ Ingested successfully: {len(train_pairs)} images to train, {len(val_pairs)} images to val."
        )
        return len(pairs)

    def generate_yaml(self) -> Path:
        """
        Generate or update traffic_data.yaml pointing to the dataset splits.
        """
        yaml_path = self.root / "traffic_data.yaml"
        dataset_config = {
            "path": str(self.root.resolve()),
            "train": "train/images",
            "val": "val/images" if any(self.val_img_dir.glob("*.*")) else "train/images",
            "names": {
                0: "car",
                1: "motorcycle",
                2: "bus",
                3: "truck",
            },
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(dataset_config, f, sort_keys=False, indent=2)

        print(f"📄 Generated dataset config: {yaml_path}")
        return yaml_path

    def get_stats(self) -> Dict[str, int]:
        """Return counts of images in train and val sets."""
        img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        train_count = sum(1 for p in self.train_img_dir.glob("*.*") if p.suffix.lower() in img_exts)
        val_count = sum(1 for p in self.val_img_dir.glob("*.*") if p.suffix.lower() in img_exts)
        return {"train_images": train_count, "val_images": val_count, "total": train_count + val_count}
