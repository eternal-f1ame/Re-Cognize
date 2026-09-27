"""
Dataset for Memory Block Training

Panel-sequence aware dataset that supports:
- Loading character crops from YOLO annotations
- Sequence-based batching for working memory training
- Panel context preservation
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset, Sampler

from recognize.protocol_constants import CROP_PADDING
from PIL import Image
import numpy as np


class MangaCharacterDataset(Dataset):
    """
    Dataset for manga character Re-ID with panel sequence support.

    Features:
    - Loads character crops from YOLO format annotations
    - Preserves panel/page sequence information
    - Supports sequence-based batching for working memory

    Args:
        data_dir: Path to dataset directory
        split: "train" or "val"
        transform: Image transforms
        padding: Bbox padding ratio
        train_ratio: Train/val split ratio
        exclude_categories: Categories to exclude (e.g., "dbox,sbox")
        return_sequence_info: If True, return sequence metadata
        combine_subdirs: If True, combine all subdirectories (for multi-manga training)
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        padding: float = CROP_PADDING,
        train_ratio: float = 0.8,
        exclude_categories: str = "",
        return_sequence_info: bool = True,
        seed: int = 42,
        combine_subdirs: bool = False,
        class_remap: Optional[Dict[int, int]] = None,  # Use external class mapping
        manga_filter: Optional[List[str]] = None,  # Only use these manga subdirs
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.padding = padding
        self.return_sequence_info = return_sequence_info
        self.combine_subdirs = combine_subdirs
        self._external_class_remap = class_remap  # Store external remap
        self.manga_filter = set(manga_filter) if manga_filter else None

        # Parse exclusions
        self.exclude_set = set()
        if exclude_categories:
            self.exclude_set = {x.strip().lower() for x in exclude_categories.split(",")}

        # Load category mapping
        self.categories = self._load_categories()

        # Load samples (from single dir or combined subdirs)
        if self.combine_subdirs:
            self.samples = self._load_samples_combined()
        else:
            self.samples = self._load_samples()

        # Split train/val
        self._split_data(train_ratio, seed)

        # Build character-to-samples mapping
        self.char_to_samples = self._build_char_mapping()

        # Build sequence mapping (page/panel order)
        self.sequence_mapping = self._build_sequence_mapping()

        print(f"Loaded {len(self.samples)} samples for {split}")
        print(f"  Characters: {len(self.char_to_samples)}")
        print(f"  Sequences: {len(self.sequence_mapping)}")

    def _load_categories(self) -> Dict[int, str]:
        """Load category ID to name mapping."""
        categories = {}

        # Try category_mapping.json
        cat_file = self.data_dir / "category_mapping.json"
        if cat_file.exists():
            with open(cat_file) as f:
                data = json.load(f)
                if isinstance(data, dict):
                    categories = {int(k): v for k, v in data.items()}
                elif isinstance(data, list):
                    categories = {i: name for i, name in enumerate(data)}

        # Try classes.txt (YOLO format)
        classes_file = self.data_dir / "classes.txt"
        if not categories and classes_file.exists():
            with open(classes_file) as f:
                for i, line in enumerate(f):
                    categories[i] = line.strip()

        return categories

    def _load_samples(self) -> List[Dict]:
        """Load all samples from YOLO annotations."""
        samples = []

        images_dir = self.data_dir / "images"
        labels_dir = self.data_dir / "labels"

        if not images_dir.exists():
            images_dir = self.data_dir

        if not labels_dir.exists():
            # Try "annotations" directory (common alternative)
            labels_dir = self.data_dir / "annotations"

        if not labels_dir.exists():
            labels_dir = self.data_dir

        # Find all label files and sort them naturally to preserve chronological page order
        import re
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]
            
        label_files = sorted(list(labels_dir.glob("*.txt")), key=lambda p: natural_sort_key(p.name))

        for label_file in label_files:
            # Find corresponding image
            stem = label_file.stem
            image_path = None

            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                continue

            # Parse annotations
            with open(label_file) as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Get category name
                    cat_name = self.categories.get(class_id, f"class_{class_id}")

                    # Skip excluded categories
                    if cat_name.lower() in self.exclude_set:
                        continue

                    samples.append({
                        "image_path": str(image_path),
                        "class_id": class_id,
                        "category": cat_name,
                        "bbox": [x_center, y_center, width, height],
                        "page_name": stem,
                        "annotation_idx": line_idx,
                    })

        return samples

    def _load_samples_combined(self) -> List[Dict]:
        """Load samples from all subdirectories (multi-manga training)."""
        all_samples = []
        global_class_offset = 0

        # Find all manga subdirectories
        subdirs = sorted([d for d in self.data_dir.iterdir()
                         if d.is_dir() and (d / "images").exists()])

        # Filter to specific manga if requested
        if self.manga_filter:
            subdirs = [d for d in subdirs if d.name in self.manga_filter]

        if not subdirs:
            print(f"No subdirectories found in {self.data_dir}, falling back to single dir")
            return self._load_samples()

        print(f"Combining {len(subdirs)} manga series...")

        for subdir in subdirs:
            manga_name = subdir.name
            images_dir = subdir / "images"
            labels_dir = subdir / "annotations"

            if not labels_dir.exists():
                labels_dir = subdir / "labels"
            if not labels_dir.exists():
                continue

            # Load category mapping for this manga
            cat_file = subdir / "category_mapping.json"
            manga_categories = {}
            if cat_file.exists():
                with open(cat_file) as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        manga_categories = {int(k): v for k, v in data.items()}
                    elif isinstance(data, list):
                        manga_categories = {i: name for i, name in enumerate(data)}

            # Update global categories with offset
            for local_id, name in manga_categories.items():
                global_id = local_id + global_class_offset
                self.categories[global_id] = f"{manga_name}:{name}"

            # Load samples, sorted to preserve page order
            label_files = sorted(list(labels_dir.glob("*.txt")))
            manga_samples = 0

            for label_file in label_files:
                stem = label_file.stem
                image_path = None

                for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                    candidate = images_dir / f"{stem}{ext}"
                    if candidate.exists():
                        image_path = candidate
                        break

                if image_path is None:
                    continue

                with open(label_file) as f:
                    for line_idx, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue

                        local_class_id = int(parts[0])
                        global_class_id = local_class_id + global_class_offset

                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        cat_name = manga_categories.get(local_class_id, f"class_{local_class_id}")

                        if cat_name.lower() in self.exclude_set:
                            continue

                        all_samples.append({
                            "image_path": str(image_path),
                            "class_id": global_class_id,
                            "category": f"{manga_name}:{cat_name}",
                            "bbox": [x_center, y_center, width, height],
                            "page_name": f"{manga_name}/{stem}",
                            "annotation_idx": line_idx,
                            "manga": manga_name,
                        })
                        manga_samples += 1

            # Update offset for next manga
            if manga_categories:
                max_local_id = max(manga_categories.keys())
                global_class_offset += max_local_id + 1

            print(f"  {manga_name}: {manga_samples} samples, {len(manga_categories)} characters")

        print(f"Total: {len(all_samples)} samples, {global_class_offset} characters")
        return all_samples

    def _split_data(self, train_ratio: float, seed: int) -> None:
        """Split data into train/val with stratification.

        When train_ratio >= 1.0, all samples are used without splitting.
        This is used with manga_filter for manga-level train/val separation.
        """
        random.seed(seed)

        # Group by class
        class_samples = defaultdict(list)
        for i, sample in enumerate(self.samples):
            class_samples[sample["class_id"]].append(i)

        if train_ratio < 1.0:
            # Standard sample-level split with stratification
            train_indices = []
            val_indices = []

            for class_id, indices in class_samples.items():
                random.shuffle(indices)
                n_train = max(1, int(len(indices) * train_ratio))

                train_indices.extend(indices[:n_train])
                val_indices.extend(indices[n_train:])

            if self.split == "train":
                self.samples = [self.samples[i] for i in train_indices]
            else:
                self.samples = [self.samples[i] for i in val_indices]

        # Use external class_remap if provided, otherwise create new one
        if self._external_class_remap is not None:
            self.class_remap = self._external_class_remap
            self.num_classes = max(self.class_remap.values()) + 1
        else:
            # Remap class IDs to contiguous range based on ALL classes (not just this split)
            # This ensures train and val use consistent labels
            all_class_ids = sorted(class_samples.keys())
            self.class_remap = {old: new for new, old in enumerate(all_class_ids)}
            self.num_classes = len(all_class_ids)

        # Apply remapping - skip samples whose class_id is not in remap
        valid_samples = []
        for sample in self.samples:
            if sample["class_id"] in self.class_remap:
                sample["label"] = self.class_remap[sample["class_id"]]
                valid_samples.append(sample)
        self.samples = valid_samples

    def _build_char_mapping(self) -> Dict[int, List[int]]:
        """Build mapping from character label to sample indices, sorted chronologically."""
        mapping = defaultdict(list)
        for i, sample in enumerate(self.samples):
            mapping[sample["label"]].append(i)
            
        import re
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

        # Ensure the lists are sorted naturally by page and annotation_idx
        for label in mapping:
            mapping[label].sort(
                key=lambda i: (natural_sort_key(self.samples[i]["page_name"]), self.samples[i]["annotation_idx"])
            )
            
        return dict(mapping)

    def _build_sequence_mapping(self) -> Dict[str, List[int]]:
        """Build mapping from page/sequence to sample indices."""
        mapping = defaultdict(list)
        for i, sample in enumerate(self.samples):
            mapping[sample["page_name"]].append(i)

        # Sort by annotation index within each page
        for page_name in mapping:
            mapping[page_name].sort(
                key=lambda i: self.samples[i]["annotation_idx"]
            )

        return dict(mapping)

    def _crop_bbox(
        self,
        image: Image.Image,
        bbox: List[float],
    ) -> Image.Image:
        """Crop bounding box with padding."""
        W, H = image.size
        x_center, y_center, w, h = bbox

        # Convert to absolute coordinates
        x_center *= W
        y_center *= H
        w *= W
        h *= H

        # Add padding
        pad_w = w * self.padding
        pad_h = h * self.padding

        x1 = max(0, int(x_center - w / 2 - pad_w))
        y1 = max(0, int(y_center - h / 2 - pad_h))
        x2 = min(W, int(x_center + w / 2 + pad_w))
        y2 = min(H, int(y_center + h / 2 + pad_h))

        return image.crop((x1, y1, x2, y2))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load and crop image
        image = Image.open(sample["image_path"]).convert("RGB")
        crop = self._crop_bbox(image, sample["bbox"])

        # Apply transforms
        if self.transform is not None:
            crop = self.transform(crop)

        result = {
            "image": crop,
            "label": sample["label"],
        }

        if self.return_sequence_info:
            result.update({
                "page_name": sample["page_name"],
                "annotation_idx": sample["annotation_idx"],
                "category": sample["category"],
            })

        return result


class PKSampler(Sampler):
    """PK sampler: P identities x K distinct crops per batch.

    Identities with fewer than K crops are excluded from PK batches (no padding with
    replacement, so support and query never share a crop). The K crops of an identity
    are a random contiguous window of its chronological crop list (a tracklet), which
    is what working memory sees at test time. Only full batches are yielded.
    """

    def __init__(self, dataset: MangaCharacterDataset, p: int = 8, k: int = 4) -> None:
        self.dataset = dataset
        self.k = k
        self.char_to_samples = dataset.char_to_samples
        all_labels = list(self.char_to_samples.keys())
        self.labels = [l for l in all_labels if len(self.char_to_samples[l]) >= k]
        self.n_excluded = len(all_labels) - len(self.labels)
        self.p = min(p, len(self.labels)) if self.labels else 1
        if self.n_excluded:
            print(f"PKSampler: {self.n_excluded} identities with < {k} crops excluded from PK batches")

    def __iter__(self):
        if not self.labels:
            return iter([])
        labels = self.labels[:]
        random.shuffle(labels)
        n_full = (len(labels) // self.p) * self.p
        for i in range(0, n_full, self.p):
            for label in labels[i:i + self.p]:
                samples = self.char_to_samples[label]
                start = random.randint(0, len(samples) - self.k)
                yield from samples[start:start + self.k]

    def __len__(self) -> int:
        if not self.labels:
            return 0
        return (len(self.labels) // self.p) * self.p * self.k


class SequenceSampler(Sampler):
    """
    Sequence-aware sampler for working memory training.

    Samples characters from the same page/sequence together
    to provide context for working memory.

    Args:
        dataset: MangaCharacterDataset
        sequence_length: Number of samples per sequence
        shuffle_sequences: Whether to shuffle sequence order
    """

    def __init__(
        self,
        dataset: MangaCharacterDataset,
        sequence_length: int = 8,
        shuffle_sequences: bool = True,
    ) -> None:
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.shuffle_sequences = shuffle_sequences

        self.sequence_mapping = dataset.sequence_mapping
        self.sequences = list(self.sequence_mapping.keys())

    def __iter__(self):
        if self.shuffle_sequences:
            random.shuffle(self.sequences)

        indices = []

        for seq_name in self.sequences:
            seq_indices = self.sequence_mapping[seq_name]

            # Pad or truncate to sequence_length
            if len(seq_indices) >= self.sequence_length:
                selected = seq_indices[:self.sequence_length]
            else:
                # Pad with random samples from same sequence
                selected = seq_indices + random.choices(
                    seq_indices,
                    k=self.sequence_length - len(seq_indices),
                )

            indices.extend(selected)

        return iter(indices)

    def __len__(self) -> int:
        return len(self.sequences) * self.sequence_length


def get_transforms(
    height: int = 256,
    width: int = 128,
    is_train: bool = True,
    normalize_type: str = "imagenet",
) -> Callable:
    """
    Get image transforms for training/validation.

    Args:
        height: Target image height
        width: Target image width
        is_train: Whether training transforms
        normalize_type: "imagenet" or "clip"

    Returns:
        Transform function
    """
    from torchvision import transforms

    if normalize_type == "clip":
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:  # imagenet
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.Resize((height, width)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
            transforms.RandomGrayscale(p=0.2),  # Important for B&W manga
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


if __name__ == "__main__":
    # Test dataset
    import sys
    from pathlib import Path

    # Find a dataset directory
    project_root = Path(__file__).parent.parent.parent.parent
    data_dirs = list((project_root / "Datasets" / "popcharacters").glob("*"))

    if not data_dirs:
        print("No dataset found")
        sys.exit(1)

    data_dir = data_dirs[0]
    print(f"Testing with: {data_dir}")

    # Create dataset
    transform = get_transforms(is_train=True)
    dataset = MangaCharacterDataset(
        data_dir=data_dir,
        split="train",
        transform=transform,
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")

    # Test sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Label: {sample['label']}")

    # Test PK sampler
    sampler = PKSampler(dataset, p=4, k=4)
    print(f"\nPK Sampler length: {len(sampler)}")

    # Test sequence sampler
    seq_sampler = SequenceSampler(dataset, sequence_length=8)
    print(f"Sequence Sampler length: {len(seq_sampler)}")
