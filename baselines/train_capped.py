"""Train the YOLO-seg baseline, optionally under a hard VRAM cap so it can share the GPU.

Usage: python train_capped.py DATA_YAML [PROJECT_DIR] [CAP_GB] [BATCH] [CACHE]
DATA_YAML is the data.yaml written by prep_yolo.py. The published baseline used
the defaults: no effective cap (48 GB card), batch 16, no image cache
(CACHE ram needs ~37 GB of RAM for the training images at 1024 px).
"""
import sys

import torch

DATA = sys.argv[1]
PROJECT = sys.argv[2] if len(sys.argv) > 2 else "runs"
CAP_GB = float(sys.argv[3]) if len(sys.argv) > 3 else 48.0
BATCH = int(sys.argv[4]) if len(sys.argv) > 4 else 16
CACHE = {"ram": "ram", "disk": "disk"}.get(sys.argv[5] if len(sys.argv) > 5 else "", False)
total = torch.cuda.get_device_properties(0).total_memory / 2**30
torch.cuda.set_per_process_memory_fraction(min(1.0, CAP_GB / total), 0)

from ultralytics import YOLO  # noqa: E402

YOLO("yolo11m-seg.pt").train(
    data=DATA, imgsz=1024, epochs=30, batch=BATCH, workers=16, seed=0,
    deterministic=True, patience=10, close_mosaic=5, cache=CACHE,
    project=PROJECT, name=f"yolo11m_seg_1024_b{BATCH}")
