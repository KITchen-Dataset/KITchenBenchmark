"""Resume the YOLO-seg baseline under the same hard VRAM cap.

Usage: python train_resume.py RUN_DIR/weights/last.pt [CAP_GB]
"""
import sys

import torch

CAP_GB = float(sys.argv[2]) if len(sys.argv) > 2 else 11.0
total = torch.cuda.get_device_properties(0).total_memory / 2**30
torch.cuda.set_per_process_memory_fraction(min(1.0, CAP_GB / total), 0)

from ultralytics import YOLO  # noqa: E402

YOLO(sys.argv[1]).train(resume=True)
