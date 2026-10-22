#!/usr/bin/env bash
# Score the YOLO-seg baseline on val and test once training has finished.
# Usage: post_train.sh RUN_DIR BENCH IMAGES OUT_DIR
#   IMAGES: N=path,... image folder of each release split (as for yolo_predict.py)
set -euo pipefail
RUN=$1 BENCH=$2 IMAGES=$3 OUT=$4
HERE=$(cd "$(dirname "$0")" && pwd)
while pgrep -f 'train_resume.py|train_capped' >/dev/null; do sleep 300; done
W=$RUN/weights/best.pt
[ -f "$W" ] || { echo "missing $W" >&2; exit 1; }
mkdir -p "$OUT/cards"
for split in val test; do
  python "$HERE/yolo_predict.py" --weights "$W" --bench "$BENCH" --images "$IMAGES" --split "$split" --out "$OUT/yolo11m_seg"
  for task in det segm; do
    kitchenbench eval "$BENCH" "$task" "$OUT/yolo11m_seg_${task}_${split}.json" --split "$split" \
      --method "$HERE/yolo11m_seg_method.json" --out "$OUT/cards"
  done
done
echo "done $(date -u +%FT%TZ)" > "$OUT/DONE"
