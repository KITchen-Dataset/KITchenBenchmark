"""Turn a frozen KITchen benchmark into an Ultralytics YOLO-seg dataset.

    python prep_yolo.py --bench kitchen_bench_v1.0 --release /root/kit_labels \
        --images 1=/data/KITchen_rgb_1/images,2=...,3=...,4=... --out /root/kb_yolo --train-stride 3

Training frames come from splits/train.txt (every --train-stride-th frame,
since neighbouring frames are near-duplicates), validation frames from
splits/val_eval.txt. Labels are the release polygons (seg_labels/, already in
YOLO format); rows marked in ignore.json and class 0 ("other") are left out.
YOLO class id == KITchen class id, so predictions need no remapping. Images are
symlinked, never copied.
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bench", type=Path, required=True)
    p.add_argument("--release", type=Path, required=True, help="folder with KITchen_labels_<N>/")
    p.add_argument("--images", required=True, help="N=path,... image folder of each release split")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-stride", type=int, default=3)
    a = p.parse_args()
    images = {int(k): Path(v) for k, v in (x.split("=", 1) for x in a.images.split(","))}
    labs = {n: a.release / f"KITchen_labels_{n}" for n in images}
    folder_of = {q.stem: n for n, d in labs.items() for q in (d / "labels").glob("*.txt")}
    ignores = {n: json.loads((d / "ignore.json").read_text()) for n, d in labs.items()}
    classes = (labs[min(labs)] / "classes.txt").read_text().splitlines()
    subsets = {"train": (a.bench / "splits" / "train.txt").read_text().split()[:: a.train_stride],
               "val": (a.bench / "splits" / "val_eval.txt").read_text().split()}
    for sub, stems in subsets.items():
        (a.out / sub / "images").mkdir(parents=True, exist_ok=True)
        (a.out / sub / "labels").mkdir(parents=True, exist_ok=True)
        missing = 0
        for s in stems:
            n = folder_of[s]
            img = images[n] / f"{s}.png"
            if not img.exists():
                missing += 1
                continue
            link = a.out / sub / "images" / f"{s}.png"
            if not link.exists():
                link.symlink_to(img)
            boxes = (labs[n] / "labels" / f"{s}.txt").read_text().splitlines()
            sp = labs[n] / "seg_labels" / f"{s}.txt"
            polys = sp.read_text().splitlines() if sp.exists() else []
            ign = set(ignores[n].get(s, []))
            out = []
            for i, line in enumerate(boxes):
                b = line.split()
                if len(b) < 5 or int(b[0]) == 0 or i in ign:
                    continue
                q = polys[i].split() if i < len(polys) else []
                if len(q) >= 7 and q[0] == b[0]:
                    out.append(" ".join(q))
                else:  # no polygon: use the box outline
                    c, cx, cy, w, h = b[0], *map(float, b[1:5])
                    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                    out.append(f"{c} {x0:.6f} {y0:.6f} {x1:.6f} {y0:.6f} {x1:.6f} {y1:.6f} {x0:.6f} {y1:.6f}")
            (a.out / sub / "labels" / f"{s}.txt").write_text("\n".join(out) + "\n")
        print(sub, len(stems) - missing, "frames", f"({missing} images missing)" if missing else "")
    names = "\n".join(f"  {i}: {json.dumps(c)}" for i, c in enumerate(classes))
    (a.out / "data.yaml").write_text(f"path: {a.out}\ntrain: train/images\nval: val/images\nnames:\n{names}\n")


if __name__ == "__main__":
    main()
