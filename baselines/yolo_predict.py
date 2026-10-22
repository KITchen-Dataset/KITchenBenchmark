"""Run a trained YOLO-seg model on the benchmark evaluation frames and write
COCO result files that `kitchenbench eval` scores.

    python yolo_predict.py --weights best.pt --bench kitchen_bench_v1.0 \
        --images 1=...,2=...,3=...,4=... --split test --out results/yolo_seg
writes results/yolo_seg_det_test.json and results/yolo_seg_segm_test.json
"""
import argparse
import json
import time
from pathlib import Path

from kitchenbench import rle


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--bench", type=Path, required=True)
    p.add_argument("--images", required=True, help="N=path,... image folder of each release split")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.01)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    from ultralytics import YOLO
    images = {int(k): Path(v) for k, v in (x.split("=", 1) for x in a.images.split(","))}
    coco = json.loads((a.bench / "coco" / f"{a.split}.json").read_text())
    paths = [images[int(im["file_name"].split("_")[2].split("/")[0])] / Path(im["file_name"]).name
             for im in coco["images"]]
    model = YOLO(a.weights)
    det, segm, t0 = [], [], time.time()
    for k in range(0, len(paths), a.batch):
        chunk = [str(x) for x in paths[k:k + a.batch]]
        for path, r in zip(chunk, model.predict(chunk, imgsz=a.imgsz, conf=a.conf, max_det=a.max_det,
                                                 retina_masks=True, verbose=False)):
            im_id = int(Path(path).stem)
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            score = r.boxes.conf.cpu().numpy()
            masks = r.masks.data.cpu().numpy() > 0.5 if r.masks is not None else None
            for j in range(len(cls)):
                x0, y0, x1, y1 = xyxy[j].tolist()
                base = {"image_id": im_id, "category_id": int(cls[j]), "score": round(float(score[j]), 5)}
                det.append({**base, "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)]})
                if masks is not None:
                    segm.append({**base, "segmentation": rle.encode(masks[j])})
        if k % (50 * a.batch) == 0:
            print(f"{k}/{len(paths)} frames", flush=True)
    per_im = (time.time() - t0) / max(1, len(paths))
    for task, res in (("det", det), ("segm", segm)):
        Path(f"{a.out}_{task}_{a.split}.json").write_text(json.dumps(res, separators=(",", ":")))
    print(f"{len(det)} detections, {per_im:.3f} s/image including mask encoding")


if __name__ == "__main__":
    main()
