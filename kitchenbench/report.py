"""Result cards: one self-describing JSON per evaluated submission.

A card holds the headline numbers, per-class breakdowns, the method facts a
reader needs to compare fairly (input modality, training data), and the
provenance that makes the number reproducible (benchmark version, hash of the
frozen ground truth, hash of the prediction file, evaluator version). The
leaderboard is built from cards only.
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from . import __version__

HEADLINE = {
    "det": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR300"],
    "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR300"],
    "pose": ["AP_6D", "AP_MSSD", "AP_MSPD", "AR_6D", "AR_MSSD", "AR_MSPD", "time_per_image_s"],
}
PRIMARY = {"det": "AP", "segm": "AP", "pose": "AP_6D"}
METHOD_FIELDS = {
    "name": "Short method name shown on the leaderboard",
    "authors": "Who ran it",
    "paper": "Paper URL or 'unpublished'",
    "code": "Code URL or 'not released'",
    "input": "rgb | rgbd | depth",
    "training_data": "kitchen-train | kitchen-train+val | external | none (zero-shot)",
    "extra_data": "Anything else used for training, e.g. 'BOP synthetic PBR', 'COCO pretrain'",
    "single_model": "yes | no: one model for all objects (KITchen rule for pose)",
    "hardware": "CPU/GPU the time was measured on (required for pose)",
}
MIN_FPS = 5.0  # KITchen paper rule: pose methods run at >= 5 fps


def pose_rules(metrics: Dict, method: Dict) -> Dict:
    """KITchen pose rules from the dataset paper: one model, >= 5 fps."""
    t = metrics.get("time_per_image_s")
    single = str(method.get("single_model", "")).strip().lower() == "yes"
    fast = t is not None and t > 0 and t <= 1.0 / MIN_FPS
    return {"single_model": single, "meets_5fps": fast,
            "fps": (1.0 / t) if t else None, "compliant": single and fast}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_card(task: str, split: str, metrics: Dict, method: Dict, bench_meta: Dict,
              pred_path: str | Path) -> Dict:
    missing = [k for k in ("name", "input", "training_data") if not method.get(k)]
    if missing:
        raise ValueError(f"method info is missing {missing}; see `kitchenbench method-template`")
    import numpy
    deps = {"python": platform.python_version(), "numpy": numpy.__version__}
    try:
        import pycocotools  # noqa: F401
        from importlib.metadata import version
        deps["pycocotools"] = version("pycocotools")
    except Exception:
        pass
    return {
        "benchmark": "KITchen",
        "benchmark_version": bench_meta["version"],
        "task": task,
        "split": split,
        "primary_metric": PRIMARY[task],
        "method": {k: method.get(k, "") for k in METHOD_FIELDS},
        "metrics": {k: metrics.get(k) for k in HEADLINE[task]},
        **({"rules": pose_rules(metrics, method)} if task == "pose" else {}),
        "breakdowns": {k: v for k, v in metrics.items() if k not in HEADLINE[task]},
        "provenance": {
            "ground_truth_sha256": bench_meta["ground_truth_sha256"],
            "predictions_sha256": sha256_file(pred_path),
            "predictions_file": Path(pred_path).name,
            "evaluator": f"kitchenbench {__version__}",
            "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "software": deps,
        },
    }


def _fmt(v, key: str = "") -> str:
    if v is None:
        return "–"
    if key.endswith("_s"):  # seconds, not a fraction
        return f"{v:.3f}"
    if isinstance(v, float):
        return f"{100 * v:.1f}" if abs(v) <= 1.0 else f"{v:.2f}"
    return str(v)


def card_markdown(card: Dict) -> str:
    keys = HEADLINE[card["task"]]
    m = card["method"]
    lines = [
        f"### {m['name']} on KITchen {card['benchmark_version']} ({card['task']}, {card['split']})",
        "",
        "| " + " | ".join(keys) + " |",
        "|" + "---|" * len(keys),
        "| " + " | ".join(_fmt(card["metrics"].get(k), k) for k in keys) + " |",
        "",
        f"Input: {m['input']}. Training data: {m['training_data']}."
        + (f" Extra data: {m['extra_data']}." if m.get("extra_data") else ""),
    ]
    if "rules" in card:
        r = card["rules"]
        fps = f"{r['fps']:.1f} fps" if r["fps"] else "no time"
        lines.append(f"Speed: {fps} on {m.get('hardware') or 'unstated hardware'}. "
                     f"Single model: {'yes' if r['single_model'] else 'no'}. "
                     f"KITchen rules (single model, >= 5 fps): {'met' if r['compliant'] else 'NOT met'}.")
    lines += [
        f"Ground truth {card['provenance']['ground_truth_sha256'][:12]}, "
        f"predictions {card['provenance']['predictions_sha256'][:12]}, "
        f"{card['provenance']['evaluator']}.",
    ]
    return "\n".join(lines) + "\n"


def card_latex(card: Dict) -> str:
    keys = [k for k in HEADLINE[card["task"]] if k != "time_per_image_s"]
    vals = " & ".join(_fmt(card["metrics"].get(k), k) for k in keys)
    return (f"% KITchen {card['benchmark_version']} {card['task']} {card['split']}: "
            f"{' & '.join(keys)}\n{card['method']['name']} & {vals} \\\\\n")


def leaderboard(cards: List[Dict], task: str) -> str:
    """Markdown leaderboard for one task, grouped by input modality."""
    cards = [c for c in cards if c["task"] == task and c["split"] == "test"]
    keys = HEADLINE[task]
    prim = PRIMARY[task]
    out = [f"## KITchen leaderboard: {task}", ""]
    for modality in sorted({c["method"]["input"] for c in cards}):
        rows = sorted((c for c in cards if c["method"]["input"] == modality),
                      key=lambda c: -(c["metrics"].get(prim) or 0))
        pose = task == "pose"
        if pose:
            rows = sorted(rows, key=lambda c: not c.get("rules", {}).get("compliant", False))
        out += [f"**Input: {modality}**", "",
                "| # | Method | Training data | " + " | ".join(keys) + (" | Rules met" if pose else "") + " | Version |",
                "|---|---|---|" + "---|" * len(keys) + ("---|" if pose else "") + "---|"]
        for i, c in enumerate(rows, 1):
            m = c["method"]
            name = f"[{m['name']}]({m['paper']})" if str(m.get("paper", "")).startswith("http") else m["name"]
            rule = (" | yes" if c.get("rules", {}).get("compliant") else " | no") if pose else ""
            out.append(f"| {i} | {name} | {m['training_data']} | "
                       + " | ".join(_fmt(c["metrics"].get(k), k) for k in keys)
                       + rule + f" | {c['benchmark_version']} |")
        out.append("")
    return "\n".join(out)


def write_card(card: Dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{card['method']['name'].replace(' ', '_')}_{card['task']}_{card['split']}"
    (out / f"{stem}.json").write_text(json.dumps(card, indent=2))
    (out / f"{stem}.md").write_text(card_markdown(card))
    (out / f"{stem}.tex").write_text(card_latex(card))
    return out / f"{stem}.json"
