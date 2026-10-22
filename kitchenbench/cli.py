"""``kitchenbench`` command line.

    kitchenbench verify    BENCH                      check the ground truth is the frozen version
    kitchenbench validate  BENCH TASK PRED            check a prediction file's format
    kitchenbench eval      BENCH TASK PRED --method method.json [--split val]
    kitchenbench leaderboard CARD.json [...]          markdown leaderboard from result cards
    kitchenbench method-template                      print an empty method.json
    kitchenbench propose-symmetries --models DIR      (maintainers) suggest symmetries for review
    kitchenbench build     --release DIR --out BENCH  (maintainers) freeze a new benchmark version

TASK is one of det (2D boxes), segm (instance masks), pose (6D poses).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, manifest, report

TASKS = ("det", "segm", "pose")


def _scores(bench: Path, task: str, pred: Path, split: str):
    if task == "pose":
        from . import pose_eval, validate
        test = pose_eval.TestSet(bench / "bop", split)
        ests = pose_eval.load_bop_csv(pred)
        return validate.check_pose(test, ests), (lambda: pose_eval.evaluate(test, ests))
    from . import coco_eval, validate
    gt = bench / "coco" / f"{split}.json"
    res = coco_eval.load_results(pred)
    return validate.check_coco(gt, res, task), (lambda: coco_eval.evaluate(gt, res, "bbox" if task == "det" else "segm"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kitchenbench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"kitchenbench {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("verify", help="check the benchmark files against their checksums")
    p.add_argument("bench", type=Path)

    for name in ("validate", "eval"):
        p = sub.add_parser(name)
        p.add_argument("bench", type=Path)
        p.add_argument("task", choices=TASKS)
        p.add_argument("pred", type=Path)
        p.add_argument("--split", default="test", choices=("val", "test"))
        if name == "eval":
            p.add_argument("--method", type=Path, required=True, help="method.json (see method-template)")
            p.add_argument("--out", type=Path, default=Path("results"))
            p.add_argument("--force", action="store_true", help="score even if validation finds problems")

    p = sub.add_parser("leaderboard")
    p.add_argument("cards", type=Path, nargs="+")
    p.add_argument("--task", choices=TASKS)

    sub.add_parser("method-template")

    p = sub.add_parser("propose-symmetries", help="(maintainers) suggest object symmetries for review")
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--classes", type=Path, help="classes.txt for readable names")
    p.add_argument("--out", type=Path, default=Path("symmetries_proposed.json"))

    p = sub.add_parser("build", help="(maintainers) freeze a benchmark version from a KITchen release")
    from .build import add_args
    add_args(p)

    a = ap.parse_args(argv)

    if a.cmd == "verify":
        probs = manifest.verify(a.bench)
        meta = manifest.load(a.bench)
        if probs:
            print(f"KITchen benchmark {meta['version']}: {len(probs)} problem(s)")
            print("\n".join(probs[:50]))
            return 1
        print(f"KITchen benchmark {meta['version']} OK ({len(meta['files'])} files, "
              f"ground truth {meta['ground_truth_sha256'][:12]})")
        return 0

    if a.cmd in ("validate", "eval"):
        meta = manifest.load(a.bench)
        probs, run = _scores(a.bench, a.task, a.pred, a.split)
        if probs:
            print("Problems in the prediction file:\n  " + "\n  ".join(probs))
        if a.cmd == "validate":
            print("OK" if not probs else "")
            return 1 if probs else 0
        if probs and not a.force:
            print("Not scored. Fix the problems above, or pass --force.")
            return 1
        method = json.loads(a.method.read_text())
        card = report.make_card(a.task, a.split, run(), method, meta, a.pred)
        path = report.write_card(card, a.out)
        print(report.card_markdown(card))
        print(f"Result card: {path} (+ .md, .tex)")
        return 0

    if a.cmd == "leaderboard":
        cards = [json.loads(c.read_text()) for c in a.cards]
        for t in ([a.task] if a.task else sorted({c["task"] for c in cards})):
            print(report.leaderboard(cards, t))
        return 0

    if a.cmd == "method-template":
        print(json.dumps({k: f"<{v}>" for k, v in report.METHOD_FIELDS.items()}, indent=2))
        return 0

    if a.cmd == "propose-symmetries":
        from . import symmetry
        names = a.classes.read_text().splitlines() if a.classes else None
        symmetry.run(a.models, a.out, names)
        return 0

    if a.cmd == "build":
        from .build import run
        return run(a)
    return 2


if __name__ == "__main__":
    sys.exit(main())
