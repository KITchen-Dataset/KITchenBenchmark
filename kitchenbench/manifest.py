"""benchmark.json: version, split definition and checksums of the frozen ground truth."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

from .report import sha256_file

MANIFEST = "benchmark.json"
GT_DIRS = ("coco", "bop", "splits")


def gt_files(bench: Path):
    for d in GT_DIRS:
        for p in sorted((bench / d).rglob("*")):
            if p.is_file() and not p.is_symlink():
                yield p


def digest(files: Dict[str, str]) -> str:
    h = hashlib.sha256()
    for k in sorted(files):
        h.update(f"{k} {files[k]}\n".encode())
    return h.hexdigest()


def write(bench: Path, meta: Dict) -> Dict:
    files = {str(p.relative_to(bench)): sha256_file(p) for p in gt_files(bench)}
    meta = {**meta, "files": files, "ground_truth_sha256": digest(files)}
    (bench / MANIFEST).write_text(json.dumps(meta, indent=2))
    return meta


def load(bench: str | Path) -> Dict:
    p = Path(bench) / MANIFEST
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: is {bench} a built KITchen benchmark folder?")
    return json.loads(p.read_text())


def verify(bench: str | Path) -> list:
    """Return a list of problems (empty when every file matches its checksum)."""
    bench = Path(bench)
    meta = load(bench)
    problems = []
    for rel, sha in meta["files"].items():
        p = bench / rel
        if not p.exists():
            problems.append(f"missing: {rel}")
        elif sha256_file(p) != sha:
            problems.append(f"changed: {rel}")
    extra = {str(p.relative_to(bench)) for p in gt_files(bench)} - set(meta["files"])
    problems += [f"unexpected: {r}" for r in sorted(extra)]
    if digest(meta["files"]) != meta["ground_truth_sha256"]:
        problems.append("benchmark.json checksum list was edited")
    return problems
