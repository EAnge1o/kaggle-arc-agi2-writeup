"""Assemble the Sunday TTT notebook: perfpatch solver + area-sorted queue + Leg C.

Combines:
- perfpatch (kaggle_kernel/failed-in-aimo.ipynb, 9 cells): speed-patched solver,
  model-path fallback, no Leg C.
- original failed-in-aimo (refs/kernels/failed-in-aimo.ipynb, 12 cells):
  starter.py with legc_skip, arc_induction_v2.py, selftest, Leg C launch, merge.

starter.py is patched to process tasks in ascending total-train-grid-area order
so small fast tasks finish first and slow 30x30 tasks fall off the time budget.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

perf = json.load(open(ROOT / "kaggle_kernel" / "failed-in-aimo.ipynb", encoding="utf-8"))
orig = json.load(open(ROOT / "refs" / "kernels" / "failed-in-aimo.ipynb", encoding="utf-8"))


def cell_by_first_line(nb, marker):
    for c in nb["cells"]:
        src = "".join(c["source"])
        if marker in src.split("\n")[0]:
            return c
    raise KeyError(marker)


def cell_containing(nb, marker):
    for c in nb["cells"]:
        if marker in "".join(c["source"]):
            return c
    raise KeyError(marker)


def cell_text(c):
    return "".join(c["source"])


# perfpatch cells
perf_c0 = cell_containing(perf, "global_end_time = time.time()")
perf_c1 = cell_containing(perf, "pip uninstall -y tensorflow")
perf_c2 = cell_containing(perf, "%%writefile arc_loader.py")
perf_c3 = cell_containing(perf, "%%writefile arc_decoder.py")
perf_c4 = cell_containing(perf, "%%writefile arc_solver.py")  # speed-patched + fallback
perf_launch = cell_containing(perf, "UNSLOTH_DISABLE_STATISTICS")

# original cells
orig_starter = cell_containing(orig, "%%writefile starter.py")
orig_ind = cell_containing(orig, "%%writefile arc_induction_v2.py")
orig_selftest = next(
    c for c in orig["cells"]
    if "".join(c["source"]).split("\n")[0].startswith("# ============ Leg C sandbox selftest")
)
orig_legc = next(
    c for c in orig["cells"]
    if "".join(c["source"]).split("\n")[0].startswith("# ===================== Leg C launch")
)
orig_sub = cell_containing(orig, "decoder.load_decoded_results")
orig_merge = next(
    c for c in orig["cells"]
    if "".join(c["source"]).split("\n")[0].startswith("# ===================== Leg C merge")
)

# patch starter.py: area-ascending queue order
starter_src = cell_text(orig_starter)
old_loop = """    for key in sorted(data.keys()):
        if not rerun_mode:
            if key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:
                continue
        if key in legc_skip:
            continue
        queue.put(key)"""
new_loop = """    def task_area(key):
        t = data[key]
        return sum(
            len(p["input"]) * len(p["input"][0]) +
            len(p["output"]) * len(p["output"][0])
            for p in t["train"]
        )

    # small grids first: fast tasks finish early, slow 30x30 tasks take the
    # leftover budget (throughput optimisation)
    for key in sorted(data.keys(), key=lambda k: (task_area(k), k)):
        if not rerun_mode:
            if key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:
                continue
        if key in legc_skip:
            continue
        queue.put(key)"""
assert old_loop in starter_src, "starter loop pattern not found"
starter_src = starter_src.replace(old_loop, new_loop)
orig_starter["source"] = starter_src.splitlines(keepends=True)

cells = [
    perf_c0, perf_c1, perf_c2, perf_c3, perf_c4,
    orig_starter, orig_ind, orig_selftest, orig_legc,
    perf_launch, orig_sub, orig_merge,
]

nb = {
    "cells": cells,
    "metadata": perf["metadata"],
    "nbformat": 4,
    "nbformat_minor": 4,
}

out = ROOT / "kaggle_kernel" / "failed-in-aimo.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
print(f"assembled {len(cells)} cells ->", out)
for i, c in enumerate(cells):
    src = "".join(c["source"])
    print(f"  {i}: {src.strip().split(chr(10))[0][:70]}")
