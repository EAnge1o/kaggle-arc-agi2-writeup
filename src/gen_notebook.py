"""Generate the heuristic submission notebook from src/solver_heuristic.py.

Usage: python gen_notebook.py [out_dir]  (default: kaggle_kernel_heuristic)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "kaggle_kernel_heuristic"

solver_src = (ROOT / "src" / "solver_heuristic.py").read_text(encoding="utf-8")

# The notebook needs the solver code verbatim, minus nothing — it's self-contained
# (numpy + collections only). Strip the module docstring header comment to keep it tidy.
lines = solver_src.split("\n")
# remove the initial docstring block
start = 0
in_doc = False
for i, l in enumerate(lines):
    if l.startswith('"""'):
        if not in_doc:
            in_doc = True
            start = i
        else:
            end = i
            break
solver_code = "\n".join(lines[:start] + ["# (solver code follows)"] + lines[end + 1:])

load_cell = [
    "# load competition data (rerun mode paths)\n",
    "import os, json\n",
    "\n",
    "candidates_paths = [\n",
    "    \"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json\",\n",
    "    \"/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json\",\n",
    "]\n",
    "test_path = None\n",
    "for p in candidates_paths:\n",
    "    if os.path.exists(p):\n",
    "        test_path = p\n",
    "        break\n",
    "if test_path is None:\n",
    "    for p in [\n",
    "        \"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json\",\n",
    "        \"/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json\",\n",
    "    ]:\n",
    "        if os.path.exists(p):\n",
    "            test_path = p\n",
    "            break\n",
    "print(\"test_path:\", test_path)\n",
    "with open(test_path, encoding=\"utf-8\") as f:\n",
    "    challenges = json.load(f)\n",
    "print(\"tasks:\", len(challenges))\n",
]

solve_cell = [
    "import time\n",
    "\n",
    "sub = {}\n",
    "t0 = time.time()\n",
    "for i, (tid, task) in enumerate(sorted(challenges.items())):\n",
    "    test_input = task[\"test\"][0][\"input\"]\n",
    "    a1, a2 = make_attempts(task, test_input)\n",
    "    sub[tid] = [{\"attempt_1\": a1, \"attempt_2\": a2}]\n",
    "    if (i + 1) % 60 == 0:\n",
    "        print(f\"solved {i+1}/{len(challenges)} ({time.time()-t0:.0f}s)\")\n",
    "\n",
    "with open(\"submission.json\", \"w\", encoding=\"utf-8\") as f:\n",
    "    json.dump(sub, f)\n",
    "print(\"wrote submission.json with\", len(sub), \"tasks in\", round(time.time()-t0, 1), \"s\")\n",
]

nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [l + "\n" for l in solver_code.split("\n")],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": load_cell,
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": solve_cell,
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

out = Path(OUT_DIR) / "arc-agi2-heuristic.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
print("wrote", out)

# sanity: exec the solver code locally to catch syntax errors
exec(compile(solver_code, "<notebook-solver>", "exec"), {})
print("solver code compiles OK")
