"""EDA for ARC-AGI-2 competition data."""
import json
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

def load(name):
    with open(DATA / name, encoding="utf-8") as f:
        return json.load(f)

for name in [
    "arc-agi_training_challenges.json",
    "arc-agi_training_solutions.json",
    "arc-agi_evaluation_challenges.json",
    "arc-agi_evaluation_solutions.json",
    "arc-agi_test_challenges.json",
    "sample_submission.json",
]:
    d = load(name)
    print(f"== {name}: {len(d)} items, type={type(d).__name__}")
    if d:
        k0 = next(iter(d))
        print(f"   first key: {k0!r}")
        v0 = d[k0] if isinstance(d, dict) else d[0]
        if isinstance(v0, dict):
            print(f"   first value keys: {list(v0.keys())[:10]}")
            # one nested level
            for kk, vv in v0.items():
                if isinstance(vv, list):
                    print(f"      {kk}: list[{len(vv)}]")
                elif isinstance(vv, dict):
                    print(f"      {kk}: dict{list(vv.keys())[:6]}")
                else:
                    print(f"      {kk}: {type(vv).__name__}")
        else:
            print(f"   first value: {str(v0)[:120]}")

# training task structure
train = load("arc-agi_training_challenges.json")
sol = load("arc-agi_training_solutions.json")
print("\n== training task detail")
tid = next(iter(train))
t = train[tid]
print(f"task {tid}: {list(t.keys())}")
for pair in t.get("train", []):
    inp = pair["input"]; out = pair["output"]
    print(f"  train pair: in {len(inp)}x{len(inp[0]) if inp else 0} -> out {len(out)}x{len(out[0]) if out else 0}")
print("  test count:", len(t.get("test", [])))
print("  solution for task:", sol.get(tid))

# grid size / color stats across training
sizes_in, sizes_out = Counter(), Counter()
colors = Counter()
for tid, t in train.items():
    for pair in t["train"]:
        sizes_in[(len(pair["input"]), len(pair["input"][0]))] += 1
        sizes_out[(len(pair["output"]), len(pair["output"][0]))] += 1
        for row in pair["input"] + pair["output"]:
            for c in row:
                colors[c] += 1
print("\n== training stats")
print("num train tasks:", len(train))
print("num train pairs total:", sum(len(t['train']) for t in train.values()))
print("num test pairs total:", sum(len(t['test']) for t in train.values()))
print("input sizes top10:", sizes_in.most_common(10))
print("output sizes top10:", sizes_out.most_common(10))
print("colors:", dict(sorted(colors.items())))
