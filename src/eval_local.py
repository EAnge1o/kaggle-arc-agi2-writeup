"""Local validation + submission generation for the heuristic solver."""
import json
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solver_heuristic import make_attempts, solve_task

DATA = Path(__file__).resolve().parent.parent / "data"


def evaluate(challenges_path, solutions_path, max_tasks=None, verbose=True):
    with open(challenges_path, encoding="utf-8") as f:
        challenges = json.load(f)
    with open(solutions_path, encoding="utf-8") as f:
        solutions = json.load(f)

    tasks = sorted(challenges.items())
    if max_tasks:
        tasks = tasks[:max_tasks]

    n_a1 = n_a2 = n_any = 0
    n = 0
    t0 = time.time()
    rule_hits = Counter()
    for tid, task in tasks:
        test_input = task["test"][0]["input"]
        sol = solutions[tid][0]
        # tag of the first rule that solves it, for coverage analysis
        tagged = solve_task([(p["input"], p["output"]) for p in task["train"]],
                            test_input, with_tags=True)
        for tag, g in tagged:
            if g == sol:
                rule_hits[tag] += 1
                break
        a1, a2 = make_attempts(task, test_input)
        hit1 = (a1 == sol)
        hit2 = (a2 == sol)
        n_a1 += hit1
        n_a2 += hit2
        n_any += (hit1 or hit2)
        n += 1
    dt = time.time() - t0
    if verbose:
        print(f"{Path(challenges_path).name}: tasks={n} attempt1={n_a1} "
              f"attempt2_only={n_a2} any={n_any} ({n_any/max(n,1)*100:.2f}%) "
              f"time={dt:.1f}s ({dt/max(n,1):.2f}s/task)")
        print("rule coverage:", rule_hits.most_common(25))
    return n_any, n


def make_submission(challenges_path, out_path, max_tasks=None):
    with open(challenges_path, encoding="utf-8") as f:
        challenges = json.load(f)
    tasks = sorted(challenges.items())
    if max_tasks:
        tasks = tasks[:max_tasks]
    sub = {}
    for i, (tid, task) in enumerate(tasks):
        test_input = task["test"][0]["input"]
        a1, a2 = make_attempts(task, test_input)
        sub[tid] = [{"attempt_1": a1, "attempt_2": a2}]
        if (i + 1) % 40 == 0:
            print(f"  solved {i+1}/{len(tasks)}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sub, f)
    print(f"wrote {out_path} with {len(sub)} tasks")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":
        evaluate(DATA / "arc-agi_training_challenges.json",
                 DATA / "arc-agi_training_solutions.json")
    elif mode == "eval":
        evaluate(DATA / "arc-agi_evaluation_challenges.json",
                 DATA / "arc-agi_evaluation_solutions.json")
    elif mode == "submit":
        make_submission(DATA / "arc-agi_test_challenges.json",
                        DATA / "submission.json")
    elif mode == "all":
        evaluate(DATA / "arc-agi_training_challenges.json",
                 DATA / "arc-agi_training_solutions.json")
        evaluate(DATA / "arc-agi_evaluation_challenges.json",
                 DATA / "arc-agi_evaluation_solutions.json")
