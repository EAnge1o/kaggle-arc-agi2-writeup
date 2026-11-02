"""Saturday TTT submission automation.

Retries `kaggle kernels push` until the weekly GPU quota is available
(resets ~Saturday 00:00 UTC), then submits the notebook as a code submission.
Robust to: GPU quota exhausted, BYOD docker image rejection, transient network.
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_DIR = os.path.join(ROOT, "kaggle_kernel")
META = os.path.join(KERNEL_DIR, "kernel-metadata.json")
LOG = os.path.join(ROOT, "scripts", "ttt_submit.log")
COMP = "arc-prize-2026-arc-agi-2"
KERNEL_ID = os.environ.get("KAGGLE_KERNEL_ID", "your-kaggle-username/arc-agi2-ttt-run1")
OUT_FILE = "submission.json"
DESC = "TTT run: failed-in-aimo fork (Qwen3-4B LoRA + turbo_dfs, 4xL4)"


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def push_kernel():
    """Return version number on success, or (None, error_message)."""
    r = subprocess.run(
        ["kaggle", "kernels", "push", "-p", KERNEL_DIR],
        capture_output=True, text=True, cwd=ROOT, timeout=600,
    )
    out = r.stdout + r.stderr
    log(f"push exit={r.returncode}: {out[-300:]}")
    m = re.search(r"version (\d+) successfully pushed", out)
    if m:
        return int(m.group(1)), None
    if "Maximum weekly GPU quota" in out or "GPU quota" in out:
        return None, "quota"
    if "docker" in out.lower() or "image" in out.lower():
        return None, "docker"
    return None, out[-200:]


def submit_code(version):
    token = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json"),
                           encoding="utf-8")).get("api_token")
    payload = {
        "competitionName": COMP,
        "kernelOwner": KERNEL_ID.split("/")[0],
        "kernelSlug": KERNEL_ID.split("/")[1],
        "kernelVersion": version,
        "fileName": OUT_FILE,
        "submissionDescription": DESC,
    }
    r = requests.post(
        "https://api.kaggle.com/v1/competitions.CompetitionApiService/CreateCodeSubmission",
        json=payload,
        headers={"Authorization": "Bearer " + token},
        timeout=120,
    )
    log(f"CreateCodeSubmission status={r.status_code} body={r.text[:300]}")
    return r.status_code == 200


def main():
    log("=== TTT submission automation start ===")
    deadline = time.time() + 12 * 3600  # retry window: 12h
    pushed_version = None
    last_submit_ts = time.time()
    while time.time() < deadline:
        if pushed_version is None:
            version, err = push_kernel()
            if version is not None:
                pushed_version = version
                log(f"kernel version {version} pushed; will submit with retries")
            elif err == "docker":
                log("docker image rejected; falling back to default image")
                meta = json.load(open(META, encoding="utf-8"))
                meta.pop("docker_image", None)
                json.dump(meta, open(META, "w", encoding="utf-8"), indent=2)
                continue  # retry immediately
            else:
                # quota or transient push error: wait and retry
                time.sleep(20 * 60)
                continue
        # submit the SAME pushed version, retrying as the version settles
        last_submit_ts = time.time()
        if submit_code(pushed_version):
            log(f"SUCCESS: submitted kernel version {pushed_version}")
            return 0
        # could be version-processing lag or transient error: wait and resubmit same version
        # but if too much time passes (1h+), re-push to refresh the version
        if time.time() - last_submit_ts > 3600:
            pushed_version = None
        else:
            time.sleep(5 * 60)
    log("FAILED: could not submit TTT kernel within retry window")
    return 1


if __name__ == "__main__":
    sys.exit(main())
