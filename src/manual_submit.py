"""Submission upload that rewrites the GCS host to storage.googleapis.com.

www.googleapis.com is unreachable from this network; storage.googleapis.com is.
Resumable-session URIs authenticate via upload_id, so the host swap works.
"""
import os
import sys
import time
import json as _json

import requests

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import (
    ApiStartSubmissionUploadRequest,
    ApiCreateSubmissionRequest,
)


def _token():
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), encoding="utf-8") as f:
        return _json.load(f).get("api_token")

COMP = "arc-prize-2026-arc-agi-2"
FILE = "data/submission.json"
MSG = sys.argv[1] if len(sys.argv) > 1 else "heuristic baseline v1"

api = KaggleApi()
api.authenticate()

req = ApiStartSubmissionUploadRequest()
req.competition_name = COMP
req.file_name = os.path.basename(FILE)
req.content_length = os.path.getsize(FILE)
req.last_modified_epoch_seconds = int(os.path.getmtime(FILE))

with api.build_kaggle_client() as kaggle:
    resp = kaggle.competitions.competition_api_client.start_submission_upload(req)
    url = resp.create_url
    print("[1] got create_url:", url[:80], "...")

    # host rewrite
    if "www.googleapis.com" in url:
        url = url.replace("www.googleapis.com", "storage.googleapis.com")
        print("    rewritten to storage.googleapis.com")

    with open(FILE, "rb") as f:
        data = f.read()

    print(f"[2] uploading {len(data)} bytes ...")
    s = requests.Session()
    ok = False
    for attempt in range(6):
        t0 = time.time()
        try:
            r = s.put(url, data=data, headers={"Content-Type": "application/json"}, timeout=180)
            dt = time.time() - t0
            print(f"    attempt {attempt}: HTTP {r.status_code} in {dt:.1f}s {r.text[:120]}")
            if r.status_code in (200, 201):
                ok = True
                break
            if r.status_code in (308,):
                # resumed session wants more info; retry fresh PUT
                pass
        except Exception as e:
            print(f"    attempt {attempt}: {type(e).__name__}: {e}")
        time.sleep(4)

    if not ok:
        print("UPLOAD FAILED")
        sys.exit(1)

    print("[3] create_submission ...")
    creq = ApiCreateSubmissionRequest()
    creq.competition_name = COMP
    creq.blob_file_tokens = resp.token
    creq.submission_description = MSG
    try:
        sresp = kaggle.competitions.competition_api_client.create_submission(creq)
        print("    result:", sresp)
    except Exception as e:
        print("    FAILED:", type(e).__name__)
        # retry via raw http to see body
        import json as _json
        payload = _json.dumps({
            "competitionName": COMP,
            "blobFileTokens": resp.token,
            "submissionDescription": MSG,
        })
        r = s.post(
            "https://api.kaggle.com/v1/competitions.CompetitionApiService/CreateSubmission",
            data=payload,
            headers={"Authorization": "Bearer " + _token(),
                     "Content-Type": "application/json"},
            timeout=60,
        )
        print("    raw status:", r.status_code)
        print("    raw body:", r.text[:500])
        sys.exit(1)
