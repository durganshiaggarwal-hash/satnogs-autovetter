#!/usr/bin/env python3
"""Phase 7 demonstration: talking to the backend the way the GUI will.

Needs the server running first, in a separate terminal::

    python -m satrecovery.backend

Then, in a second terminal::

    python scripts/demo_phase7.py

Uses only the standard library (``urllib``), to show there is nothing
special about the GUI's access: any program that can send HTTP can drive
the whole project.  Takes about a minute, most of it training one model.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"


def call(method: str, path: str, body: dict | None = None):
    """Send one request and return (status, decoded JSON)."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        BASE + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
            return response.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"null")


def wait(job: dict, label: str) -> dict:
    """Poll a job, drawing a text progress bar -- what the GUI will draw graphically."""
    while True:
        _, job = call("GET", f"/api/jobs/{job['id']}")
        filled = int(30 * job["fraction"])
        print(f"\r  {label} [{'#' * filled}{'.' * (30 - filled)}] {job['state']:8s}", end="", flush=True)
        if job["state"] in ("done", "failed"):
            print()
            return job
        time.sleep(0.3)


def main() -> None:
    try:
        _, health = call("GET", "/api/health")
    except urllib.error.URLError:
        print("No server at", BASE)
        print("Start it first, in another terminal:  python -m satrecovery.backend")
        sys.exit(1)

    print("Phase 7 -- the backend, driven over HTTP")
    print("=" * 70)
    print(f"Server is up. Data folder: {health['data_dir']}")
    print(f"Interactive docs: {BASE}/docs\n")

    # ------------------------------------------------------------------
    print("1. What can be asked for")
    print("-" * 70)
    _, schemes = call("GET", "/api/modulations")
    print("  modulations:", ", ".join(s["name"] for s in schemes))
    _, receivers = call("GET", "/api/receivers?modulation=16QAM")
    for r in receivers:
        ready = "ready" if r["available_for"] else "needs a trained model"
        print(f"  {r['name']:20s} {ready}")

    # ------------------------------------------------------------------
    print("\n2. One frame, two receivers -- what the GUI's main button does")
    print("-" * 70)
    request = {
        "modulation": "16QAM",
        "conditions": {"snr_db": 4, "freq_offset_hz": 700, "timing_offset": 0.3},
        "receivers": ["classical", "classical-hard"],
    }
    _, body = call("POST", "/api/experiments", request)
    print(f"  seed {body['frame']['seed']} (send it back to get this exact frame again)")
    for r in body["results"]:
        ranking = "n/a" if r["ranking_quality"] is None else f"{r['ranking_quality']:.3f}"
        print(f"  {r['receiver']:16s} {r['bit_errors']:3d} of {r['n_bits']} bits wrong   "
              f"ranking quality {ranking}   ECE {r['ece']:.4f}")
    print("  Same bits wrong, different self-knowledge -- the Phase 5 finding, live.")

    # ------------------------------------------------------------------
    print("\n3. Following the frame through every stage")
    print("-" * 70)
    _, stages = call("POST", "/api/experiments/stages", {
        "modulation": request["modulation"],
        "conditions": request["conditions"],
        "receiver": "classical",
        "seed": body["frame"]["seed"],
    })
    for stage in stages["stages"]:
        print(f"  {stage['name']:14s} {len(stage['samples']['i']):5d} values  ({stage['rate']} rate)")

    # ------------------------------------------------------------------
    print("\n4. Mistakes are refused clearly, before any work starts")
    print("-" * 70)
    for label, bad in (
        ("SNR out of range", {"conditions": {"snr_db": 99}}),
        ("misspelt field", {"modulaton": "16QAM"}),
        ("unknown receiver", {"receivers": ["magic"]}),
    ):
        status, error = call("POST", "/api/experiments", bad)
        detail = error.get("detail") if isinstance(error, dict) else error
        if isinstance(detail, list):   # FastAPI's own type errors come as a list
            detail = detail[0].get("msg")
        print(f"  {label:18s} -> {status}: {str(detail)[:70]}")

    # ------------------------------------------------------------------
    print("\n5. Training a hybrid, as a background job")
    print("-" * 70)
    _, job = call("POST", "/api/models/16QAM/train", {"frames_each": 6, "epochs": 60})
    job = wait(job, "training")
    if job["state"] == "done":
        print(f"  trained on {job['result']['rows']} symbols in {job['result']['seconds']}s")
    else:
        print("  failed:", job["error"])

    # ------------------------------------------------------------------
    print("\n6. A small benchmark, as a background job")
    print("-" * 70)
    status, job = call("POST", "/api/benchmarks", {
        "name": "phase 7 demo",
        "modulations": ["16QAM"],
        "snrs_db": [0, 4, 8],
        "profiles": ["clean", "full"],
        "receivers": ["classical", "hybrid", "hybrid-noise"],
        "frames_per_scenario": 10,
    })
    job = wait(job, "benchmark")
    run_id = job["result"]["run_id"]
    _, cells = call("GET", f"/api/runs/{run_id}/cells?profile=full")
    print(f"\n  stored as run {run_id}; the 'full' profile:")
    print(f"  {'receiver':14s} {'SNR':>4s} {'BER':>10s} {'95% interval':>24s}")
    for c in cells:
        print(f"  {c['receiver']:14s} {c['snr_db']:4.0f} {c['ber']:10.2e} "
              f"[{c['ber_low']:.2e}, {c['ber_high']:.2e}]")

    print("\n" + "=" * 70)
    print("Everything above went over HTTP. The GUI in Phase 8 does the same,")
    print(f"and so can a browser: open {BASE}/docs and try any endpoint.")


if __name__ == "__main__":
    main()
