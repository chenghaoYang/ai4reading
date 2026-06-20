#!/usr/bin/env python3
"""No-GPU experiment prep: download & cache the evaluation benchmarks.

Pulls GSM8K / HumanEval / MBPP / MMLU / BBH through HF `datasets` (the same
suites LLaDA, Dream, and dLLM-Cache report on) and builds a small calibration
prompt set for later real-PTQ.  Needs network + `pip install datasets` but no
GPU.  Safe to run on the prototyping box to stage everything before GPU time.

Usage:
    pip install datasets
    python scripts/prepare_data.py --benchmarks gsm8k humaneval --calib-n 128
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dkv_quant.data import BENCHMARKS, load_benchmark, build_calibration_set


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS),
                    choices=list(BENCHMARKS))
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap samples per benchmark (smoke-test with e.g. 16)")
    ap.add_argument("--calib-n", type=int, default=128)
    ap.add_argument("--calib-from", default="gsm8k", choices=list(BENCHMARKS))
    ap.add_argument("--out-dir", default="results/data_manifest")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {}
    for key in args.benchmarks:
        try:
            ds = load_benchmark(key, max_samples=args.max_samples)
            manifest[key] = {"n": len(ds), "columns": list(ds.column_names),
                             "status": "ok"}
            print(f"[ok] {key:10s} {len(ds):>6d} rows  cols={ds.column_names}")
        except ImportError as e:
            print(f"[skip] {key}: {e}")
            manifest[key] = {"status": "datasets-not-installed"}
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"[err] {key}: {e}")
            manifest[key] = {"status": f"error: {e}"}

    # Build calibration set (best-effort).
    try:
        calib = build_calibration_set(args.calib_from, n=args.calib_n,
                                      max_samples=args.max_samples)
        with open(os.path.join(args.out_dir, "calibration.json"), "w") as f:
            json.dump({"source": args.calib_from, "prompts": calib}, f, indent=2)
        print(f"[ok] calibration set: {len(calib)} prompts from {args.calib_from}")
        manifest["calibration"] = {"n": len(calib), "source": args.calib_from}
    except Exception as e:  # noqa: BLE001
        print(f"[skip] calibration: {e}")
        manifest["calibration"] = {"status": f"skipped: {e}"}

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest -> {os.path.join(args.out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
