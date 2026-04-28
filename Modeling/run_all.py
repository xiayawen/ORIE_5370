"""End-to-end orchestrator. Builds data, trains models, evaluates, plots.

Usage:

    python run_all.py            # full pipeline
    python run_all.py --skip-data    # reuse existing panel.npz
    python run_all.py --skip-train   # reuse trained models in results/models/

Each step prints its progress and timing. The data step is by far the slowest
on a fresh machine (it reads ~500 ticker CSVs); training is dominated by the
IPO sweeps. Evaluation and plotting take a few seconds each.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


# OpenMP safety. On Anaconda + PyTorch on macOS we have observed two issues:
#   1. ``libiomp5`` and ``libomp`` are both linked into the process, which
#      prints "OMP: Error #15" and aborts. Setting ``KMP_DUPLICATE_LIB_OK=TRUE``
#      lets the process proceed.
#   2. Multi-threaded BLAS deadlocks during the MLP plug-in training (the
#      python process goes to 0% CPU and never returns). Pinning OMP / MKL
#      threads to 1 avoids that path entirely. The pipeline is plenty fast
#      single-threaded on the cross-section sizes we use here.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


HERE = Path(__file__).resolve().parent


def _step(name: str, fn) -> None:
    print(f"\n========== {name} ==========")
    t0 = time.time()
    fn()
    print(f"========== {name} done in {time.time() - t0:.1f}s ==========")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true",
                        help="reuse data_cache/panel.npz instead of rebuilding")
    parser.add_argument("--skip-train", action="store_true",
                        help="reuse results/models/ instead of retraining")
    args = parser.parse_args()

    if not args.skip_data:
        import build_dataset
        _step("build_dataset", build_dataset.main)
    else:
        print("skipping build_dataset (--skip-data)")

    if not args.skip_train:
        import train
        _step("train", train.main)
    else:
        print("skipping train (--skip-train)")

    import evaluate
    _step("evaluate", evaluate.main)

    import make_figures
    _step("make_figures", make_figures.main)


if __name__ == "__main__":
    main()
