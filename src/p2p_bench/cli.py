"""CLI for p2p-bench."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main() -> None:
    p = argparse.ArgumentParser(prog="p2p-bench")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Run P2P matrix")
    r.add_argument("--config", type=Path)
    r.add_argument("--out", type=Path, default=Path("results/p2p_run"))
    r.add_argument("--gpus", type=str, default="all")
    r.add_argument("--size-gb", type=float)

    args = r.parse_args()
    cfg = {}
    if args.config and args.config.is_file():
        cfg = yaml.safe_load(args.config.read_text()) or {}

    try:
        import torch
    except ImportError:
        print("PyTorch CUDA required.", file=sys.stderr)
        sys.exit(2)
    if not torch.cuda.is_available():
        print("CUDA not available.", file=sys.stderr)
        sys.exit(3)

    n = torch.cuda.device_count()
    gpus = list(range(n)) if args.gpus == "all" else [int(x) for x in args.gpus.split(",")]
    size_gb = args.size_gb or float(cfg.get("size_gb", 1.0))
    iterations = int(cfg.get("iterations", 50))
    warmup = int(cfg.get("warmup", 5))

    from p2p_bench.p2p import measure_p2p_gbps
    from p2p_bench.report import write_json, write_p2p_csv

    rows = []
    for src in gpus:
        for dst in gpus:
            if src == dst:
                continue
            res = measure_p2p_gbps(src, dst, size_gb, iterations, warmup)
            if res is None:
                rows.append({"src_gpu": src, "dst_gpu": dst, "gbps": "", "ok": False})
            else:
                rows.append({"src_gpu": src, "dst_gpu": dst, "gbps": round(res.gbps, 2), "ok": True})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_p2p_csv(out / "p2p_pairs.csv", rows)
    write_json(out / "metadata.json", {"size_gb": size_gb, "iterations": iterations, "gpus": gpus})
    print(f"Wrote {out / 'p2p_pairs.csv'}")


if __name__ == "__main__":
    main()
