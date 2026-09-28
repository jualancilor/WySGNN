"""Run the WySGNN ablation + sample-efficiency study and commit results.

Trains the ablation variants on the (synthetic-teacher or JARVIS) piezoelectric
dataset, evaluates them on the held-out test split, and writes machine-readable
result files into ``results/`` from which the paper numbers are drawn.

Usage:
    python scripts/run_ablation.py [--n N] [--epochs E] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wysgnn.data.datamodule import PiezoDataModule  # noqa: E402
from wysgnn.eval import evaluate  # noqa: E402
from wysgnn.nn.baseline import InvariantBaseline, InvariantBaselineConfig  # noqa: E402
from wysgnn.nn.model import WySGNN, WySGNNConfig  # noqa: E402
from wysgnn.train import TrainConfig, config_hash, train  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def build_variant(name, cfg):
    m, c = cfg["model"], cfg
    if name == "invariant":
        return InvariantBaseline(InvariantBaselineConfig(
            hidden=c["baseline"]["hidden"], num_layers=c["baseline"]["num_layers"],
            cutoff=m["cutoff"], num_basis=c["baseline"]["num_basis"], neumann_enabled=False))
    if name == "invariant+neumann":
        return InvariantBaseline(InvariantBaselineConfig(
            hidden=c["baseline"]["hidden"], num_layers=c["baseline"]["num_layers"],
            cutoff=m["cutoff"], num_basis=c["baseline"]["num_basis"], neumann_enabled=True))
    wyck = name in ("wyckoff", "wysgnn_full")
    neum = name == "wysgnn_full"
    return WySGNN(WySGNNConfig(
        lmax=m["lmax"], mul=m["mul"], num_layers=m["num_layers"], cutoff=m["cutoff"],
        sh_lmax=m["sh_lmax"], out_sh_lmax=m["out_sh_lmax"], num_basis=m["num_basis"],
        radial_hidden=m["radial_hidden"], avg_num_neighbors=m["avg_num_neighbors"],
        wyckoff_enabled=wyck, neumann_enabled=neum))


VARIANTS = ["invariant", "invariant+neumann", "equivariant", "wyckoff", "wysgnn_full"]


def env_freeze():
    out = {}
    for pkg in ["torch", "torch_geometric", "e3nn", "pymatgen", "spglib", "ase", "numpy"]:
        try:
            from importlib.metadata import version
            out[pkg] = version(pkg)
        except Exception:
            out[pkg] = "?"
    out["python"] = sys.version.split()[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = yaml.safe_load(open(os.path.join(root, "config.yaml")))
    if args.quick:
        cfg["data"]["n_structures"] = 80
        cfg["train"]["epochs"] = 60
    if args.n:
        cfg["data"]["n_structures"] = args.n
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs

    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(cfg["seed"])

    print(f"[setup] building dataset n={cfg['data']['n_structures']} source={cfg['data']['source']}")
    t0 = time.time()
    dm = PiezoDataModule(
        n_structures=cfg["data"]["n_structures"], cutoff=cfg["data"]["cutoff"],
        lmax=cfg["data"]["lmax"], mul=cfg["data"]["mul"], symprec=cfg["data"]["symprec"],
        seed=cfg["seed"], source=cfg["data"]["source"],
        splits_path=os.path.join(root, cfg["data"]["splits_path"]))
    dm.setup()
    print(f"[setup] done in {time.time()-t0:.0f}s; "
          f"train/val/test = {len(dm.split_ids['train'])}/{len(dm.split_ids['val'])}/{len(dm.split_ids['test'])}")

    tcfg = TrainConfig(**cfg["train"])
    ablation = []
    for name in VARIANTS:
        print(f"\n[train] {name}")
        torch.manual_seed(cfg["seed"])
        model = build_variant(name, cfg).double()
        model, history, meta = train(model, dm, tcfg, log_every=max(1, tcfg.epochs // 4))
        metrics = evaluate(model, dm, "test", batch_size=tcfg.batch_size,
                           violation_rel_thresh=cfg["eval"]["violation_rel_thresh"])
        ablation.append({"variant": name, "metrics": metrics, "meta": meta,
                         "n_params": sum(p.numel() for p in model.parameters())})
        print(f"[eval] {name}: frob_rel={metrics['frobenius_rel_error']:.4f} "
              f"d33_mae={metrics['d33_mae']:.4f} viol_rate={metrics['symmetry_violation_rate']:.4f}")
        json.dump(ablation, open(os.path.join(RESULTS, "ablation.json"), "w"), indent=2)

    # Sample-efficiency curve (full WySGNN vs invariant baseline).
    print("\n[sample-efficiency]")
    full_train = list(dm.split_ids["train"])
    se = {"fractions": cfg["eval"]["sample_fractions"], "curves": {}}
    for variant in ["wysgnn_full", "invariant"]:
        se["curves"][variant] = []
        for frac in cfg["eval"]["sample_fractions"]:
            k = max(8, int(frac * len(full_train)))
            dm.split_ids["train"] = full_train[:k]
            torch.manual_seed(cfg["seed"])
            model = build_variant(variant, cfg).double()
            model, _, _ = train(model, dm, tcfg, verbose=False)
            m = evaluate(model, dm, "test", batch_size=tcfg.batch_size)
            se["curves"][variant].append(m["frobenius_rel_error"])
            print(f"  {variant} frac={frac:.2f} (k={k}) frob_rel={m['frobenius_rel_error']:.4f}")
        json.dump(se, open(os.path.join(RESULTS, "sample_efficiency.json"), "w"), indent=2)
    dm.split_ids["train"] = full_train

    # Run metadata.
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = "N/A (not a git repository)"
    run_meta = {"git_sha": sha, "config_hash": config_hash(cfg), "seed": cfg["seed"],
                "env": env_freeze(), "config": cfg,
                "dataset": {"source": cfg["data"]["source"],
                            "n": len(dm.samples),
                            "standardizer_scale": dm.standardizer.scale}}
    json.dump(run_meta, open(os.path.join(RESULTS, "run_meta.json"), "w"), indent=2)

    # CSV summary.
    with open(os.path.join(RESULTS, "metrics.csv"), "w") as f:
        f.write("variant,n_params,frobenius_rel_error,overall_mae,d33_mae,d15_mae,"
                "symmetry_violation_rate,mean_forced_zero_magnitude\n")
        for r in ablation:
            m = r["metrics"]
            f.write(f"{r['variant']},{r['n_params']},{m['frobenius_rel_error']:.6f},"
                    f"{m['overall_mae']:.6f},{m['d33_mae']:.6f},{m['d15_mae']:.6f},"
                    f"{m['symmetry_violation_rate']:.6f},{m['mean_forced_zero_magnitude']:.6f}\n")
    print(f"\n[done] results written to {RESULTS}")


if __name__ == "__main__":
    main()
