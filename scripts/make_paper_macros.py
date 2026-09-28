"""Generate LaTeX macros + tables from committed result files.

Reads ``results/ablation.json``, ``results/sample_efficiency.json`` and
``results/run_meta.json`` and writes ``paper/results_macros.tex`` (number macros)
plus ``paper/ablation_table.tex`` and ``paper/sampleeff_table.tex``.  Every paper
number therefore traces directly to ``eval.py`` output.
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PAPER = os.path.join(ROOT, "paper")

PRETTY = {
    "invariant": "Invariant baseline (CGCNN/MEGNet-style)",
    "invariant+neumann": "Invariant + Neumann readout",
    "equivariant": "Equivariant (no projections)",
    "wyckoff": "Equivariant + Wyckoff",
    "wysgnn_full": "WySGNN (Wyckoff + Neumann)",
}
ORDER = ["invariant", "invariant+neumann", "equivariant", "wyckoff", "wysgnn_full"]


def fmt(x, p=3):
    return f"{x:.{p}f}"


def main():
    ablation = json.load(open(os.path.join(RES, "ablation.json")))
    se = json.load(open(os.path.join(RES, "sample_efficiency.json")))
    meta = json.load(open(os.path.join(RES, "run_meta.json")))
    by = {r["variant"]: r for r in ablation}

    macros = []

    def mac(name, val):
        macros.append(rf"\newcommand{{\{name}}}{{{val}}}")

    mac("gitSHA", meta["git_sha"].replace("_", r"\_"))
    mac("configHash", meta["config_hash"])
    mac("seedVal", str(meta["seed"]))
    mac("datasetSource", meta["dataset"]["source"])
    mac("datasetN", str(meta["dataset"]["n"]))
    mac("envTorch", meta["env"].get("torch", "?"))
    mac("envEthreenn", meta["env"].get("e3nn", "?"))
    mac("envPy", meta["env"].get("python", "?"))

    cmd = {"invariant": "Inv", "invariant+neumann": "InvN", "equivariant": "Eqv",
           "wyckoff": "Wyck", "wysgnn_full": "Wys"}
    for v, c in cmd.items():
        if v not in by:
            continue
        m = by[v]["metrics"]
        mac(f"frob{c}", fmt(m["frobenius_rel_error"]))
        mac(f"mae{c}", fmt(m["overall_mae"]))
        mac(f"dthree{c}", fmt(m["d33_mae"]))
        mac(f"viol{c}", fmt(m["symmetry_violation_rate"]))
        mac(f"vmag{c}", f"{m['mean_forced_zero_magnitude']:.2e}")

    # Difference among equivariant variants (the no-op equivalence, empirical).
    if all(k in by for k in ["equivariant", "wyckoff", "wysgnn_full"]):
        fs = [by[k]["metrics"]["frobenius_rel_error"] for k in
              ["equivariant", "wyckoff", "wysgnn_full"]]
        mac("eqvSpread", f"{max(fs) - min(fs):.2e}")

    open(os.path.join(PAPER, "results_macros.tex"), "w").write("\n".join(macros) + "\n")

    # Ablation table.
    rows = []
    for v in ORDER:
        if v not in by:
            continue
        m = by[v]["metrics"]
        rows.append(
            f"{PRETTY[v]} & {by[v]['n_params']:,} & {fmt(m['frobenius_rel_error'])} & "
            f"{fmt(m['overall_mae'])} & {fmt(m['d33_mae'])} & {fmt(m['symmetry_violation_rate'])} \\\\"
        )
    table = (
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Model & Params & Frob.\\ rel.\\ err.\\ $\\downarrow$ & MAE $\\downarrow$ & "
        "$d_{33}$ MAE $\\downarrow$ & Sym.\\ viol.\\ rate $\\downarrow$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    open(os.path.join(PAPER, "ablation_table.tex"), "w").write(table)

    # Sample-efficiency table.
    fr = se["fractions"]
    head = " & ".join([f"{int(f*100)}\\%" for f in fr])
    lines = ["\\begin{tabular}{l" + "r" * len(fr) + "}", "\\toprule",
             f"Model & {head} \\\\", "\\midrule"]
    label = {"wysgnn_full": "WySGNN", "invariant": "Invariant baseline"}
    for v, curve in se["curves"].items():
        lines.append(label.get(v, v) + " & " + " & ".join(fmt(x) for x in curve) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(PAPER, "sampleeff_table.tex"), "w").write("\n".join(lines) + "\n")

    print("wrote paper/results_macros.tex, ablation_table.tex, sampleeff_table.tex")


if __name__ == "__main__":
    main()
