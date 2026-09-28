"""Generate the sample-efficiency figure from committed results.

Reads ``results/sample_efficiency.json`` and writes ``paper/sample_efficiency.pdf``.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    se = json.load(open(os.path.join(ROOT, "results", "sample_efficiency.json")))
    fr = [int(f * 100) for f in se["fractions"]]
    labels = {"wysgnn_full": "WySGNN (equivariant + symmetry)",
              "invariant": "Invariant baseline"}
    markers = {"wysgnn_full": "o-", "invariant": "s--"}

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for v, curve in se["curves"].items():
        ax.plot(fr, curve, markers.get(v, "o-"), label=labels.get(v, v), linewidth=2)
    ax.set_xlabel("Training-set fraction (\\%)" if False else "Training-set fraction (%)")
    ax.set_ylabel("Frobenius relative error (test)")
    ax.set_title("Sample efficiency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "sample_efficiency.pdf")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
