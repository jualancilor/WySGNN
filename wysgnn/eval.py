"""U11 — Evaluation metrics.

Computes, on a split, in physical units (after inverse standardisation):

* per-component Voigt (3x6) MAE and overall MAE,
* tensor Frobenius relative error,
* targeted ``d33`` / ``d15`` MAE (device-relevant components),
* **symmetry-violation rate**: fraction of Neumann-forced-zero components that a
  model predicts above a threshold (zero by construction for Neumann models;
  the headline qualitative result against invariant baselines).
"""

from __future__ import annotations

import torch

from .nn.irreps import full_to_voigt
from .nn.readout import neumann_project
from .train import model_uses_neumann

# Voigt locations of device-relevant components.
_D33 = (2, 2)
_D15 = (0, 4)


def forced_zero_mask(ops: torch.Tensor, n_samples: int = 64, tol: float = 1e-6) -> torch.Tensor:
    """Boolean ``(3,6)`` mask of Voigt components forced to zero by Neumann.

    A component is forced-zero if the Neumann projection of random tensors never
    populates it.
    """
    rnd = torch.randn(n_samples, 3, 3, 3, dtype=ops.dtype)
    proj = torch.stack([neumann_project(rnd[i], ops) for i in range(n_samples)])
    voigt = full_to_voigt(proj)  # (n,3,6)
    return voigt.abs().amax(dim=0) < tol


@torch.no_grad()
def evaluate(model, dm, split: str, batch_size: int = 16, violation_rel_thresh: float = 0.01) -> dict:
    model.eval()
    scale = dm.standardizer.scale
    thresh = violation_rel_thresh * scale

    abs_err_sum = torch.zeros(3, 6, dtype=torch.float64)
    n_comp = 0
    sq_num, sq_den = 0.0, 0.0  # aggregate Frobenius relative error
    d33_err, d15_err = 0.0, 0.0
    viol_forced, viol_above = 0, 0
    viol_mag_sum = 0.0

    use_n = model_uses_neumann(model)
    for batch, ops, _y, y_raw, _meta in dm.iter_batches(split, batch_size):
        pred_norm = model(batch, ops if use_n else None)
        pred = dm.standardizer.inverse(pred_norm)
        target = y_raw

        pv = full_to_voigt(pred)      # (B,3,6) physical
        tv = full_to_voigt(target)
        abs_err_sum += (pv - tv).abs().sum(dim=0)
        n_comp += pv.shape[0]

        # Aggregate (dataset-level) Frobenius relative error: robust to
        # individual structures whose target tensor has near-zero norm.
        sq_num += (pred - target).pow(2).sum().item()
        sq_den += target.pow(2).sum().item()

        d33_err += (pv[:, _D33[0], _D33[1]] - tv[:, _D33[0], _D33[1]]).abs().sum().item()
        d15_err += (pv[:, _D15[0], _D15[1]] - tv[:, _D15[0], _D15[1]]).abs().sum().item()

        for b in range(pv.shape[0]):
            mask = forced_zero_mask(ops[b])
            forced = pv[b][mask]
            viol_forced += int(mask.sum())
            viol_above += int((forced.abs() > thresh).sum())
            viol_mag_sum += float(forced.abs().sum())

    per_comp_mae = (abs_err_sum / max(n_comp, 1))
    return {
        "split": split,
        "n": n_comp,
        "overall_mae": float(per_comp_mae.mean()),
        "per_component_mae": per_comp_mae.tolist(),
        "frobenius_rel_error": (sq_num / sq_den) ** 0.5 if sq_den > 1e-12 else 0.0,
        "d33_mae": d33_err / max(n_comp, 1),
        "d15_mae": d15_err / max(n_comp, 1),
        "symmetry_violation_rate": viol_above / max(viol_forced, 1),
        "mean_forced_zero_magnitude": viol_mag_sum / max(viol_forced, 1),
        "n_forced_zero_components": viol_forced,
    }
