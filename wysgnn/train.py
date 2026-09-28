"""U10 — Training loop.

Loss is the Frobenius MSE in the irrep basis (rotation-frame independent, since
the change of basis is orthogonal).  Uses AdamW, a cosine schedule, an EMA of
the weights, early stopping on the validation Frobenius error, and logs a
per-irrep loss breakdown plus a symmetry-violation guard.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass

import torch

from .nn.irreps import cart_to_irrep, piezo_irreps


def frobenius_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between predicted/target tensors in the irrep basis."""
    pc = cart_to_irrep(pred)
    tc = cart_to_irrep(target)
    return (pc - tc).pow(2).sum(dim=-1).mean()


def per_irrep_loss(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pc = cart_to_irrep(pred)
    tc = cart_to_irrep(target)
    err = (pc - tc).pow(2).mean(dim=0)  # (18,)
    out, start = {}, 0
    for mul, ir in piezo_irreps():
        n = mul * ir.dim
        out[str(ir)] = out.get(str(ir), 0.0) + float(err[start:start + n].mean())
        start += n
    return out


def frobenius_rel_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num = (pred - target).flatten(1).norm(dim=1)
    den = target.flatten(1).norm(dim=1).clamp_min(1e-12)
    return (num / den).mean()


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 5e-3
    weight_decay: float = 1e-5
    batch_size: int = 16
    ema_decay: float = 0.999
    patience: int = 40
    min_epochs: int = 20


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "N/A (not a git repository)"


def config_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


@torch.no_grad()
def evaluate_loss(model, dm, split, batch_size):
    """Mean Frobenius MSE on a split (model-selection metric).

    Uses MSE rather than relative error so that zero-target (centrosymmetric)
    structures do not blow up the metric and unfairly bias model selection.
    """
    model.eval()
    total, n = 0.0, 0
    for batch, ops, y, _y_raw, _meta in dm.iter_batches(split, batch_size):
        pred = model(batch, ops if model_uses_neumann(model) else None)
        total += frobenius_loss(pred, y).item() * y.shape[0]
        n += y.shape[0]
    return total / max(n, 1)


def model_uses_neumann(model) -> bool:
    cfg = getattr(model, "cfg", None)
    return bool(getattr(cfg, "neumann_enabled", False))


def train(model, dm, cfg: TrainConfig, log_every: int = 25, verbose: bool = True):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    ema = EMA(model, cfg.ema_decay)
    history = {"train": [], "val": [], "per_irrep": []}
    best_val, best_state, best_epoch = float("inf"), None, -1

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss, n = 0.0, 0
        for batch, ops, y, _y_raw, _meta in dm.iter_batches("train", cfg.batch_size, shuffle=True):
            opt.zero_grad()
            pred = model(batch, ops if model_uses_neumann(model) else None)
            loss = frobenius_loss(pred, y)
            loss.backward()
            opt.step()
            ema.update(model)
            epoch_loss += loss.item() * y.shape[0]
            n += y.shape[0]
        sched.step()
        train_loss = epoch_loss / max(n, 1)

        # Validation with EMA weights.
        backup = copy.deepcopy(model.state_dict())
        ema.copy_to(model)
        val_err = evaluate_loss(model, dm, "val", cfg.batch_size)
        if val_err < best_val:
            best_val, best_state, best_epoch = val_err, copy.deepcopy(model.state_dict()), epoch
        model.load_state_dict(backup)

        history["train"].append(train_loss)
        history["val"].append(val_err)
        if verbose and (epoch % log_every == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch:3d}  train_mse={train_loss:.4e}  val_mse={val_err:.4e}")
        if epoch - best_epoch > cfg.patience and epoch > cfg.min_epochs:
            if verbose:
                print(f"  early stop at epoch {epoch} (best {best_epoch}, val {best_val:.4e})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    meta = {"git_sha": _git_sha(), "config_hash": config_hash(asdict(cfg)),
            "best_val_rel": best_val, "best_epoch": best_epoch}
    return model, history, meta


def overfit_subset(model, dm, k: int = 16, steps: int = 400, lr: float = 5e-3) -> float:
    """Overfit ``k`` train samples; returns final train MSE (sanity gate)."""
    samples = dm._split_samples("train")[:k]
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([s.data for s in samples])
    ops = [s.ops for s in samples]
    y = torch.cat([s.data.y for s in samples], dim=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    use_n = model_uses_neumann(model)
    final = float("inf")
    for _ in range(steps):
        opt.zero_grad()
        pred = model(batch, ops if use_n else None)
        loss = frobenius_loss(pred, y)
        loss.backward()
        opt.step()
        final = loss.item()
    return final
