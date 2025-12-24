import os
import json
import time
import argparse
import importlib
import traceback
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8

# ============================================================
# Helpers
# ============================================================

def safe_torch_save(obj, path: str, use_legacy_zip: bool = True) -> None:
    """Write a torch object safely on Windows.

    Writes to <path>.tmp then atomically replaces.
    Optionally uses legacy (non-zip) serialization.
    """
    path = os.fspath(path)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    if use_legacy_zip:
        torch.save(obj, tmp, _use_new_zipfile_serialization=False)
    else:
        torch.save(obj, tmp)
    os.replace(tmp, path)


def make_run_id(csv_path: str, seed: int, n_rules: int, sigma: float,
                exp_name: str = "", use_time_tag: bool = False) -> str:
    stem = Path(csv_path).stem
    base = exp_name.strip() if exp_name and exp_name.strip() else stem
    rid = f"{base}_seed{seed}_R{n_rules}_sig{sigma}"
    if use_time_tag:
        rid += "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    return rid


def build_paths(out_dir: str, run_id: str) -> Tuple[str, str, str]:
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, f"split_manifest_{run_id}.pt")
    teacher_path  = os.path.join(out_dir, f"ifltnm_teacher_{run_id}.pt")
    student_path  = os.path.join(out_dir, f"s_ifltnm_student_{run_id}.pt")
    return manifest_path, teacher_path, student_path


def _import_tabnet(module_name: Optional[str], class_name: str = "TabNet"):
    """Import TabNet from a user-provided module (default: tabnet_core.TabNet)."""
    tried = []
    candidates = [module_name] if module_name else []
    candidates += ["tabnet_core", "tabnet_CAD6usethis", "tab_network"]
    for mod in candidates:
        try:
            m = importlib.import_module(mod)
        except Exception as e:
            tried.append(f"[module fail] {mod}: {e}")
            continue
        for nm in [class_name, "TabNet", "TabNetNoEmbeddings", "TabNetModel", "TabNetEncoder"]:
            if hasattr(m, nm):
                print(f"[Import] Using {mod}.{nm}", flush=True)
                return getattr(m, nm)
        tried.append(f"[class miss] {mod} has none of {[class_name,'TabNet','TabNetNoEmbeddings','TabNetModel','TabNetEncoder']}")
    raise ImportError("Cannot import TabNet. Tried:\n" + "\n".join(tried))


def _get_logits(rule: nn.Module, X: torch.Tensor) -> torch.Tensor:
    out = rule(X)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def encode_labels(y: np.ndarray) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    cls, y_idx = np.unique(y, return_inverse=True)
    label2id = {str(c): int(i) for i, c in enumerate(cls)}
    id2label = {int(i): str(c) for i, c in enumerate(cls)}
    return y_idx.astype(np.int64), label2id, id2label


def minmax_scale(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mn = X.min(axis=0, keepdims=True)
    mx = X.max(axis=0, keepdims=True)
    Xs = (X - mn) / (mx - mn + 1e-12)
    return Xs.astype(np.float32), mn.squeeze(0).astype(np.float32), mx.squeeze(0).astype(np.float32)


def load_csv(csv_path: str,
             sep: str = ",",
             label_col: Optional[str] = None,
             label_idx: int = -1,
             no_header: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, int], Dict[int, str]]:
    header = None if no_header else "infer"
    df = pd.read_csv(csv_path, sep=sep, header=header)

    if label_col is not None:
        if no_header:
            raise ValueError("--label (column name) cannot be used with --no_header")
        y_raw = df[label_col].to_numpy()
        X_df = df.drop(columns=[label_col])
    else:
        y_raw = df.iloc[:, label_idx].to_numpy()
        X_df = df.drop(df.columns[label_idx], axis=1)

    # feature names
    if no_header:
        feature_names = [f"x{j+1}" for j in range(X_df.shape[1])]
    else:
        feature_names = [str(c) for c in X_df.columns]

    # make numeric (best-effort)
    X = X_df.to_numpy()
    if not np.issubdtype(X.dtype, np.number):
        X = X_df.apply(pd.to_numeric, errors="coerce").to_numpy()
    if np.isnan(X).any():
        # Simple imputation for open-source baseline
        col_means = np.nanmean(X, axis=0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_means, idx[1])

    y_idx, label2id, id2label = encode_labels(y_raw)
    return X.astype(np.float32), y_idx, feature_names, label2id, id2label


def apply_symmetric_label_noise(y: np.ndarray, noise_rate: float, n_classes: int, seed: int) -> np.ndarray:
    """Flip labels with probability noise_rate to a *different* class."""
    if noise_rate <= 0:
        return y
    rng = np.random.default_rng(seed)
    y = y.copy()
    mask = rng.random(len(y)) < noise_rate
    if mask.any():
        for i in np.where(mask)[0]:
            cur = y[i]
            # sample a different class
            cand = rng.integers(0, n_classes - 1)
            y[i] = cand if cand < cur else cand + 1
    return y

# ============================================================
# Antecedents: Gaussian memberships
# ============================================================

class GaussianAntecedent(nn.Module):
    """Product-of-Gaussians antecedent.

    memberships(X) -> normalized firing strengths w_{i,k}.
    """

    def __init__(self, centers: np.ndarray, sigma: float | np.ndarray):
        super().__init__()
        centers = np.asarray(centers, dtype=np.float32)
        self.R, self.D = centers.shape
        self.register_buffer("centers", torch.from_numpy(centers))

        sig_np = np.asarray(sigma, dtype=np.float32)
        if np.isscalar(sigma) or sig_np.ndim == 0:
            sigma_arr = np.full_like(centers, float(sig_np), dtype=np.float32)
        else:
            assert sig_np.shape == centers.shape, "sigma must be scalar or same shape as centers"
            sigma_arr = sig_np.astype(np.float32)
        self.register_buffer("sigma", torch.from_numpy(sigma_arr))

    def memberships(self, X: torch.Tensor) -> torch.Tensor:
        diff = (X.unsqueeze(1) - self.centers.unsqueeze(0))
        z = diff / (self.sigma.unsqueeze(0) + _EPS)
        g = torch.exp(-0.5 * z.pow(2)).clamp_min(_EPS)
        mem = g.prod(dim=2)
        return mem / (mem.sum(dim=1, keepdim=True) + _EPS)


# ============================================================
# Consequent Additive Decomposition (CAD)
# ============================================================

class LinearDecomposition(nn.Module):
    """Replace an nn.Linear with two additive parts: clean + noisy.

    During training (use_noisy=True):  W = W_c + W_n
    For final teacher inference (use_noisy=False): W = W_c
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight_c = nn.Parameter(linear.weight.detach().clone())
        self.weight_n = nn.Parameter(torch.zeros_like(linear.weight))
        if linear.bias is not None:
            self.bias_c = nn.Parameter(linear.bias.detach().clone())
            self.bias_n = nn.Parameter(torch.zeros_like(linear.bias))
        else:
            self.bias_c = None
            self.bias_n = None
        self.use_noisy: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_noisy:
            w = self.weight_c + self.weight_n
            b = (self.bias_c + self.bias_n) if self.bias_c is not None else None
        else:
            w = self.weight_c
            b = self.bias_c
        return F.linear(x, w, b)


def wrap_with_decomposition(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LinearDecomposition(child))
        else:
            wrap_with_decomposition(child)
    return module


def set_use_noisy(module: nn.Module, use_noisy: bool) -> None:
    for m in module.modules():
        if isinstance(m, LinearDecomposition):
            m.use_noisy = bool(use_noisy)


def collect_decomp_params(module: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Return lists of (clean_params, noisy_params) in LinearDecomposition layers."""
    clean_params: List[nn.Parameter] = []
    noisy_params: List[nn.Parameter] = []
    for m in module.modules():
        if isinstance(m, LinearDecomposition):
            clean_params.append(m.weight_c)
            noisy_params.append(m.weight_n)
            if m.bias_c is not None:
                clean_params.append(m.bias_c)
                noisy_params.append(m.bias_n)
    return clean_params, noisy_params


# ============================================================
# Teacher wrapper
# ============================================================

class TeacherIFLTNM(nn.Module):
    def __init__(self, rules: List[nn.Module], antecedent: GaussianAntecedent):
        super().__init__()
        self.rules = nn.ModuleList(rules)
        self.ante = antecedent
        self._C: Optional[int] = None

    def infer_num_classes(self, X: torch.Tensor) -> int:
        with torch.no_grad():
            self._C = int(_get_logits(self.rules[0], X[:1]).shape[1])
        return self._C

    @property
    def C(self) -> int:
        if self._C is None:
            raise RuntimeError("call infer_num_classes() first")
        return self._C

    def forward_rule_logits(self, X: torch.Tensor) -> torch.Tensor:
        return torch.stack([_get_logits(r, X) for r in self.rules], dim=1)  # (B,R,C)

    def forward_final_logits(self, X: torch.Tensor) -> torch.Tensor:
        w = self.ante.memberships(X).unsqueeze(2)  # (B,R,1)
        z = self.forward_rule_logits(X)            # (B,R,C)
        return (w * z).sum(dim=1)

    def forward_final_probs(self, X: torch.Tensor) -> torch.Tensor:
        w = self.ante.memberships(X).unsqueeze(2)
        z = self.forward_rule_logits(X)
        return (w * torch.softmax(z, dim=2)).sum(dim=1)

    @torch.no_grad()
    def partition_correct(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split indices by whether teacher prediction matches observed label."""
        yhat = self.forward_final_probs(X).argmax(1)
        m = (yhat == y)
        clean_idx = torch.nonzero(m, as_tuple=False).view(-1)
        noisy_idx = torch.nonzero(~m, as_tuple=False).view(-1)
        return clean_idx, noisy_idx

# ============================================================
# Teacher training (CAD)
# ============================================================

@dataclass
class TeacherConfig:
    epochs: int = 100
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    clip_grad_norm: float = 5.0
    # CAD schedule hyper-params
    pF: float = 1e-2   # coefficient scale for beta_F (clean stability)
    pY: float = 1e-2   # coefficient scale for beta_Y (noisy suppression)


def beta_F_linear(ep: int, E: int, pF: float) -> float:
    # linearly increasing
    return pF * (ep / max(E, 1))


def beta_Y_exp(ep: int, E: int, pY: float) -> float:
    # exponentially decreasing
    t = ep / max(E, 1)
    return pY * float(np.exp(-t))


def build_teacher(D: int, C: int, R: int, TabNet, tabnet_params: Dict, device: str) -> List[nn.Module]:
    rules: List[nn.Module] = []
    for _ in range(R):
        m = TabNet(input_dim=D, output_dim=C, **tabnet_params).to(device)
        wrap_with_decomposition(m)
        rules.append(m)
    return rules


def train_ifltnm_teacher(
    X: torch.Tensor,
    y: torch.Tensor,
    teacher: TeacherIFLTNM,
    cfg: TeacherConfig,
) -> None:
    """Train teacher with CAD objective.

    Loss = CE(y, teacher(x; Wc+Wn))
         + beta_F(ep) * sum ||Wc - Wc_prev||^2
         + beta_Y(ep) * sum ||Wn||^2

    - beta_F increases linearly over epochs.
    - beta_Y decreases exponentially over epochs.
    """
    device = X.device
    rules = list(teacher.rules)

    # collect all params
    params = [p for r in rules for p in r.parameters()]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # previous-epoch clean params
    clean_prev: List[torch.Tensor] = []
    clean_params_all: List[nn.Parameter] = []
    noisy_params_all: List[nn.Parameter] = []
    for r in rules:
        c, n = collect_decomp_params(r)
        clean_params_all += c
        noisy_params_all += n
    clean_prev = [p.detach().clone() for p in clean_params_all]

    N = X.size(0)
    n_batches = (N + cfg.batch_size - 1) // cfg.batch_size

    print(f"[Teacher] Start training (CAD): epochs={cfg.epochs}, batch_size={cfg.batch_size}, batches/epoch={n_batches}", flush=True)

    for ep in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(N, device=device)
        bF = beta_F_linear(ep, cfg.epochs, cfg.pF)
        bY = beta_Y_exp(ep, cfg.epochs, cfg.pY)

        running = 0.0
        for bi in range(n_batches):
            idx = perm[bi * cfg.batch_size : (bi + 1) * cfg.batch_size]
            Xb = X[idx]
            yb = y[idx]

            # teacher forward (uses noisy + clean by default)
            probs = teacher.forward_final_probs(Xb)
            ce = F.nll_loss(torch.log(probs + _EPS), yb)

            reg_clean = 0.0
            if ep > 1 and bF > 0:
                for p, p_prev in zip(clean_params_all, clean_prev):
                    reg_clean = reg_clean + (p - p_prev).pow(2).sum()
                reg_clean = bF * reg_clean

            reg_noisy = 0.0
            if bY > 0:
                for p in noisy_params_all:
                    reg_noisy = reg_noisy + p.pow(2).sum()
                reg_noisy = bY * reg_noisy

            loss = ce + reg_clean + reg_noisy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.clip_grad_norm is not None and cfg.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(params, cfg.clip_grad_norm)
            opt.step()

            running += float(loss.item())

        # update previous clean params
        clean_prev = [p.detach().clone() for p in clean_params_all]
        dt = time.perf_counter() - t0
        print(f"[Teacher] epoch {ep:03d}/{cfg.epochs}  loss={running/max(n_batches,1):.4f}  betaF={bF:.2e} betaY={bY:.2e}  ({dt:.1f}s)", flush=True)


@torch.no_grad()
def evaluate_teacher(teacher: TeacherIFLTNM, X: torch.Tensor, y: torch.Tensor, batch_size: int = 2048) -> float:
    preds = []
    for i in range(0, X.size(0), batch_size):
        probs = teacher.forward_final_probs(X[i : i + batch_size])
        preds.append(probs.argmax(1).cpu())
    yhat = torch.cat(preds)
    return float((yhat == y.cpu()).float().mean().item())


def save_teacher(path: str, teacher: TeacherIFLTNM, tabnet_params: Dict) -> None:
    ckpt = {
        "rule_centers": teacher.ante.centers.detach().cpu(),
        "sigma": teacher.ante.sigma.detach().cpu(),
        "rules": [r.state_dict() for r in teacher.rules],
        "tabnet_params": tabnet_params,
        "use_decomposition": True,
    }
    safe_torch_save(ckpt, path)


def load_teacher(path: str, D: int, C: int, TabNet, device: str = "cpu") -> Tuple[TeacherIFLTNM, Dict]:
    ckpt = torch.load(path, map_location="cpu")
    tabnet_params = ckpt.get("tabnet_params", {})

    rules: List[nn.Module] = []
    for sd in ckpt["rules"]:
        m = TabNet(input_dim=D, output_dim=C, **tabnet_params).to(device)
        if ckpt.get("use_decomposition", False):
            wrap_with_decomposition(m)
        m.load_state_dict(sd)
        m.eval()
        rules.append(m)

    centers = ckpt["rule_centers"].detach().cpu().numpy().astype(np.float32)
    sigma_t = ckpt["sigma"]
    if isinstance(sigma_t, torch.Tensor) and sigma_t.dim() == 0:
        sigma = float(sigma_t.item())
    else:
        sigma = sigma_t.detach().cpu().numpy().astype(np.float32)

    ante = GaussianAntecedent(centers, sigma).to(device)
    teacher = TeacherIFLTNM(rules, ante).to(device)
    teacher._C = C
    return teacher, tabnet_params

# ============================================================
# Student: first-order TSK + 4-loss distillation
# ============================================================

@dataclass
class KDConfig:
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    clip_grad_norm: float = 5.0

    # KD hyper-params
    temperature: float = 4.0
    alpha_kd: float = 1.0      # weight for final-output KD (Eq. 16)
    beta_rule: float = 1.0     # weight for rule-wise KD (Eq. 17)

    # ANL hyper-params on almost-noisy subset (Eq. 18)
    xi: float = 0.1            # trade-off between y and y_bar

    # balancing between clean block (Eq. 15-17) and noisy regularizer (Eq. 18)
    omega: float = 0.2
    zeta: Optional[float] = None   # if None: zeta = 1 - omega


class StudentFirstOrderTSK(nn.Module):
    """First-order TSK student with shared antecedent.

    Rule k: f_k(x) = b_k + W_k x
    Aggregation: sum_k w_k(x) softmax(f_k(x)).

    Shapes:
      W: (R, C, D)
      b: (R, C)
    """

    def __init__(self, D: int, C: int, R: int, antecedent: GaussianAntecedent):
        super().__init__()
        self.D, self.C, self.R = D, C, R
        self.ante = antecedent
        self.W = nn.Parameter(torch.zeros(R, C, D))
        self.b = nn.Parameter(torch.zeros(R, C))
        nn.init.normal_(self.W, mean=0.0, std=0.02)
        nn.init.zeros_(self.b)

    def forward_rule_logits(self, X: torch.Tensor) -> torch.Tensor:
        # (B,R,C) = einsum('bd,rcd->brc') + b
        return torch.einsum('bd,rcd->brc', X, self.W) + self.b.unsqueeze(0)

    def forward_final_logits(self, X: torch.Tensor) -> torch.Tensor:
        w = self.ante.memberships(X).unsqueeze(2)
        z = self.forward_rule_logits(X)
        return (w * z).sum(dim=1)

    def forward_final_probs(self, X: torch.Tensor) -> torch.Tensor:
        w = self.ante.memberships(X).unsqueeze(2)
        z = self.forward_rule_logits(X)
        return (w * torch.softmax(z, dim=2)).sum(dim=1)


def kd_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    """Standard KD: T^2 * KL( softmax(z_t/T) || softmax(z_s/T) )."""
    p_t = torch.softmax(teacher_logits / T, dim=1)
    log_p_s = torch.log_softmax(student_logits / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction='batchmean') * (T * T)


def rule_l1_kd(student_rule_logits: torch.Tensor, teacher_rule_logits: torch.Tensor, T: float) -> torch.Tensor:
    """Rule-wise KD via L1 distance between softened rule distributions."""
    ps = torch.softmax(student_rule_logits / T, dim=2)
    pt = torch.softmax(teacher_rule_logits / T, dim=2)
    return torch.mean(torch.abs(ps - pt))


def anl_loss(student_probs: torch.Tensor, y: torch.Tensor, xi: float) -> torch.Tensor:
    """ANL regularizer: (1-xi) * CE(y) + xi * CE(y_bar).

    y_bar is uniform over classes except the observed class.
    """
    C = int(student_probs.shape[1])
    ce = F.nll_loss(torch.log(student_probs + _EPS), y)

    # uniform on all other classes
    onehot = F.one_hot(y, num_classes=C).float()
    ybar = (1.0 - onehot) / max(C - 1, 1)
    ce_bar = -(ybar * torch.log(student_probs + _EPS)).sum(dim=1).mean()
    return (1.0 - xi) * ce + xi * ce_bar

# ============================================================
# Student distillation training
# ============================================================

@torch.no_grad()
def evaluate_student(student: StudentFirstOrderTSK, X: torch.Tensor, y: torch.Tensor, batch_size: int = 2048) -> float:
    preds = []
    for i in range(0, X.size(0), batch_size):
        probs = student.forward_final_probs(X[i:i+batch_size])
        preds.append(probs.argmax(1).cpu())
    yhat = torch.cat(preds)
    return float((yhat == y.cpu()).float().mean().item())


def distill_student(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    clean_idx: torch.Tensor,
    noisy_idx: torch.Tensor,
    teacher_clean: TeacherIFLTNM,
    student: StudentFirstOrderTSK,
    cfg: KDConfig,
) -> None:
    """Train s-IFLTNM with 4 losses.

    Clean subset (D_clean):
      (Eq.15) CE(y, p_s)
      (Eq.16) KL distillation on final logits
      (Eq.17) L1 distillation on per-rule logits

    Noisy subset (D_noisy):
      (Eq.18) ANL regularizer using complementary distribution

    Final combination (a practical core version):
      L = zeta*(L15+alpha*L16+beta*L17) + omega*L18
      where zeta defaults to 1-omega if not provided.
    """

    device = X_train.device
    opt = torch.optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    omega = float(cfg.omega)
    zeta = float(cfg.zeta) if cfg.zeta is not None else float(max(0.0, 1.0 - omega))

    Xc = X_train[clean_idx]
    yc = y_train[clean_idx]
    Xn = X_train[noisy_idx]
    yn = y_train[noisy_idx]

    Nc = Xc.size(0)
    Nn = Xn.size(0)
    nb_c = (Nc + cfg.batch_size - 1) // cfg.batch_size
    nb_n = (Nn + cfg.batch_size - 1) // cfg.batch_size if Nn > 0 else 0

    print(f"[Student] Start distillation: epochs={cfg.epochs}, clean={Nc}, noisy={Nn}, omega={omega}, zeta={zeta}", flush=True)

    for ep in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()

        # shuffle indices each epoch
        perm_c = torch.randperm(Nc, device=device)
        perm_n = torch.randperm(Nn, device=device) if Nn > 0 else None

        running = {"ce": 0.0, "kd": 0.0, "rk": 0.0, "anl": 0.0, "tot": 0.0}
        steps = 0

        # iterate over clean batches (primary)
        for bi in range(nb_c):
            idx = perm_c[bi * cfg.batch_size : (bi + 1) * cfg.batch_size]
            Xb = Xc[idx]
            yb = yc[idx]

            # teacher targets (clean-only teacher)
            with torch.no_grad():
                t_rule = teacher_clean.forward_rule_logits(Xb)  # (B,R,C)
                t_final = teacher_clean.forward_final_logits(Xb)  # (B,C)

            s_rule = student.forward_rule_logits(Xb)
            s_final = student.forward_final_logits(Xb)
            s_prob = student.forward_final_probs(Xb)

            L_ce = F.nll_loss(torch.log(s_prob + _EPS), yb)
            L_kd = kd_kl(s_final, t_final, cfg.temperature)
            L_rk = rule_l1_kd(s_rule, t_rule, cfg.temperature)

            loss_clean = L_ce + cfg.alpha_kd * L_kd + cfg.beta_rule * L_rk

            # optionally add a noisy batch for ANL this step
            if nb_n > 0:
                j = bi % nb_n
                idxn = perm_n[j * cfg.batch_size : (j + 1) * cfg.batch_size]
                Xnb = Xn[idxn]
                ynb = yn[idxn]
                s_prob_n = student.forward_final_probs(Xnb)
                L_anl = anl_loss(s_prob_n, ynb, cfg.xi)
            else:
                L_anl = torch.tensor(0.0, device=device)

            loss = zeta * loss_clean + omega * L_anl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.clip_grad_norm is not None and cfg.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(student.parameters(), cfg.clip_grad_norm)
            opt.step()

            running["ce"] += float(L_ce.item())
            running["kd"] += float(L_kd.item())
            running["rk"] += float(L_rk.item())
            running["anl"] += float(L_anl.item())
            running["tot"] += float(loss.item())
            steps += 1

        dt = time.perf_counter() - t0
        print(
            f"[Student] epoch {ep:03d}/{cfg.epochs}  "
            f"tot={running['tot']/max(steps,1):.4f}  "
            f"ce={running['ce']/max(steps,1):.4f}  kd={running['kd']/max(steps,1):.4f}  rk={running['rk']/max(steps,1):.4f}  anl={running['anl']/max(steps,1):.4f}  "
            f"({dt:.1f}s)",
            flush=True,
        )

# ============================================================
# Checkpointing
# ============================================================

def save_student(path: str, student: StudentFirstOrderTSK, meta: Dict) -> None:
    ckpt = {
        "W": student.W.detach().cpu(),
        "b": student.b.detach().cpu(),
        "rule_centers": student.ante.centers.detach().cpu(),
        "sigma": student.ante.sigma.detach().cpu(),
        "meta": meta,
    }
    safe_torch_save(ckpt, path)


def save_manifest(path: str, manifest: Dict) -> None:
    safe_torch_save(manifest, path)


# ============================================================
# Pipeline
# ============================================================

def make_run_id(csv_path: str, seed: int, R: int, sigma: float, noise_rate: float) -> str:
    stem = Path(csv_path).stem
    return f"{stem}_seed{seed}_R{R}_sig{sigma}_nr{noise_rate}"


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_pipeline(
    csv_path: str,
    label_idx: int,
    seed: int,
    test_size: float,
    n_rules: int,
    sigma: float,
    noise_rate: float,
    out_dir: str,
    tabnet_module: str,
    tabnet_class: str,
    tabnet_params: Dict,
    teacher_cfg: TeacherConfig,
    kd_cfg: KDConfig,
    no_train_teacher: bool,
    teacher_ckpt: Optional[str],
    do_distill: bool,
    header: bool,
    sep: Optional[str],
    device: str,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = str(Path(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    X, y, feature_names, label2id, id2label = load_csv(csv_path, label_idx=label_idx, sep=sep, header=header)

    # split
    from sklearn.model_selection import train_test_split
    idx_all = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx_all, test_size=test_size, random_state=seed, stratify=y)

    Xtr = X[tr_idx]
    ytr = y[tr_idx]
    Xte = X[te_idx]
    yte = y[te_idx]

    # optionally inject symmetric noise into TRAIN labels only
    if noise_rate and noise_rate > 0:
        ytr_noisy = apply_symmetric_label_noise(ytr, noise_rate=noise_rate, seed=seed, n_classes=int(y.max() + 1))
        ytr = ytr_noisy

    # scale using train min/max
    Xtr_s, mn, mx = minmax_scale(Xtr)
    Xte_s = (Xte - mn) / (mx - mn + _EPS)
    Xte_s = Xte_s.astype(np.float32)

    D = Xtr_s.shape[1]
    C = int(max(y.max(), yte.max()) + 1)

    # rule centers: sample from {0,0.25,...,1}
    rng = np.random.default_rng(seed)
    grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    centers = rng.choice(grid, size=(n_rules, D), replace=True).astype(np.float32)

    device = pick_device(device)

    Xtr_t = torch.from_numpy(Xtr_s).to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    Xte_t = torch.from_numpy(Xte_s).to(device)
    yte_t = torch.from_numpy(yte).long().to(device)

    ante = GaussianAntecedent(centers, sigma).to(device)

    # import TabNet
    TabNet = _import_tabnet(tabnet_module, tabnet_class)

    # build / load teacher
    run_id = make_run_id(csv_path, seed, n_rules, sigma, noise_rate)
    teacher_out = str(Path(out_dir) / f"ifltnm_teacher_{run_id}.pt")
    manifest_out = str(Path(out_dir) / f"split_manifest_{run_id}.pt")

    if teacher_ckpt is None:
        teacher_ckpt = teacher_out

    if no_train_teacher:
        if not Path(teacher_ckpt).exists():
            raise FileNotFoundError(f"--no_train_teacher given, but teacher_ckpt not found: {teacher_ckpt}")
        print(f"[Teacher] Loading from {teacher_ckpt}", flush=True)
        teacher, _ = load_teacher(teacher_ckpt, D=D, C=C, TabNet=TabNet, device=device)
    else:
        print(f"[Teacher] Building: R={n_rules}, D={D}, C={C}, device={device}", flush=True)
        rules = build_teacher(D=D, C=C, R=n_rules, TabNet=TabNet, tabnet_params=tabnet_params, device=device)
        teacher = TeacherIFLTNM(rules, ante).to(device)
        teacher.infer_num_classes(Xtr_t[:1])
        set_use_noisy(teacher, True)
        train_ifltnm_teacher(Xtr_t, ytr_t, teacher, teacher_cfg)

        # evaluate with full (clean+noisy) teacher
        acc_t = evaluate_teacher(teacher, Xte_t, yte_t)
        print(f"[Teacher] Test Acc (full) = {acc_t:.4f}", flush=True)

        save_teacher(teacher_ckpt, teacher, tabnet_params)
        print(f"[Teacher] Saved to {teacher_ckpt}", flush=True)

    # switch to clean-only teacher for partition/distillation
    set_use_noisy(teacher, False)

    # partition train set
    clean_idx, noisy_idx = teacher.partition_correct(Xtr_t, ytr_t)
    print(f"[Split] almost-clean={int(clean_idx.numel())} almost-noisy={int(noisy_idx.numel())}", flush=True)

    manifest = {
        "train_idx": torch.as_tensor(tr_idx).long(),
        "test_idx": torch.as_tensor(te_idx).long(),
        "mn": torch.as_tensor(mn).float(),
        "mx": torch.as_tensor(mx).float(),
        "feature_names": feature_names,
        "label2id": label2id,
        "id2label": id2label,
        "seed": seed,
        "n_rules": n_rules,
        "sigma": sigma,
        "noise_rate": noise_rate,
        "clean_local_idx": clean_idx.detach().cpu(),
        "noisy_local_idx": noisy_idx.detach().cpu(),
    }
    save_manifest(manifest_out, manifest)
    print(f"[Manifest] Saved to {manifest_out}", flush=True)

    if not do_distill:
        return

    # distill student
    student = StudentFirstOrderTSK(D=D, C=C, R=n_rules, antecedent=ante).to(device)
    distill_student(Xtr_t, ytr_t, clean_idx, noisy_idx, teacher, student, kd_cfg)

    acc_s = evaluate_student(student, Xte_t, yte_t)
    print(f"[Student] Test Acc = {acc_s:.4f}", flush=True)

    student_out = str(Path(out_dir) / f"s_ifltnm_student_{run_id}.pt")
    meta = {
        "csv": str(csv_path),
        "run_id": run_id,
        "tabnet_module": tabnet_module,
        "tabnet_class": tabnet_class,
        "tabnet_params": tabnet_params,
        "teacher_ckpt": str(teacher_ckpt),
        "manifest": str(manifest_out),
        "kd_cfg": kd_cfg.__dict__,
        "teacher_cfg": teacher_cfg.__dict__,
    }
    save_student(student_out, student, meta)
    print(f"[Student] Saved to {student_out}", flush=True)


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--sep", type=str, default=None, help="CSV separator. If None, pandas will infer.")
    ap.add_argument("--header", action="store_true", help="Use this if the CSV file has a header row.")
    ap.add_argument("--label_idx", type=int, default=-1, help="Label column index, default: last column.")

    # split / noise
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--noise_rate", type=float, default=0.0, help="Symmetric label noise rate applied to TRAIN labels only.")

    # model
    ap.add_argument("--n_rules", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=0.8)

    # TabNet import
    ap.add_argument("--tabnet_module", type=str, default="tabnet_core")
    ap.add_argument("--tabnet_class", type=str, default="TabNet")

    # runtime
    ap.add_argument("--out_dir", type=str, default="models")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # teacher control
    ap.add_argument("--no_train_teacher", action="store_true")
    ap.add_argument("--teacher_ckpt", type=str, default=None)
    ap.add_argument("--teacher_epochs", type=int, default=100)
    ap.add_argument("--teacher_batch", type=int, default=256)
    ap.add_argument("--teacher_lr", type=float, default=3e-4)
    ap.add_argument("--teacher_wd", type=float, default=1e-4)
    ap.add_argument("--pF", type=float, default=1e-2)
    ap.add_argument("--pY", type=float, default=1e-2)

    # student / distill
    ap.add_argument("--do_distill", action="store_true", help="Enable student distillation after teacher training/loading.")
    ap.add_argument("--student_epochs", type=int, default=200)
    ap.add_argument("--student_batch", type=int, default=256)
    ap.add_argument("--student_lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--alpha_kd", type=float, default=1.0)
    ap.add_argument("--beta_rule", type=float, default=1.0)
    ap.add_argument("--xi", type=float, default=0.1)
    ap.add_argument("--omega", type=float, default=0.2)
    ap.add_argument("--zeta", type=float, default=None)

    args = ap.parse_args()

    teacher_cfg = TeacherConfig(
        epochs=args.teacher_epochs,
        batch_size=args.teacher_batch,
        lr=args.teacher_lr,
        weight_decay=args.teacher_wd,
        pF=args.pF,
        pY=args.pY,
    )

    kd_cfg = KDConfig(
        epochs=args.student_epochs,
        batch_size=args.student_batch,
        lr=args.student_lr,
        temperature=args.temperature,
        alpha_kd=args.alpha_kd,
        beta_rule=args.beta_rule,
        xi=args.xi,
        omega=args.omega,
        zeta=args.zeta,
    )

    # expose TabNet params (core defaults). You can add more CLI knobs here if needed.
    tabnet_params = {
        "n_d": 8,
        "n_a": 8,
        "n_steps": 3,
        "gamma": 1.3,
        "n_independent": 2,
        "n_shared": 2,
        "virtual_batch_size": 128,
        "momentum": 0.02,
        "mask_type": "sparsemax",
    }

    try:
        run_pipeline(
            csv_path=args.csv,
            label_idx=args.label_idx,
            seed=args.seed,
            test_size=args.test_size,
            n_rules=args.n_rules,
            sigma=args.sigma,
            noise_rate=args.noise_rate,
            out_dir=args.out_dir,
            tabnet_module=args.tabnet_module,
            tabnet_class=args.tabnet_class,
            tabnet_params=tabnet_params,
            teacher_cfg=teacher_cfg,
            kd_cfg=kd_cfg,
            no_train_teacher=args.no_train_teacher,
            teacher_ckpt=args.teacher_ckpt,
            do_distill=args.do_distill,
            header=args.header,
            sep=args.sep,
            device=args.device,
        )
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
