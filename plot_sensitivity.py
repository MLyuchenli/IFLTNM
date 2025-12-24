# -*- coding: utf-8 -*-
# plot_sensitivity.py — ω/ξ 网格敏感性可视化（3D Surface / 2D Heatmap / 3D Bar / 2D Grouped Bar）
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def set_pub_style(font="Times New Roman", fontsize=12):
    plt.rcParams.update({
        "font.family": font,
        "mathtext.fontset": "stix",      # 数学字体接近 Times
        "axes.titlesize": fontsize+2,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize-1,
        "ytick.labelsize": fontsize-1,
        "legend.fontsize": fontsize-1,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

def _ensure_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """兼容 grid_*_valacc_matrix.csv（已是矩阵）与 grid_*.csv（明细，需要 pivot）"""
    cols = {c.lower(): c for c in df.columns}
    if "val_acc_all" in cols:  # 明细表 -> 透视
        omega_col = cols.get("omega", "omega")
        zeta_col  = cols.get("zeta",  "zeta")
        val_col   = cols["val_acc_all"]
        mat = df.pivot(index=omega_col, columns=zeta_col, values=val_col)
    else:  # 已是矩阵
        if df.columns[0].lower() in ("omega", "ω", "w", "index"):
            df = df.set_index(df.columns[0])
        mat = df
    mat = mat.sort_index().sort_index(axis=1)
    mat.index = mat.index.astype(float)
    mat.columns = mat.columns.astype(float)
    return mat.astype(float)

# ---------------- 原有两种：3D 曲面 / 2D 热力 ----------------
def plot_surface(mat: pd.DataFrame, out_png: Path, cmap="viridis", elev=28, azim=-135, zlim=None):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    O, Z = mat.index.values, mat.columns.values
    OO, ZZ = np.meshgrid(O, Z, indexing="ij")
    VV = mat.to_numpy()
    fig = plt.figure(figsize=(6.2, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(OO, ZZ, VV, cmap=cmap, linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel(r'$\omega_1$'); ax.set_ylabel(r'$\xi$'); ax.set_zlabel('Accuracy')
    ax.view_init(elev=elev, azim=azim)
    if zlim is not None: ax.set_zlim(*zlim)
    cbar = fig.colorbar(surf, shrink=0.75, pad=0.08); cbar.set_label('Accuracy')
    zmin = np.nanmin(VV)
    ax.contour(OO, ZZ, VV, zdir='z', offset=zmin, cmap=cmap, linewidths=0.8)
    fig.savefig(out_png, dpi=320); plt.close(fig)

def plot_heatmap(mat: pd.DataFrame, out_png: Path, cmap="viridis", annotate=True):
    O, Z = mat.index.values, mat.columns.values
    VV = mat.to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    im = ax.imshow(VV, origin="lower", cmap=cmap,
                   extent=[Z.min(), Z.max(), O.min(), O.max()],
                   aspect="auto", interpolation="nearest")
    ax.set_xlabel(r'$\xi$'); ax.set_ylabel(r'$\omega_1$')
    cbar = plt.colorbar(im, ax=ax); cbar.set_label('Accuracy')
    ax.set_xticks(Z); ax.set_yticks(O); ax.tick_params(axis='x', rotation=0)
    if annotate:
        norm = plt.Normalize(vmin=np.nanmin(VV), vmax=np.nanmax(VV))
        for i, o in enumerate(O):
            for j, z in enumerate(Z):
                v = VV[i, j]
                color = "white" if norm(v) > 0.55 else "black"
                ax.text(z, o, f"{v:.2f}", ha="center", va="center", fontsize=9, color=color)
    fig.savefig(out_png, dpi=320); plt.close(fig)

# ---------------- 新增：3D 柱状图 ----------------
def plot_bar3d(mat: pd.DataFrame, out_png: Path, cmap="viridis"):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    O, Z = mat.index.values, mat.columns.values
    VV = mat.to_numpy()

    # 网格转平面坐标
    OO, ZZ = np.meshgrid(O, Z, indexing="ij")
    x = OO.ravel(); y = ZZ.ravel(); z = np.zeros_like(x)
    dz = VV.ravel()

    # 估计柱宽（按网格间距的 80%）
    def _spacing(vals):
        if len(vals) < 2: return 1.0
        diffs = np.diff(np.sort(vals))
        return float(np.median(diffs))
    dx = _spacing(O) * 0.8
    dy = _spacing(Z) * 0.8

    # 颜色映射
    norm = plt.Normalize(vmin=np.nanmin(VV), vmax=np.nanmax(VV))
    cmap_ = plt.get_cmap(cmap)
    colors = cmap_(norm(dz))

    fig = plt.figure(figsize=(7.0, 5.6))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=colors, edgecolor='black', linewidth=0.3)
    ax.set_xlabel(r'$\omega_1$'); ax.set_ylabel(r'$\xi$'); ax.set_zlabel('Accuracy')
    ax.view_init(elev=28, azim=-135)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_), ax=ax, shrink=0.75, pad=0.08)
    cbar.set_label('Accuracy')

    # 适当稀疏刻度（避免过密）
    def _tick_subset(vals, max_ticks=8):
        vals = np.unique(np.round(vals, 4))
        if len(vals) <= max_ticks: return vals
        step = max(1, int(np.ceil(len(vals)/max_ticks)))
        return vals[::step]
    ax.set_xticks(_tick_subset(O))
    ax.set_yticks(_tick_subset(Z))

    fig.savefig(out_png, dpi=320); plt.close(fig)

# ---------------- 新增：2D 分组柱状图 ----------------
def plot_bar2d_grouped(mat: pd.DataFrame, out_png: Path):
    O, Z = mat.index.values, mat.columns.values  # 行=ω1, 列=ξ
    VV = mat.to_numpy()
    n_o, n_z = len(O), len(Z)
    x = np.arange(n_o)

    # 每组内的柱宽/间距
    total_width = 0.82
    bw = total_width / n_z
    offsets = (np.arange(n_z) - (n_z-1)/2.0) * bw

    # 颜色方案：从 viridis 中等距采样，保证可区分
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, n_z))

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    handles = []
    for j, (zeta, color) in enumerate(zip(Z, colors)):
        yj = VV[:, j]
        h = ax.bar(x + offsets[j], yj, width=bw*0.95, color=color, edgecolor='black', linewidth=0.3,
                   label=rf'$\xi={zeta:.2f}$')
        # 数值标注（两位小数），高度较小时换白字
        for rect, v in zip(h, yj):
            if np.isnan(v): continue
            ax.annotate(f"{v:.2f}",
                        xy=(rect.get_x() + rect.get_width()/2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, color="black" if v < 0.55 else "white")
        handles.append(h)

    # 轴/刻度/标签
    ax.set_xlabel(r'$\omega_1$')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in O])
    ax.set_ylim(bottom=0.0, top=min(1.0, np.nanmax(VV)*1.08))
    ax.legend(ncol=min(n_z, 6), frameon=False, title=r'$\xi$')

    fig.savefig(out_png, dpi=320); plt.close(fig)

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser("Plot ω/ξ sensitivity (surface / heatmap / bar3d / bar2d)")
    ap.add_argument("--table", type=str, required=True,
                    help="grid_*_valacc_matrix.csv 或 grid_*.csv（含 omega/zeta/val_acc_all）")
    ap.add_argument("--out_prefix", type=str, default=None)
    ap.add_argument("--font", type=str, default="Times New Roman")
    ap.add_argument("--cmap", type=str, default="viridis")
    # 老图
    ap.add_argument("--no_surface", action="store_true")
    ap.add_argument("--no_heatmap", action="store_true")
    # 新图
    ap.add_argument("--bar3d", action="store_true", help="绘制 3D 柱状图")
    ap.add_argument("--bar2d", action="store_true", help="绘制 2D 分组柱状图")
    args = ap.parse_args()

    set_pub_style(font=args.font)
    path = Path(args.table)
    df = pd.read_csv(path)
    mat = _ensure_matrix(df)

    prefix = args.out_prefix or path.stem.replace("_valacc_matrix", "")

    # 3D 曲面 / 2D 热力（保持原功能）
    if not args.no_surface:
        plot_surface(mat, Path(f"{prefix}_surface.png"), cmap=args.cmap)
    if not args.no_heatmap:
        plot_heatmap(mat, Path(f"{prefix}_heatmap.png"), cmap=args.cmap)

    # 新的柱状图
    if args.bar3d:
        plot_bar3d(mat, Path(f"{prefix}_bar3d.png"), cmap=args.cmap)
    if args.bar2d:
        plot_bar2d_grouped(mat, Path(f"{prefix}_bar2d.png"))

if __name__ == "__main__":
    main()
