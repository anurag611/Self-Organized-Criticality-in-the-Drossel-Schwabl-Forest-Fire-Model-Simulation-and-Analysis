#!/usr/bin/env python3
"""
Forest Fire Model: Self-Organized Criticality Simulation (v2)
==============================================================

Improved version with proper cluster-based avalanche measurement.
Each lightning strike initiates a fire that burns an entire connected
cluster of trees. The size of that cluster is the avalanche size.

This version uses a two-phase approach:
  Phase 1 (growth): Trees grow on empty cells with probability p.
  Phase 2 (fire):   A random tree is struck by lightning with probability f.
                    The fire then burns the entire connected cluster instantly.

Author : Generated with the assistance of Claude (Anthropic)
Date   : April 2026
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import deque
import os
import warnings
warnings.filterwarnings("ignore") 

EMPTY = 0
TREE  = 1

class ForestFireSOC:
    """
    Forest Fire model with instantaneous cluster burning.
    
    At each time step:
      1. Each empty cell grows a tree with probability p.
      2. With probability f, a random tree is struck by lightning.
         The entire connected cluster of trees burns instantly.
    """

    def __init__(self, L=100, p=0.01, f=1e-4, connectivity="von_neumann", seed=42):
        self.L = L
        self.p = p
        self.f = f
        self.connectivity = connectivity
        self.rng = np.random.default_rng(seed)
        self.grid = np.zeros((L, L), dtype=np.int8)
        
        if connectivity == "von_neumann":
            self.offsets = [(-1,0),(1,0),(0,-1),(0,1)]
        else:  # moore
            self.offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        
        self.fire_sizes = []
        self.tree_density = []
        self.fire_grids = []  # store grids during large fires for visualization

    def _burn_cluster(self, r0, c0):
        """BFS to burn the connected cluster starting at (r0, c0). Returns size."""
        L = self.L
        if self.grid[r0, c0] != TREE:
            return 0
        
        queue = deque([(r0, c0)])
        self.grid[r0, c0] = EMPTY
        size = 1
        
        while queue:
            r, c = queue.popleft()
            for dr, dc in self.offsets:
                nr, nc = (r + dr) % L, (c + dc) % L
                if self.grid[nr, nc] == TREE:
                    self.grid[nr, nc] = EMPTY
                    queue.append((nr, nc))
                    size += 1
        return size

    def step(self):
        """One time step: grow trees, then possibly strike lightning."""
        L = self.L
        
        # Growth phase
        empty = (self.grid == EMPTY)
        grow = empty & (self.rng.random((L, L)) < self.p)
        self.grid[grow] = TREE
        
        # Lightning phase: each tree can be struck
        tree_positions = np.argwhere(self.grid == TREE)
        if len(tree_positions) > 0:
            # Each tree ignites independently with probability f
            strikes = self.rng.random(len(tree_positions)) < self.f
            strike_indices = np.where(strikes)[0]
            
            for idx in strike_indices:
                r, c = tree_positions[idx]
                if self.grid[r, c] == TREE:  # might already be burned
                    size = self._burn_cluster(r, c)
                    if size > 0:
                        self.fire_sizes.append(size)
        
        self.tree_density.append(np.mean(self.grid == TREE))

    def run(self, n_steps=5000, verbose=True):
        for t in range(1, n_steps + 1):
            self.step()
            if verbose and t % 1000 == 0:
                print(f"  Step {t:>5d}/{n_steps} | density={self.tree_density[-1]:.3f} | fires={len(self.fire_sizes)}")
        return self


def log_binned_pdf(data, n_bins=30):
    """Compute log-binned probability density."""
    data = np.array(data, dtype=float)
    data = data[data > 0]
    if len(data) < 10:
        return None, None, None
    
    bins = np.logspace(np.log10(data.min()), np.log10(data.max()), n_bins + 1)
    counts, edges = np.histogram(data, bins=bins)
    widths = np.diff(edges)
    centres = np.sqrt(edges[:-1] * edges[1:])
    mask = counts > 0
    pdf = counts[mask] / (widths[mask] * len(data))
    return centres[mask], pdf, mask


def mle_power_law(data, s_min=1):
    """MLE estimate of power-law exponent for discrete data."""
    data = np.array(data, dtype=float)
    data = data[data >= s_min]
    if len(data) < 30:
        return None
    n = len(data)
    tau = 1.0 + n / np.sum(np.log(data / (s_min - 0.5)))
    return tau


def main():
    out = "figures"
    os.makedirs(out, exist_ok=True)

    print("=" * 65)
    print("  FOREST FIRE SOC MODEL — Cluster-Based Avalanche Measurement")
    print("=" * 65)

    results = {}

    for conn, color, marker in [("von_neumann", "#228B22", "o"), ("moore", "#1E90FF", "s")]:
        label = "Von Neumann (4)" if conn == "von_neumann" else "Moore (8)"
        print(f"\n{'─'*50}")
        print(f"  Running: {label} connectivity")
        print(f"{'─'*50}")

        model = ForestFireSOC(L=128, p=0.05, f=0.0002, connectivity=conn, seed=42)
        model.run(n_steps=8000, verbose=True)

        results[conn] = {
            "model": model,
            "color": color,
            "marker": marker,
            "label": label
        }

        sizes = np.array(model.fire_sizes)
        print(f"\n  Results for {label}:")
        print(f"    Fires recorded: {len(sizes)}")
        if len(sizes) > 0:
            print(f"    Mean size:  {sizes.mean():.1f}")
            print(f"    Max size:   {sizes.max()}")
            print(f"    Median:     {np.median(sizes):.1f}")
            for s_min in [1, 5, 10]:
                tau = mle_power_law(sizes, s_min=s_min)
                if tau:
                    print(f"    tau (s_min={s_min:>2d}): {tau:.3f}")

    # ── Plot 1: Grid snapshots ──────────────────────────────────────
    print("\n  Generating grid snapshots...")
    model_vis = ForestFireSOC(L=128, p=0.05, f=0.0002, connectivity="von_neumann", seed=99)
    snaps = {}
    for t in range(1, 6001):
        model_vis.step()
        if t in [200, 2000, 5000]:
            snaps[t] = model_vis.grid.copy()

    cmap = mcolors.ListedColormap(["#3d3226", "#228B22"])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (t, g) in zip(axes, snaps.items()):
        ax.imshow(g, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"t = {t}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle("Forest Fire Lattice Snapshots (Von Neumann)", fontsize=14)
    plt.tight_layout()
    fig.savefig(f"{out}/grid_snapshots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}/grid_snapshots.png")

    # ── Plot 2: Tree density time series ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, conn in zip(axes, ["von_neumann", "moore"]):
        r = results[conn]
        d = r["model"].tree_density
        ax.plot(d, color=r["color"], lw=0.4, alpha=0.8)
        mean_d = np.mean(d[len(d)//2:])
        ax.axhline(mean_d, color="red", ls="--", lw=1.5,
                    label=f"Steady state mean = {mean_d:.3f}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Tree density")
        ax.set_title(r["label"])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out}/tree_density.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}/tree_density.png")

    # ── Plot 3: Avalanche size distribution ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for conn in ["von_neumann", "moore"]:
        r = results[conn]
        sizes = np.array(r["model"].fire_sizes)
        x, pdf, _ = log_binned_pdf(sizes, n_bins=35)
        if x is not None:
            tau = mle_power_law(sizes, s_min=5)
            lbl = r["label"]
            if tau:
                lbl += f" ($\\tau$={tau:.2f})"
            ax.scatter(x, pdf, s=25, alpha=0.7, color=r["color"],
                       marker=r["marker"], label=lbl, edgecolors="k", linewidths=0.3)
            if tau:
                xf = np.logspace(np.log10(5), np.log10(sizes.max()), 100)
                yf = xf ** (-tau)
                # scale to data
                idx = np.argmin(np.abs(x - 10))
                if idx < len(pdf):
                    yf *= pdf[idx] / (10.0 ** (-tau))
                ax.plot(xf, yf, "--", color=r["color"], lw=1.5, alpha=0.6)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Avalanche size $s$", fontsize=12)
    ax.set_ylabel("$P(s)$", fontsize=12)
    ax.set_title("Avalanche Size Distribution (Log-Binned)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out}/avalanche_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}/avalanche_distribution.png")

    # ── Plot 4: CCDF ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for conn in ["von_neumann", "moore"]:
        r = results[conn]
        sizes = np.sort(np.array(r["model"].fire_sizes))
        n = len(sizes)
        ccdf = 1.0 - np.arange(1, n + 1) / n
        ax.step(sizes, ccdf, color=r["color"], lw=1.2, label=r["label"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Avalanche size $s$")
    ax.set_ylabel("$P(S \\geq s)$")
    ax.set_title("Complementary CDF of Fire Sizes")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out}/ccdf.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}/ccdf.png")

    # ── Plot 5: Connectivity comparison bar chart ───────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    conns = ["von_neumann", "moore"]
    means = [np.mean(results[c]["model"].fire_sizes) for c in conns]
    maxes = [np.max(results[c]["model"].fire_sizes) for c in conns]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, means, w, label="Mean size", color="#228B22", alpha=0.8)
    ax.bar(x + w/2, maxes, w, label="Max size", color="#FF4500", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Von Neumann (4)", "Moore (8)"])
    ax.set_ylabel("Fire cluster size")
    ax.set_title("Effect of Connectivity on Avalanche Statistics")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out}/connectivity_bars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}/connectivity_bars.png")

    # ── Plot 6: Fire size time series ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3))
    sizes = results["von_neumann"]["model"].fire_sizes
    ax.bar(range(len(sizes)), sizes, width=1.0, color="#FF4500", alpha=0.6, linewidth=0)
    ax.set_xlabel("Fire event index")
    ax.set_ylabel("Avalanche size")
    ax.set_title("Avalanche Sizes Over Time (Von Neumann)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out}/fire_timeseries.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}/fire_timeseries.png")

    print(f"\n  All figures saved in: {os.path.abspath(out)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
