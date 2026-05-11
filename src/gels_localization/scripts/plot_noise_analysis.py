#!/usr/bin/env python3
"""
Noise analysis plotting utilities.

Provides plotting functions for position and orientation errors vs distance.
"""

import numpy as np


def plot_errors(ep_mm, eR_deg, dist_mm, B_norm, csv_path):
    """
    Plot ep and eR vs distance using subplots.

    Args:
        ep_mm: position error in mm
        eR_deg: orientation error in degrees
        dist_mm: distance from sample to pbar in mm
        B_norm: magnetic field strength in Gs
        csv_path: path to CSV (used to derive plot output path)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fontsize = 14
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sort by distance for cleaner line
    sort_idx = np.argsort(dist_mm)
    dist_sorted = dist_mm[sort_idx]
    ep_sorted = ep_mm[sort_idx]
    eR_sorted = eR_deg[sort_idx]

    # ep vs distance
    ax = axes[0]
    ax.plot(dist_sorted, ep_sorted, 'b.-', linewidth=1.5, markersize=5)
    ax.set_xlabel('d_pbar [mm]')
    ax.set_ylabel('ep [mm]')
    ax.set_title('Position Error vs Distance to pbar')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2, linestyle=':')

    # eR vs distance
    ax = axes[1]
    ax.plot(dist_sorted, eR_sorted, 'r.-', linewidth=1.5, markersize=5)
    ax.set_xlabel('d_pbar [mm]')
    ax.set_ylabel('eR [deg]')
    ax.set_title('Orientation Error vs Distance to pbar')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2, linestyle=':')

    plt.tight_layout(pad=1.0)

    plot_path = csv_path.replace('.csv', '.pdf')
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {plot_path}")
    plt.close()
