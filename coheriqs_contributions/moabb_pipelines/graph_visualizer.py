"""Visualizer for sparse wavelet coherence graphs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch


def visualize_graph(
    core_model,
    raw_x: torch.Tensor,
    sampling_rate: int,
    sample_idx: int = 0,
    figsize: tuple = (16, 10),
) -> dict:
    """Visualize the sparse graph built from a sample.
    
    Parameters
    ----------
    core_model : WCTEvidenceGNNCore
        The model with `build_graph` method.
    raw_x : torch.Tensor
        Input signal, shape (B, C, T).
    sampling_rate : int
        Sampling rate in Hz.
    sample_idx : int
        Which batch sample to visualize (default 0).
    figsize : tuple
        Figure size (width, height).
    
    Returns
    -------
    dict
        Statistics: n_windows, n_channels, n_edges, active_count_per_window, etc.
    """
    # Build graph
    graph = core_model.build_graph(raw_x, sampling_rate)
    batch_size = graph["batch_size"]
    n_channels = graph["n_channels"]
    nfreqs = graph["nfreqs"]
    windows = graph["windows"]
    
    if sample_idx >= batch_size:
        raise ValueError(f"sample_idx {sample_idx} out of range [0, {batch_size})")
    
    n_windows = len(windows)
    n_edges = (n_channels - 1) * n_channels  # directed edges
    
    # Gather statistics
    active_counts_per_window = []
    active_edges_per_window = []
    active_freqs_per_window = []
    
    for win_idx, win in enumerate(windows):
        edge_idx_b, freq_idx_b = win["active"][sample_idx]
        n_active = int(edge_idx_b.numel())
        active_counts_per_window.append(n_active)
        active_edges_per_window.append(edge_idx_b.cpu().numpy() if n_active > 0 else np.array([], dtype=int))
        active_freqs_per_window.append(freq_idx_b.cpu().numpy() if n_active > 0 else np.array([], dtype=int))
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"Sparse Graph Visualization (Sample {sample_idx})", fontsize=14, fontweight='bold')
    
    # 1. Active edge count per window
    ax = axes[0, 0]
    ax.bar(range(n_windows), active_counts_per_window, color='steelblue', alpha=0.7)
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Active Edge-Frequency Slots')
    ax.set_title('Activity per Window')
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Heatmap of active slots (edges × frequencies for first window if it exists)
    ax = axes[0, 1]
    if active_counts_per_window[0] > 0:
        heatmap = np.zeros((n_edges, nfreqs))
        edges_w0 = active_edges_per_window[0]
        freqs_w0 = active_freqs_per_window[0]
        for e, f in zip(edges_w0, freqs_w0):
            heatmap[e, f] = 1
        im = ax.imshow(heatmap, aspect='auto', cmap='Blues', origin='lower')
        ax.set_xlabel('Frequency Bin')
        ax.set_ylabel('Edge Index')
        ax.set_title(f'Active Slots - Window 0 (n_active={active_counts_per_window[0]})')
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, 'No active slots in Window 0', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Active Slots - Window 0')
    
    # 3. Total active slots across all windows
    ax = axes[1, 0]
    total_possible = n_windows * n_edges * nfreqs
    total_active = sum(active_counts_per_window)
    sparsity = 100.0 * (1.0 - total_active / total_possible) if total_possible > 0 else 0
    
    stats_text = f"""
    Batch Size: {batch_size}
    Channels: {n_channels}
    Directed Edges: {n_edges}
    Frequencies: {nfreqs}
    Windows: {n_windows}
    
    Total Possible Slots: {total_possible:,}
    Total Active Slots: {total_active:,}
    Sparsity: {sparsity:.1f}%
    Activity Ratio: {100*total_active/total_possible:.2f}%
    
    Avg Active/Window: {np.mean(active_counts_per_window):.1f}
    Max Active/Window: {max(active_counts_per_window)}
    Min Active/Window: {min(active_counts_per_window)}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', verticalalignment='center',
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    
    # 4. Frequency distribution of active slots
    ax = axes[1, 1]
    freq_counts = np.zeros(nfreqs)
    for freqs in active_freqs_per_window:
        if len(freqs) > 0:
            np.add.at(freq_counts, freqs, 1)
    ax.plot(freq_counts, marker='o', linewidth=2, markersize=4, color='darkgreen')
    ax.set_xlabel('Frequency Bin')
    ax.set_ylabel('Active Count')
    ax.set_title('Frequency Distribution (All Windows)')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        "n_windows": n_windows,
        "n_channels": n_channels,
        "n_edges": n_edges,
        "nfreqs": nfreqs,
        "total_possible_slots": total_possible,
        "total_active_slots": total_active,
        "sparsity_percent": sparsity,
        "active_counts_per_window": active_counts_per_window,
    }


def print_graph_summary(
    core_model,
    raw_x: torch.Tensor,
    sampling_rate: int,
    sample_idx: int = 0,
) -> None:
    """Print a text summary of the graph structure.
    
    Parameters
    ----------
    core_model : WCTEvidenceGNNCore
        The model with `build_graph` method.
    raw_x : torch.Tensor
        Input signal, shape (B, C, T).
    sampling_rate : int
        Sampling rate in Hz.
    sample_idx : int
        Which batch sample to summarize (default 0).
    """
    graph = core_model.build_graph(raw_x, sampling_rate)
    batch_size = graph["batch_size"]
    n_channels = graph["n_channels"]
    nfreqs = graph["nfreqs"]
    windows = graph["windows"]
    
    if sample_idx >= batch_size:
        raise ValueError(f"sample_idx {sample_idx} out of range [0, {batch_size})")
    
    n_windows = len(windows)
    n_edges = (n_channels - 1) * n_channels
    total_possible = n_windows * n_edges * nfreqs
    
    print("\n" + "="*60)
    print("SPARSE GRAPH SUMMARY")
    print("="*60)
    print(f"Sample Index: {sample_idx} (of {batch_size})")
    print(f"Channels: {n_channels}")
    print(f"Directed Edges: {n_edges}")
    print(f"Frequency Bins: {nfreqs}")
    print(f"Windows: {n_windows}")
    print(f"Total Possible Slots: {total_possible:,}")
    print("-"*60)
    
    total_active = 0
    for win_idx, win in enumerate(windows):
        edge_idx_b, freq_idx_b = win["active"][sample_idx]
        n_active = int(edge_idx_b.numel())
        total_active += n_active
        t_center = win["t_center"]
        print(f"Window {win_idx:2d} (t_center={t_center:5d}): {n_active:6d} active slots")
    
    sparsity = 100.0 * (1.0 - total_active / total_possible) if total_possible > 0 else 0
    print("-"*60)
    print(f"Total Active Slots: {total_active:,} ({100*total_active/total_possible:.2f}%)")
    print(f"Sparsity: {sparsity:.1f}%")
    print("="*60 + "\n")
