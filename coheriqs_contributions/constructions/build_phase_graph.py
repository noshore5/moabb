import numpy as np
from itertools import combinations
import sys
from pathlib import Path
import networkx as nx

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Coherent_Multiplex.utils.coherence_utils import coherence

def build_phase_graph(coeffs_all, freqs, threshold=0.7, window_size=25, phase_threshold_deg=30, max_edges_per_window=True) -> nx.MultiDiGraph:
    """
    Build a directed multigraph from wavelet coherence data.
    
    Parameters:
    -----------
    coeffs_all : list of ndarray
        List of complex CWT coefficient arrays, one per channel, each shaped (n_freqs, n_times)
    freqs : ndarray
        Frequency values corresponding to the frequency axis
    threshold : float
        Coherence threshold for creating edges (default 0.7)
    window_size : int
        Width of non-overlapping time windows in samples (default 25)
    phase_threshold_deg : float
        Minimum absolute phase difference in degrees for an edge to exist (default 30)
    max_edges_per_window : bool
        If True, keep only the strongest coherence edge per (ch1, ch2, time) window (default True)
        
    Returns:
    --------
    nx.MultiDiGraph
        Directed multigraph where edges are time-specific (and optionally frequency-specific).
        Each edge stores:
        - time: window center sample index
        - frequency: frequency value of this edge (if max_edges_per_window=False)
    """
    n_channels = len(coeffs_all)
    n_freqs, n_times = coeffs_all[0].shape
    
    # Convert phase threshold to radians
    phase_threshold_rad = np.deg2rad(phase_threshold_deg)
    
    # Create empty multigraph
    G = nx.MultiDiGraph()
    
    # Add nodes for all channels
    G.add_nodes_from(range(n_channels))
    
    # Get all channel pairs
    channel_pairs = list(combinations(range(n_channels), 2))
    
    # Diagnostics
    edges_created = 0
    windows_processed = 0
    
    # Process each channel pair
    for ch1, ch2 in channel_pairs:
        coeff1 = coeffs_all[ch1]
        coeff2 = coeffs_all[ch2]
        
        # Use proper coherence function from Coherent_Multiplex
        coh, _, _ = coherence(coeff1, coeff2, freqs)
        
        # Compute phase difference
        phase1 = np.angle(coeff1)
        phase2 = np.angle(coeff2)
        phase_diff = phase1 - phase2
        
        # Process non-overlapping time windows
        for window_start in range(0, n_times - window_size + 1, window_size):
            window_end = window_start + window_size
            window_center = (window_start + window_end - 1) / 2.0
            windows_processed += 1
            
            # Process each frequency independently
            coh_window = coh[:, window_start:window_end]
            phase_window = phase_diff[:, window_start:window_end]
            
            if max_edges_per_window:
                # Find the frequency with highest coherence that passes thresholds
                best_freq_idx = None
                best_coh = 0
                best_phase = 0
                best_direction = None
                
                for freq_idx in range(n_freqs):
                    # Average coherence across time for this frequency
                    mean_coh_freq = np.mean(coh_window[freq_idx, :])
                    
                    # Check if coherence exceeds threshold
                    if mean_coh_freq > threshold:
                        # Average phase difference across time for this frequency
                        mean_phase_freq = np.mean(phase_window[freq_idx, :])
                        
                        # Check if absolute phase difference exceeds threshold
                        if np.abs(mean_phase_freq) > phase_threshold_rad:
                            # Keep track of best (highest coherence)
                            if mean_coh_freq > best_coh:
                                best_coh = mean_coh_freq
                                best_freq_idx = freq_idx
                                best_phase = mean_phase_freq
                
                # Add best edge if found
                if best_freq_idx is not None:
                    if best_phase > 0:
                        # Edge from ch1 to ch2
                        G.add_edge(ch1, ch2, time=window_center, frequency=freqs[best_freq_idx])
                    else:
                        # Edge from ch2 to ch1
                        G.add_edge(ch2, ch1, time=window_center, frequency=freqs[best_freq_idx])
                    edges_created += 1
            else:
                # Original behavior: keep all frequency-specific edges
                for freq_idx in range(n_freqs):
                    # Average coherence across time for this frequency
                    mean_coh_freq = np.mean(coh_window[freq_idx, :])
                    
                    # Add edge if coherence exceeds threshold AND phase difference exceeds threshold
                    if mean_coh_freq > threshold:
                        # Average phase difference across time for this frequency
                        mean_phase_freq = np.mean(phase_window[freq_idx, :])
                        
                        # Check if absolute phase difference exceeds threshold
                        if np.abs(mean_phase_freq) > phase_threshold_rad:
                            if mean_phase_freq > 0:
                                # Edge from ch1 to ch2
                                G.add_edge(ch1, ch2, time=window_center, frequency=freqs[best_freq_idx])
                                edges_created += 1
                            else:
                                # Edge from ch2 to ch1
                                G.add_edge(ch2, ch1, time=window_center, frequency=freqs[best_freq_idx])
                                edges_created += 1
    
    # Print diagnostics
    print(f"  Windows processed: {windows_processed}")
    print(f"  Edges created: {edges_created}")
    
    return G