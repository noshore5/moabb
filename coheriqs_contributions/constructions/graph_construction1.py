from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
import numpy as np
from itertools import combinations
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - no window
import matplotlib.pyplot as plt
from build_phase_graph import build_phase_graph
from visualiser import print_node_names, print_edges_sample

# Add parent directories to path (after standard imports to avoid shadowing)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Coherent_Multiplex.utils.coherence_utils import transform, coherence

# Initialize dataset
dataset = BNCI2014_001()

# Initialize paradigm with fmin=8 and fmax=35
paradigm = LeftRightImagery(fmin=8, fmax=35)

# Get data for subject 1
X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])

# Separate left vs right samples
left_indices = np.where(y == 'left_hand')[0]  # left_hand label
right_indices = np.where(y == 'right_hand')[0]  # right_hand label

left_samples = X[left_indices]
right_samples = X[right_indices]

print(f"Left samples shape: {left_samples.shape}")
print(f"Right samples shape: {right_samples.shape}")

# Compute wavelet coherence for all channel pairings in multiple samples
frame_rate = 250  # Adjust based on your sampling rate
fmin = 8
fmax = 35

# Generate all channel pairings
n_channels = left_samples[0].shape[0]
channel_pairs = list(combinations(range(n_channels), 2))

# Process first two samples
for sample_idx in range(min(2, len(left_samples))):
    print(f"\n--- Processing Sample {sample_idx} ---")
    sample = left_samples[sample_idx]
    
    # Compute wavelet transforms for all channels
    coeffs_all = []
    for ch in range(n_channels):
        coeffs, freqs = transform(sample[ch], frame_rate, fmax, fmin, nfreqs=25)
        coeffs_all.append(coeffs)

    # Compute coherence for all pairs
    coherence_matrix = np.zeros((len(channel_pairs), len(freqs)))
    for pair_idx, (ch1, ch2) in enumerate(channel_pairs):
        coh, _, _ = coherence(coeffs_all[ch1], coeffs_all[ch2], freqs)
        coherence_matrix[pair_idx] = np.mean(coh, axis=1)  # Average across time

    print(f"Coherence matrix shape: {coherence_matrix.shape}")
    print(f"Number of channel pairs: {len(channel_pairs)}")
    print(f"Coherence min: {coherence_matrix.min():.4f}, max: {coherence_matrix.max():.4f}, mean: {coherence_matrix.mean():.4f}")
    print(f"Coherence > 0.7: {(coherence_matrix > 0.7).sum()} / {coherence_matrix.size}")
    
    # Build phase-based directed graph
    G = build_phase_graph(coeffs_all, freqs, threshold=0.7, window_size=25)
    
    print(f"Graph nodes: {G.number_of_nodes()}")
    print(f"Graph edges: {G.number_of_edges()}")
    
    # Print first edge
    if G.number_of_edges() > 0:
        first_edge = next(iter(G.edges(keys=True, data=True)))
        ch1, ch2, key, attrs = first_edge
        print(f"First edge: {ch1} -> {ch2} (key={key})")
        print(f"  Attributes: {attrs}")
    
    # Print node names with degree

# Plot the first 10 channel pairs with wavelet coherence
# n_pairs_to_plot = min(10, len(channel_pairs))
# fig, axes = plt.subplots(2, 5, figsize=(16, 8))
# axes = axes.flatten()

# for pair_idx in range(n_pairs_to_plot):
#     ch1, ch2 = channel_pairs[pair_idx]
#     coh, _, _ = coherence(coeffs_all[ch1], coeffs_all[ch2], freqs)
#     
#     # Plot wavelet coherence (time x frequency)
#     im = axes[pair_idx].imshow(coh, aspect='auto', origin='lower', cmap='hot', vmin=0, vmax=1)
#     axes[pair_idx].set_title(f'Ch {ch1} vs Ch {ch2}')
#     axes[pair_idx].set_xlabel('Time')
#     axes[pair_idx].set_ylabel('Frequency (Hz)')
#     # Set y-axis limits and ticks to show actual frequency range
#     axes[pair_idx].set_ylim([0, len(freqs)-1])
#     freq_ticks = np.linspace(0, len(freqs)-1, 6)
#     freq_labels = np.linspace(fmin, fmax, 6)
#     axes[pair_idx].set_yticks(freq_ticks)
#     axes[pair_idx].set_yticklabels([f'{f:.1f}' for f in freq_labels])
#     plt.colorbar(im, ax=axes[pair_idx], label='Coherence')

# plt.tight_layout()
# plt.savefig('/Users/noahshore/Documents/CoherIQs/moabb/coheriqs_contributions/constructions/coherograms.png', dpi=100, bbox_inches='tight')
# print("Coherograms saved to coheriqs_contributions/constructions/coherograms.png")


from visualiser import print_node_names
print_node_names(G)