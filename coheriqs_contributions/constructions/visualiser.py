import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import mne
from matplotlib import animation

from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

# Add parent directories to path (AFTER standard imports to avoid shadowing)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from build_phase_graph import build_phase_graph
from Coherent_Multiplex.utils.coherence_utils import transform, coherence


def get_electrode_names_from_epochs(epochs, n_channels=None):
    """Extract channel names from an MNE Epochs object."""
    try:
        ch_names = list(epochs.info["ch_names"])
        if n_channels is not None:
            return ch_names[:n_channels]
        return ch_names
    except Exception:
        return None


def get_electrode_names(n_channels=22, epochs=None):
    """Get electrode names from epochs when available, otherwise fallback."""
    if epochs is not None:
        ch_names = get_electrode_names_from_epochs(epochs, n_channels=n_channels)
        if ch_names:
            return ch_names

    return [f"Ch{i}" for i in range(n_channels)]


def print_node_names(G, X=None, epochs=None):
    """
    Print all node names (channel indices and electrode names) in the graph.
    
    Parameters:
    -----------
    G : nx.MultiDiGraph
        The directed multigraph to inspect
    X : ndarray, optional
        Raw data array with shape (n_samples, n_channels, n_times) to extract channel names from MNE info
    """
    print(f"Graph has {G.number_of_nodes()} nodes:")
    
    # Get electrode names from dataset
    electrode_names = get_electrode_names(G.number_of_nodes(), epochs=epochs)
    
    for node in sorted(G.nodes()):
        # Count incoming and outgoing edges
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        
        # Get electrode name if available
        if electrode_names and node < len(electrode_names):
            name = f"Node {node} ({electrode_names[node]})"
        else:
            name = f"Node {node}"
        
        print(f"  {name}: in_degree={in_degree}, out_degree={out_degree}")


def print_edges_sample(G, n_edges=10):
    """
    Print a sample of edges from the graph.
    
    Parameters:
    -----------
    G : nx.MultiDiGraph
        The directed multigraph to inspect
    n_edges : int
        Number of edges to print (default 10)
    """
    print(f"\nFirst {n_edges} edges:")
    for i, (ch1, ch2, key, attrs) in enumerate(G.edges(keys=True, data=True)):
        if i >= n_edges:
            break
        print(f"  {ch1} -> {ch2} (key={key}): time={attrs.get('time', 'N/A')}, freq={attrs.get('frequency', 'N/A')}")


def _get_sensor_layout(epochs, labels):
    if epochs is None:
        return None

    montage = epochs.get_montage()
    if montage is None:
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
        except Exception:
            return None

    ch_pos = montage.get_positions().get("ch_pos", {})
    if not ch_pos:
        return None

    positions = {}
    for node, label in labels.items():
        pos = ch_pos.get(label)
        if pos is None:
            continue
        positions[node] = (float(pos[0]), float(pos[1]))

    return positions if positions else None


def visualize_graph(
    G,
    title="Network Graph",
    output_path=None,
    figsize=(16, 12),
    epochs=None,
    verbose=False,
):
    """
    Create a visual representation of the directed graph.
    
    Parameters:
    -----------
    G : nx.MultiDiGraph
        The directed multigraph to visualize
    title : str
        Title for the plot
    output_path : str, optional
        Path to save the figure. If None, displays the plot.
    figsize : tuple
        Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get electrode names
    electrode_names = get_electrode_names(G.number_of_nodes(), epochs=epochs)
    labels = {i: electrode_names[i] if i < len(electrode_names) else f"Ch{i}" for i in G.nodes()}

    # Layout electrodes using sensor positions when available
    pos = _get_sensor_layout(epochs, labels)
    if pos is None or len(pos) < len(G.nodes()):
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes with size proportional to degree
    node_sizes = [300 + 10 * (G.in_degree(node) + G.out_degree(node)) for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_sizes, ax=ax)
    
    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=15, arrowstyle='->', width=0.5, 
                          alpha=0.6, connectionstyle='arc3,rad=0.1', ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        if verbose:
            print(f"Graph visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def animate_graph_by_time(
    G,
    title="Network Graph (Animated)",
    output_path=None,
    figsize=(16, 12),
    epochs=None,
    fps=6,
    n_frames_per_window=5,
    show_base_connections=True,
):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0b0f1a")
    ax.set_facecolor("#0b0f1a")

    electrode_names = get_electrode_names(G.number_of_nodes(), epochs=epochs)
    labels = {i: electrode_names[i] if i < len(electrode_names) else f"Ch{i}" for i in G.nodes()}

    pos = _get_sensor_layout(epochs, labels)
    if pos is None or len(pos) < len(G.nodes()):
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    node_sizes = [260 + 8 * (G.in_degree(node) + G.out_degree(node)) for node in G.nodes()]
    times = sorted({attrs.get("time") for _, _, attrs in G.edges(data=True) if attrs.get("time") is not None})
    if not times:
        times = [0]

    freqs = [attrs.get("frequency") for _, _, attrs in G.edges(data=True) if attrs.get("frequency") is not None]
    if freqs:
        vmin, vmax = float(min(freqs)), float(max(freqs))
    else:
        vmin, vmax = 0.0, 1.0
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Frequency (Hz)", color="#e6edf7")
    cbar.ax.yaxis.set_tick_params(color="#e6edf7")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#e6edf7")

    sfreq = float(epochs.info.get("sfreq", 1.0)) if epochs is not None else 1.0

    frame_specs = [(time_value, frame_idx) for time_value in times for frame_idx in range(n_frames_per_window)]
    if not frame_specs:
        frame_specs = [(0, 0)]

    def draw_frame(frame_spec):
        time_value, frame_idx = frame_spec
        progress = (frame_idx + 1) / max(n_frames_per_window, 1)
        time_seconds = time_value / sfreq if sfreq else 0.0
        ax.clear()
        ax.set_facecolor("#0b0f1a")
        ax.set_title(title, fontsize=14, fontweight="bold", color="#e6edf7")
        ax.axis("off")
        ax.text(
            0.99,
            0.02,
            f"t = {time_seconds:.2f} s",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color="#e6edf7",
            fontsize=10,
        )

        if show_base_connections:
            base_edges = [(u, v) for u in G.nodes() for v in G.nodes() if u < v]
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=base_edges,
                edge_color="#2b3b5f",
                width=0.6,
                alpha=0.5,
                ax=ax,
            )

        nx.draw_networkx_nodes(G, pos, node_color="#8bd3ff", node_size=node_sizes, ax=ax)
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="#0b0f1a", ax=ax)

        edges = [(u, v, attrs) for u, v, attrs in G.edges(data=True) if attrs.get("time") == time_value]
        if edges:
            particle_positions = []
            particle_colors = []
            for u, v, attrs in edges:
                start = np.array(pos[u], dtype=float)
                end = np.array(pos[v], dtype=float)
                point = (1.0 - progress) * start + progress * end
                particle_positions.append(point)
                freq = attrs.get("frequency")
                particle_colors.append(cmap(norm(freq)) if freq is not None else "#8fffe2")

            particles = np.array(particle_positions)
            ax.scatter(
                particles[:, 0],
                particles[:, 1],
                s=30,
                c=particle_colors,
                alpha=0.9,
                edgecolors="none",
                zorder=4,
            )
            ax.scatter(
                particles[:, 0],
                particles[:, 1],
                s=80,
                c="#4cc9f0",
                alpha=0.25,
                edgecolors="none",
                zorder=3,
            )

    anim = animation.FuncAnimation(
        fig,
        func=draw_frame,
        frames=frame_specs,
        interval=int(1000 / max(fps, 1)),
        repeat=True,
    )

    if output_path:
        anim.save(output_path, writer="pillow", fps=fps)
    else:
        plt.show()

    plt.close()


def create_graph_from_sample(sample, n_channels, frame_rate=250, fmin=8, fmax=35, 
                            nfreqs=25, threshold=0.7, window_size=25, phase_threshold_deg=30):
    """
    Create a phase-based graph from a single EEG sample.
    
    Parameters:
    -----------
    sample : ndarray
        EEG sample of shape (n_channels, n_times)
    n_channels : int
        Number of channels
    frame_rate : int
        Sampling rate in Hz
    fmin, fmax : float
        Frequency bounds
    nfreqs : int
        Number of frequency bins
    threshold : float
        Coherence threshold
    window_size : int
        Time window size
    phase_threshold_deg : float
        Phase difference threshold in degrees
        
    Returns:
    --------
    nx.MultiDiGraph
        The constructed graph
    """
    # Compute wavelet transforms
    coeffs_all = []
    for ch in range(n_channels):
        coeffs, freqs = transform(sample[ch], frame_rate, fmax, fmin, nfreqs=nfreqs)
        coeffs_all.append(coeffs)
    
    # Build graph
    G = build_phase_graph(coeffs_all, freqs, threshold=threshold, window_size=window_size, 
                         phase_threshold_deg=phase_threshold_deg, max_edges_per_window=True)
    
    return G


if __name__ == "__main__":
    # Load data
    dataset = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=8, fmax=35)
    epochs, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1], return_epochs=True)
    X = epochs.get_data()
    
    # Get right hand samples
    right_indices = np.where(y == 'right_hand')[0]
    right_samples = X[right_indices]

        # Get left hand samples
    left_indices = np.where(y == 'left_hand')[0]
    left_samples = X[left_indices]

    # Create graph from first sample
    sample_index = 0
    use_right = True  # Set to False for left hand samples
    
    if use_right:
        sample = right_samples[sample_index]
        sample_label = y[right_indices[sample_index]]
    else:
        sample = left_samples[sample_index]
        sample_label = y[left_indices[sample_index]]
    
    G = create_graph_from_sample(sample, n_channels=22)
    
    # Create visualization
    output_path = Path(__file__).parent / "graph_visualization.png"
    title = f"EEG Phase-based Directed Graph Network | sample {sample_index} | {sample_label}"
    visualize_graph(
        G,
        title=title,
        output_path=str(output_path),
        epochs=epochs,
        verbose=False,
    )

    gif_path = Path(__file__).parent / "graph_animation.gif"
    animate_graph_by_time(
        G,
        title=title,
        output_path=str(gif_path),
        epochs=epochs,
        fps=20,
        n_frames_per_window=10,
    )

