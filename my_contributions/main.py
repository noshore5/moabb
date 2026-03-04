"""
Example: Using CustomClassifier for Motor Imagery Classification

This example demonstrates how to:
1. Load real EEG data from MOABB
2. Preprocess using LeftRightImagery paradigm
3. Train the CustomClassifier
4. Make predictions
5. Evaluate performance with scores
"""

import warnings
warnings.filterwarnings('ignore', message='warnEpochs')

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Import MOABB components
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

# ============================================================================
# CLASSIFIER SELECTION - EASILY EDITABLE
# ============================================================================
# Set to 'wavelet' to use WaveletTransformClassifier (Random Forest)
# Set to 'cnn' to use CoherenceCNNClassifier (PyTorch CNN)
# set to 'cwtcnn' to use CWT CNN classifier (PyTorch CNN with CWT features)
# Set to 'eegnet' to use EEGNetClassifier (EEG-specific compact CNN)
CLASSIFIER_TYPE = 'wavelet'  # Change this to switch classifiers
# ============================================================================

# Import custom classifiers (lazy load based on selection)
import sys
sys.path.insert(0, '/Users/noahshore/Documents/CoherIQs/moabb/my_contributions')

if CLASSIFIER_TYPE == 'wavelet':
    from moabb_pipelines.custom_classifiers import WaveletTransformClassifier
elif CLASSIFIER_TYPE == 'cnn':
    from moabb_pipelines.coherence_cnn_classifier import CoherenceCNNClassifier
elif CLASSIFIER_TYPE == 'cwtcnn':
    from moabb_pipelines.CWT_CNN import CWTCNNClassifier
elif CLASSIFIER_TYPE == 'eegnet':
    from moabb_pipelines.EEGNet import EEGNetClassifier
else:
    raise ValueError(f"Unknown classifier type: {CLASSIFIER_TYPE}. Use 'wavelet', 'cnn', 'cwtcnn', or 'eegnet'")

# Also import wavelet transform function for visualization
sys.path.insert(0, '/Users/noahshore/Documents/CoherIQs/moabb/Coherent_Multiplex')
from utils.coherence_utils import transform


def main():
    print("=" * 70)
    print("Custom Classifier - Motor Imagery Classification")
    print("=" * 70)
    
    # Load real EEG data from MOABB
    print("\n1. Loading real EEG data from BNCI2014_001 dataset...")
    dataset = BNCI2014_001()
    # Load only the first subject
    all_subjects = [1]
    
    # Define the paradigm (left vs right hand imagery, 8-35 Hz)
    print("   Using LeftRightImagery paradigm (8-35 Hz bandpass)...")
    paradigm = LeftRightImagery(fmin=8, fmax=35)
    
    # Get preprocessed EEG data from all subjects
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=all_subjects)
    print(f"   Raw data shape: {X.shape}")
    print(f"   Number of subjects: {len(np.unique(metadata['subject']))}")
    print(f"   Classes: {np.unique(y)}")
    
    # Encode string labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"   Class distribution: {np.bincount(y_encoded)}")
    print(f"   Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"   Data will be used as-is: (n_samples, n_channels, n_timepoints)")
    
    # Split into training and evaluation using the built-in dataset sessions
    print("\n2. Splitting data using built-in train/eval sessions...")
    
    # Debug: Print metadata structure
    print(f"   Metadata columns: {metadata.columns.tolist()}")
    if 'session' in metadata.columns:
        print(f"   Unique sessions: {metadata['session'].unique()}")
    
    # Extract session info from metadata - handle both 'session' and 'session_index'
    if 'session' in metadata.columns:
        sessions = metadata['session'].values
        print(f"   Unique sessions: {np.unique(sessions)}")
        # Sessions can be '0train', '1test' or 'T', 'E' format
        import pandas as pd
        sessions_series = pd.Series(sessions)
        train_mask = (sessions == 'T') | (sessions == '0train') | sessions_series.str.contains('train', case=False, na=False).values
        eval_mask = (sessions == 'E') | (sessions == '1test') | sessions_series.str.contains('test', case=False, na=False).values
    elif 'session_index' in metadata.columns:
        # Some MOABB datasets use session_index: odd = train, even = eval or vice versa
        sessions = metadata['session_index'].values
        train_mask = sessions == 0  # First session is training
        eval_mask = sessions == 1   # Second session is evaluation
    else:
        # Fallback: use 70/30 split if session info not available
        print("   Warning: session column not found, falling back to 70/30 split")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42
        )
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Evaluation samples: {X_test.shape[0]}")
        print(f"   Data shape per sample: {X_train.shape[1:]}")
        train_mask = None
    
    if train_mask is not None:
        X_train, y_train = X[train_mask], y_encoded[train_mask]
        X_test, y_test = X[eval_mask], y_encoded[eval_mask]
        
        print(f"   Training samples: {X_train.shape[0]} (from train sessions)")
        print(f"   Evaluation samples: {X_test.shape[0]} (from test sessions)")
        if len(y_train) > 0:
            print(f"   Training class distribution: {np.bincount(y_train)}")
        if len(y_test) > 0:
            print(f"   Evaluation class distribution: {np.bincount(y_test)}")
        print(f"   Data shape per sample: {X_train.shape[1:]}")
    
    # Create the classifier with wavelet parameters
    print("\n3. Creating Classifier...")
    
    if CLASSIFIER_TYPE == 'wavelet':
        clf = WaveletTransformClassifier(
            lowest=4,
            highest=40,
            nfreqs=50,
            sampling_rate=250
        )
        print(f"   Classifier: WaveletTransformClassifier")
        print(f"   Features: Wavelet transform + coherence -> Random Forest")
    elif CLASSIFIER_TYPE == 'cnn':
        clf = CoherenceCNNClassifier(
            lowest=4,
            highest=40,
            nfreqs=50,
            sampling_rate=250,
            epochs=100,
            batch_size=32,
            learning_rate=0.001,
            device='cpu',
            use_class_weights=False  # Disabled - may be causing bias issues
        )
        print(f"   Classifier: CoherenceCNNClassifier")
        print(f"   Features: Coherence matrices -> PyTorch CNN")
    elif CLASSIFIER_TYPE == 'cwtcnn':
        clf = CWTCNNClassifier(
            lowest=4,
            highest=40,
            nfreqs=50,
            sampling_rate=250,
            epochs=100,
            batch_size=32,
            learning_rate=0.001,
            device='cpu',
            use_class_weights=False  # Disabled - may be causing bias issues
        )
        print(f"   Classifier: CWTCNNClassifier")
        print(f"   Features: Raw wavelet transforms -> PyTorch CNN")
    elif CLASSIFIER_TYPE == 'eegnet':
        clf = EEGNetClassifier(
            n_channels=22,
            n_timepoints=X_train.shape[2],  # Use actual timepoints from data
            epochs=100,
            batch_size=32,
            learning_rate=0.001,
            dropout_rate=0.5,
            device='cpu'
        )
        print(f"   Classifier: EEGNetClassifier")
        print(f"   Features: Raw EEG signals -> EEG-specific CNN (~4k parameters)")
    
    print(f"   Frequency range: {clf.lowest}-{clf.highest} Hz" if hasattr(clf, 'lowest') else "   (No frequency filtering - raw signal)")
    
    if hasattr(clf, 'nfreqs'):
        print(f"   Number of frequency scales: {clf.nfreqs}")
    
    # Train on training set and evaluate on test set
    print("\n4. Training the classifier on training data...")
    clf.fit(X_train, y_train)
    print(f"   Classes learned: {clf.classes_}")
    print("   ✓ Training complete")
    
    # Make predictions on test data
    print("\n5. Making predictions on test data...")
    y_pred = clf.predict(X_test)
    print(f"   Predictions shape: {y_pred.shape}")
    print(f"   First 10 predictions: {y_pred[:10]}")
    
    # Calculate accuracy score from already-computed predictions
    print("\n6. Evaluating performance on test set...")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Test Accuracy Score: {accuracy:.4f}")
    
    # Additional metrics
    print("\n7. Additional Metrics...")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=le.classes_,
                              zero_division=0))
    
    print(f"   Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("=" * 70)
    
    # DEBUG: Isolate just the 0th channel of the 0th sample
    print("\n8. DEBUGGING: Testing wavelet transform on isolated signal...")
    
    # Get 0th sample, 0th channel
    debug_signal = X[0, 0, :]  # Shape: (n_timepoints,)
    print(f"   Debug signal shape: {debug_signal.shape}")
    print(f"   Debug signal min/max: {debug_signal.min():.4f} / {debug_signal.max():.4f}")
    print(f"   Debug signal dtype: {debug_signal.dtype}")
    
    # Test wavelet transform
    print(f"\n   Calling transform with:")
    print(f"     signal.shape = {debug_signal.shape}")
    print(f"     frame_rate = 250")
    print(f"     highest = 40")
    print(f"     lowest = 4")
    print(f"     nfreqs = 50")
    
    try:
        coeffs, freqs = transform(
            debug_signal,
            frame_rate=250,
            highest=40,
            lowest=4,
            nfreqs=50
        )
        print(f"\n   ✓ Transform succeeded!")
        print(f"   Frequencies shape: {freqs.shape}")
        print(f"   Coefficients shape: {coeffs.shape}")
        print(f"   Coefficients dtype: {coeffs.dtype}")
        print(f"   Coefficients min/max: {np.abs(coeffs).min():.6f} / {np.abs(coeffs).max():.6f}")
        
        power = np.abs(coeffs) ** 2
        print(f"   Power min/max: {power.min():.6f} / {power.max():.6f}")
        
    except Exception as e:
        print(f"\n   ✗ Transform FAILED!")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # # Visualize raw data, FFT, and wavelet transform of first test sample
    # print("\n10. Visualizing Raw Data, FFT, and Wavelet Transform...")
    # first_sample = X_test[0]  # First test sample (n_channels, n_timepoints)
    # n_channels = first_sample.shape[0]
    # n_timepoints = first_sample.shape[1]
    # time_axis = np.arange(n_timepoints) / 250  # Convert to seconds
    # 
    # # Get first channel
    # signal = first_sample[0, :]
    # 
    # # Create figure with 3 subplots
    # fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # 
    # # Plot 1: Raw signal
    # axes[0].plot(time_axis, signal, linewidth=0.8, color='blue')
    # axes[0].set_xlabel('Time (seconds)')
    # axes[0].set_ylabel('Amplitude (µV)')
    # axes[0].set_title('Raw EEG Signal - Channel 0')
    # axes[0].grid(True, alpha=0.3)
    # 
    # # Plot 2: FFT (Power Spectrum)
    # fft_vals = np.fft.fft(signal)
    # fft_power = np.abs(fft_vals) ** 2
    # freqs_fft = np.fft.fftfreq(len(signal), 1/250)  # Frequency resolution
    # 
    # # Only plot positive frequencies
    # positive_freqs = freqs_fft[:len(freqs_fft)//2]
    # positive_power = fft_power[:len(fft_power)//2]
    # 
    # axes[1].semilogy(positive_freqs, positive_power, linewidth=1, color='green')
    # axes[1].set_xlabel('Frequency (Hz)')
    # axes[1].set_ylabel('Power (log scale)')
    # axes[1].set_title('FFT Power Spectrum - Channel 0')
    # axes[1].grid(True, alpha=0.3, which='both')
    # axes[1].set_xlim([0, 50])  # Focus on 0-50 Hz
    # 
    # # Add bands
    # axes[1].axvspan(4, 8, alpha=0.1, color='red', label='Delta (4-8 Hz)')
    # axes[1].axvspan(8, 12, alpha=0.1, color='yellow', label='Alpha (8-12 Hz)')
    # axes[1].axvspan(12, 30, alpha=0.1, color='orange', label='Beta (12-30 Hz)')
    # axes[1].legend(fontsize=8)
    # 
    # # Plot 3: Wavelet Transform
    # try:
    #     coeffs, freqs = transform(
    #         signal,
    #         frame_rate=250,
    #         highest=40,
    #         lowest=4,
    #         nfreqs=50
    #     )
    #     
    #     power = np.abs(coeffs) ** 2
    #     
    #     if power.ndim == 1:
    #         power = power.reshape(-1, 1)
    #     
    #     im = axes[2].imshow(
    #         power,
    #         aspect='auto',
    #         origin='lower',
    #         cmap='viridis',
    #         extent=[0, signal.shape[0]/250, 4, 40]
    #     )
    #     axes[2].set_xlabel('Time (seconds)')
    #     axes[2].set_ylabel('Frequency (Hz)')
    #     axes[2].set_title('Wavelet Power - Channel 0')
    #     plt.colorbar(im, ax=axes[2], label='Power')
    #     
    # except Exception as e:
    #     axes[2].text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center')
    #     axes[2].set_title(f'Wavelet - Error')
    # 
    # plt.tight_layout()
    # plt.savefig('/Users/noahshore/Documents/CoherIQs/moabb/my_contributions/fft_analysis_corrected.png', dpi=100)
    # print(f"   FFT and wavelet analysis saved!")
    # print(f"   File: my_contributions/fft_analysis_corrected.png")
    # plt.show()
    
    # # Create multi-channel raw + wavelet visualization
    # print("\n11. Creating multi-channel raw and wavelet visualization...")
    # first_sample = X_test[0]  # Shape: (22, 1001)
    # n_channels = first_sample.shape[0]
    # 
    # fig, axes = plt.subplots(n_channels, 2, figsize=(14, 4*n_channels))
    # 
    # for ch_idx in range(n_channels):
    #     signal = first_sample[ch_idx, :]
    #     
    #     # Plot raw signal
    #     axes[ch_idx, 0].plot(time_axis, signal, linewidth=0.5, color='blue')
    #     axes[ch_idx, 0].set_ylabel(f'Ch {ch_idx}', fontsize=9)
    #     axes[ch_idx, 0].set_xlim([0, 4])
    #     axes[ch_idx, 0].grid(True, alpha=0.2)
    #     if ch_idx == 0:
    #         axes[ch_idx, 0].set_title('Raw EEG Signal', fontsize=11)
    #     if ch_idx == n_channels - 1:
    #         axes[ch_idx, 0].set_xlabel('Time (seconds)', fontsize=9)
    #     
    #     # Plot wavelet transform
    #     try:
    #         coeffs, freqs = transform(
    #             signal,
    #             frame_rate=250,
    #             highest=40,
    #             lowest=4,
    #             nfreqs=50
    #         )
    #         power = np.abs(coeffs) ** 2
    #         
    #         im = axes[ch_idx, 1].imshow(
    #             power,
    #             aspect='auto',
    #             origin='lower',
    #             cmap='viridis',
    #             extent=[0, 4, 4, 40],
    #             vmin=0,
    #             vmax=np.percentile(power, 95)  # Normalize to 95th percentile
    #         )
    #         axes[ch_idx, 1].set_xlim([0, 4])
    #         axes[ch_idx, 1].set_ylim([4, 40])
    #         if ch_idx == 0:
    #             axes[ch_idx, 1].set_title('Wavelet Power (Hz)', fontsize=11)
    #         if ch_idx == n_channels - 1:
    #             axes[ch_idx, 1].set_xlabel('Time (seconds)', fontsize=9)
    #         axes[ch_idx, 1].set_ylabel('Freq (Hz)', fontsize=8)
    #         
    #     except Exception as e:
    #         axes[ch_idx, 1].text(0.5, 0.5, f'Error: {str(e)[:30]}', 
    #                             ha='center', va='center', transform=axes[ch_idx, 1].transAxes)
    #         axes[ch_idx, 1].set_title(f'Wavelet - Error')
    # 
    # plt.tight_layout()
    # plt.savefig('/Users/noahshore/Documents/CoherIQs/moabb/my_contributions/raw_and_wavelet_visualization.png', dpi=100)
    # print(f"   ✓ Raw and wavelet visualization saved!")
    # print(f"   File: my_contributions/raw_and_wavelet_visualization.png")
    # plt.show()


if __name__ == "__main__":
    main()
