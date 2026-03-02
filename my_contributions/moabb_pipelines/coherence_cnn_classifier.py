"""Coherence-based CNN classifier for EEG signal classification.

This module contains a CNN classifier that uses wavelet coherence features
for EEG signal classification with PyTorch.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.signal import resample
import sys
sys.path.insert(0, '/Users/noahshore/Documents/CoherIQs/moabb/Coherent_Multiplex')
from utils.coherence_utils import transform, coherence

log = logging.getLogger(__name__)


class CoherenceCNN(nn.Module):
    """PyTorch CNN model for coherence matrix classification."""
    
    def __init__(self, n_classes, input_shape):
        super(CoherenceCNN, self).__init__()
        n_pairs, nfreqs, n_timepoints = input_shape
        
        # Input shape: (batch_size, n_pairs, nfreqs, n_timepoints)
        # Treat n_pairs as the channel dimension
        
        # First convolutional block
        self.conv1 = nn.Conv2d(n_pairs, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(0.25)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout(0.25)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout3 = nn.Dropout(0.25)
        
        # Calculate flattened size after convolutions
        # After each pool operation, dimensions are halved
        flattened_nfreqs = nfreqs // 8
        flattened_ntimepoints = n_timepoints // 8
        flattened_size = 128 * flattened_nfreqs * flattened_ntimepoints
        
        # Dense layers
        self.fc1 = nn.Linear(flattened_size, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout_fc2 = nn.Dropout(0.5)
        
        self.fc_out = nn.Linear(128, n_classes)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = torch.relu(x)
        x = self.dropout_fc1(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = torch.relu(x)
        x = self.dropout_fc2(x)
        
        x = self.fc_out(x)
        return x


class CoherenceCNNClassifier(BaseEstimator, ClassifierMixin):
    """CNN classifier using wavelet coherence features for EEG classification.
    
    This classifier computes wavelet coherence between all unique channel pairs
    and uses a CNN architecture to learn spatial-temporal patterns from the
    coherence matrices.
    
    Parameters
    ----------
    lowest : float, default=4
        Lowest frequency (Hz) for the wavelet transform
    
    highest : float, default=40
        Highest frequency (Hz) for the wavelet transform
    
    nfreqs : int, default=50
        Number of frequency scales for the wavelet transform
    
    sampling_rate : int, default=250
        Sampling rate of the EEG signal (Hz)
    
    epochs : int, default=50
        Number of training epochs for the neural network
    
    batch_size : int, default=32
        Batch size for training
    
    learning_rate : float, default=0.001
        Learning rate for the optimizer
    
    device : str, default='cpu'
        Device to use for training ('cpu' or 'cuda')
    
    use_class_weights : bool, default=True
        Whether to use class weights to handle imbalanced data
    
    verbose : int, default=0
        Verbosity level for training
        
    Examples
    --------
    >>> clf = CoherenceCNNClassifier(lowest=4, highest=40, nfreqs=50)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """
    
    def __init__(self, lowest=4, highest=40, nfreqs=50, sampling_rate=250,
                 epochs=50, batch_size=32, learning_rate=0.001, device='cpu',
                 use_class_weights=True, verbose=0):
        self.lowest = lowest
        self.highest = highest
        self.nfreqs = nfreqs
        self.sampling_rate = sampling_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.use_class_weights = use_class_weights
        self.verbose = verbose
        self.model_ = None
        self.classes_ = None
        self.class_to_idx_ = None
        
    def _compute_coherence_matrices(self, X):
        """Compute wavelet coherence matrices between all channel pairs.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_channels, n_timepoints)
            EEG data
            
        Returns
        -------
        coherence_matrices : array-like, shape (n_samples, n_pairs, nfreqs, n_timepoints)
            Coherence matrices for each unique channel pair
        """
        n_samples, n_channels, n_timepoints = X.shape
        unique_pairs = n_channels * (n_channels - 1) // 2
        
        log.info(f"Computing coherence matrices for {n_samples} samples, {n_channels} channels")
        log.info(f"Number of unique channel pairs: {unique_pairs}")
        print(f"   Computing coherence matrices for {unique_pairs} channel pairs...", flush=True)
        
        # Pre-compute all wavelet transforms
        log.info("Pre-computing all wavelet transforms...")
        wavelet_coeffs = {}
        
        for sample_idx in range(n_samples):
            for ch_idx in range(n_channels):
                signal = X[sample_idx, ch_idx, :]
                
                try:
                    coeffs, freqs = transform(
                        signal, 
                        self.sampling_rate, 
                        self.highest, 
                        self.lowest, 
                        nfreqs=self.nfreqs
                    )
                    
                    # Downsample the wavelet coefficients to 100 timepoints
                    if coeffs.ndim == 2:
                        # Shape: (nfreqs, n_timepoints)
                        coeffs = resample(coeffs, 100, axis=1)
                    elif coeffs.ndim == 1:
                        # Shape: (n_timepoints,)
                        coeffs = resample(coeffs, 100)
                    
                    wavelet_coeffs[(sample_idx, ch_idx)] = (coeffs, freqs)
                except Exception as e:
                    log.debug(f"Error in wavelet transform for sample {sample_idx}, channel {ch_idx}: {e}")
                    wavelet_coeffs[(sample_idx, ch_idx)] = (None, None)
        
        log.info("✓ All wavelet transforms completed")
        print(f"   ✓ Completed all {n_samples * n_channels} wavelet transforms!", flush=True)
        
        # Compute coherence matrices
        print(f"   Computing coherence for {n_samples * unique_pairs} pairs...", flush=True)
        coherence_matrices = []
        pair_count = 0
        total_pairs = n_samples * unique_pairs
        
        for sample_idx in range(n_samples):
            sample_coherences = []
            
            # Get coherence for each unique pair of channels
            for ch_i in range(n_channels):
                for ch_j in range(ch_i + 1, n_channels):
                    pair_count += 1
                    if pair_count % max(1, total_pairs // 10) == 0:
                        print(f"     Progress: {pair_count}/{total_pairs} pairs computed ({100*pair_count/total_pairs:.0f}%)", flush=True)
                    
                    coeffs_i, freqs_i = wavelet_coeffs.get((sample_idx, ch_i), (None, None))
                    coeffs_j, freqs_j = wavelet_coeffs.get((sample_idx, ch_j), (None, None))
                    
                    if coeffs_i is not None and coeffs_j is not None:
                        try:
                            # Compute wavelet coherence
                            coh, _, _ = coherence(coeffs_i, coeffs_j, freqs_i)
                            sample_coherences.append(coh)
                        except Exception as e:
                            log.debug(f"Error computing coherence for sample {sample_idx}, channels ({ch_i}, {ch_j}): {e}")
                            # Fallback: create zero matrix with expected shape (nfreqs, 100)
                            zero_coh = np.zeros((self.nfreqs, 100))
                            sample_coherences.append(zero_coh)
                    else:
                        # Missing data: create zero matrix (nfreqs, 100)
                        zero_coh = np.zeros((self.nfreqs, 100))
                        sample_coherences.append(zero_coh)
            
            coherence_matrices.append(sample_coherences)
        
        # Convert to numpy array: (n_samples, n_pairs, nfreqs, n_timepoints)
        print(f"   Converting {len(coherence_matrices)} samples with coherence data to numpy array...", flush=True)
        log.info(f"Converting {len(coherence_matrices)} samples to numpy array...")
        coherence_matrices = np.array(coherence_matrices)
        print(f"   ✓ Conversion complete! Array shape: {coherence_matrices.shape}", flush=True)
        
        # Handle any NaN values
        print(f"   Cleaning NaN values...", flush=True)
        coherence_matrices = np.nan_to_num(coherence_matrices, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"   ✓ Cleaned!", flush=True)
        
        log.info(f"Computed coherence matrices shape: {coherence_matrices.shape}")
        print(f"   ✓ Completed coherence matrix computation!", flush=True)
        
        return coherence_matrices
    
    def _build_cnn_model(self, input_shape, n_classes):
        """Build and return PyTorch CNN model.
        
        Parameters
        ----------
        input_shape : tuple
            Shape of input data (n_pairs, nfreqs, n_timepoints)
        n_classes : int
            Number of output classes
            
        Returns
        -------
        model : CoherenceCNN
            PyTorch CNN model
        """
        model = CoherenceCNN(n_classes, input_shape)
        model = model.to(self.device)
        return model
    
    def fit(self, X, y):
        """Fit the CNN classifier using coherence matrices.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_channels, n_timepoints)
            Training EEG data
        y : array-like, shape (n_samples,)
            Target labels
            
        Returns
        -------
        self
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Create label mapping for consistent class indexing
        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_idx = np.array([self.class_to_idx_[c] for c in y])
        
        log.info(f"Fitting CoherenceCNNClassifier with {len(self.classes_)} classes")
        log.info(f"Training data shape: {X.shape}")
        
        # Compute coherence matrices
        print("Computing coherence matrices for training data...", flush=True)
        coherence_matrices = self._compute_coherence_matrices(X)
        
        # Normalize coherence matrices to [0, 1]
        log.info("Normalizing coherence matrices...")
        coherence_min = coherence_matrices.min()
        coherence_max = coherence_matrices.max()
        self.coherence_min_ = coherence_min
        self.coherence_max_ = coherence_max
        
        if coherence_max > coherence_min:
            coherence_matrices = (coherence_matrices - coherence_min) / (coherence_max - coherence_min)
        else:
            coherence_matrices = np.zeros_like(coherence_matrices)
        
        # Convert to PyTorch tensors and prepare data loader
        n_samples = coherence_matrices.shape[0]
        
        # Shape: (n_samples, n_pairs, nfreqs, n_timepoints)
        # Treat n_pairs as channel dimension for CNN
        coherence_tensors = torch.from_numpy(coherence_matrices).float()
        y_tensors = torch.from_numpy(y_idx).long()
        
        dataset = TensorDataset(coherence_tensors, y_tensors)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Build and train CNN model
        log.info(f"Building CNN model with input shape: {(coherence_matrices.shape[1], coherence_matrices.shape[2], coherence_matrices.shape[3])}")
        self.model_ = self._build_cnn_model(
            (coherence_matrices.shape[1], coherence_matrices.shape[2], coherence_matrices.shape[3]),
            n_classes
        )
        
        # Compute class weights if requested
        if self.use_class_weights:
            class_counts = np.bincount(y_idx)
            # Weight inversely proportional to class frequency
            class_weights = torch.from_numpy(1.0 / (class_counts / class_counts.sum())).float()
            class_weights = class_weights / class_weights.sum() * n_classes  # Normalize
            class_weights = class_weights.to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            log.info(f"Using class weights: {class_weights.cpu().numpy()}")
            print(f"   Using class weights: {class_weights.cpu().numpy()}", flush=True)
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
        print("Training CNN model...", flush=True)
        self.model_.train()
        
        print_interval = max(1, self.epochs // 20)  # Print every 5%
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batch_count = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            
            # Print progress every print_interval epochs
            if (epoch + 1) % print_interval == 0 or epoch == 0:
                progress = 100 * (epoch + 1) / self.epochs
                print(f"   Epoch [{epoch+1:3d}/{self.epochs}] ({progress:5.1f}%) | Loss: {avg_loss:.4f}", flush=True)
        
        log.info("CNN model training complete")
        print(f"   ✓ Training complete! Final loss: {avg_loss:.4f}", flush=True)
        return self
    
    def predict(self, X):
        """Predict class labels using the trained CNN model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_channels, n_timepoints)
            Test EEG data
            
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet")
        
        # Compute coherence matrices
        coherence_matrices = self._compute_coherence_matrices(X)
        
        # Normalize using fitted parameters
        if self.coherence_max_ > self.coherence_min_:
            coherence_matrices = (coherence_matrices - self.coherence_min_) / (self.coherence_max_ - self.coherence_min_)
        else:
            coherence_matrices = np.zeros_like(coherence_matrices)
        
        # Convert to PyTorch tensor
        coherence_tensors = torch.from_numpy(coherence_matrices).float()
        coherence_tensors = coherence_tensors.to(self.device)
        
        # Make predictions
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(coherence_tensors)
            y_pred_idx = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # Map back to original class labels
        idx_to_class = {idx: cls for cls, idx in self.class_to_idx_.items()}
        y_pred = np.array([idx_to_class[idx] for idx in y_pred_idx])
        
        return y_pred
    
    def predict_proba(self, X):
        """Predict class probabilities using the trained CNN model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_channels, n_timepoints)
            Test EEG data
            
        Returns
        -------
        proba : array, shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet")
        
        # Compute coherence matrices
        coherence_matrices = self._compute_coherence_matrices(X)
        
        # Normalize using fitted parameters
        if self.coherence_max_ > self.coherence_min_:
            coherence_matrices = (coherence_matrices - self.coherence_min_) / (self.coherence_max_ - self.coherence_min_)
        else:
            coherence_matrices = np.zeros_like(coherence_matrices)
        
        # Convert to PyTorch tensor
        coherence_tensors = torch.from_numpy(coherence_matrices).float()
        coherence_tensors = coherence_tensors.to(self.device)
        
        # Get probabilities
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(coherence_tensors)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()
        
        return proba
    
    def score(self, X, y):
        """Compute the mean accuracy score.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_channels, n_timepoints)
            Test data
        y : array-like, shape (n_samples,)
            Target labels
            
        Returns
        -------
        score : float
            Mean accuracy score
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
