"""CWT-based CNN classifier for EEG signal classification.

This module contains a CNN classifier that uses raw wavelet transform features
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


class CWTCNN(nn.Module):
    """PyTorch CNN model for wavelet transform classification."""
    
    def __init__(self, n_classes, input_shape):
        super(CWTCNN, self).__init__()
        n_channels, nfreqs, n_timepoints = input_shape
        
        # Input shape: (batch_size, n_pairs, nfreqs, n_timepoints)
        # Treat n_pairs as the channel dimension
        
        # First convolutional block - Asymmetric kernel for temporal patterns
        # (1, 15) = very narrow in frequency, wide in time to capture long-range temporal dynamics
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=(1, 15), padding=(0, 7))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(0.25)
        
        # Second convolutional block - Standard kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5), padding=(2, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout(0.25)
        
        # Third convolutional block - Standard kernel
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout3 = nn.Dropout(0.25)
        
        # Calculate flattened size after convolutions
        # After each pool operation, dimensions are halved
        flattened_nfreqs = nfreqs // 8
        flattened_ntimepoints = n_timepoints // 8
        flattened_size = 128 * flattened_nfreqs * flattened_ntimepoints
        
        # Global average pooling instead of flattening
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dense layers - much smaller to prevent overfitting
        self.fc1 = nn.Linear(128, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.dropout_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn_fc2 = nn.BatchNorm1d(32)
        self.dropout_fc2 = nn.Dropout(0.5)
        
        self.fc_out = nn.Linear(32, n_classes)
    
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
        
        # Global average pooling
        x = self.global_pool(x)
        
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


class CWTCNNClassifier(BaseEstimator, ClassifierMixin):
    """CNN classifier using wavelet transform features for EEG classification.
    
    This classifier computes wavelet transforms for each channel
    and uses a CNN architecture to learn spatial-temporal patterns from the
    wavelet coefficient matrices.
    
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
    >>> clf = CWTCNNClassifier(lowest=4, highest=40, nfreqs=50)
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
        
    def _compute_wavelet_transforms(self, X):
        """Compute wavelet transforms for all channels.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_channels, n_timepoints)
            EEG data
            
        Returns
        -------
        wavelet_transforms : array-like, shape (n_samples, n_channels, nfreqs, n_timepoints)
            Wavelet transforms for each channel
        """
        n_samples, n_channels, n_timepoints = X.shape
        
        log.info(f"Computing wavelet transforms for {n_samples} samples, {n_channels} channels")
        print(f"   Computing wavelet transforms for {n_channels} channels...", flush=True)
        
        # Compute wavelet transforms for each channel
        log.info("Computing all wavelet transforms...")
        wavelet_transforms = []
        
        for sample_idx in range(n_samples):
            sample_transforms = []
            
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
                    
                    # Compute magnitude (absolute value)
                    coeffs = np.abs(coeffs)
                    sample_transforms.append(coeffs)
                    
                except Exception as e:
                    log.debug(f"Error in wavelet transform for sample {sample_idx}, channel {ch_idx}: {e}")
                    # Fallback: create zero matrix with expected shape (nfreqs, 100)
                    zero_transform = np.zeros((self.nfreqs, 100))
                    sample_transforms.append(zero_transform)
            
            wavelet_transforms.append(sample_transforms)
            
            if (sample_idx + 1) % max(1, n_samples // 10) == 0:
                print(f"     Progress: {sample_idx + 1}/{n_samples} samples processed ({100*(sample_idx+1)/n_samples:.0f}%)", flush=True)
        
        # Convert to numpy array: (n_samples, n_channels, nfreqs, n_timepoints)
        print(f"   Converting {len(wavelet_transforms)} samples to numpy array...", flush=True)
        log.info(f"Converting {len(wavelet_transforms)} samples to numpy array...")
        wavelet_transforms = np.array(wavelet_transforms)
        print(f"   ✓ Conversion complete! Array shape: {wavelet_transforms.shape}", flush=True)
        
        # Handle any NaN values
        print(f"   Cleaning NaN values...", flush=True)
        wavelet_transforms = np.nan_to_num(wavelet_transforms, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"   ✓ Cleaned!", flush=True)
        
        log.info(f"Computed wavelet transforms shape: {wavelet_transforms.shape}")
        print(f"   ✓ Completed wavelet transform computation!", flush=True)
        
        return wavelet_transforms
    
    def _build_cnn_model(self, input_shape, n_classes):
        """Build and return PyTorch CNN model.
        
        Parameters
        ----------
        input_shape : tuple
            Shape of input data (n_channels, nfreqs, n_timepoints)
        n_classes : int
            Number of output classes
            
        Returns
        -------
        model : CWTCNN
            PyTorch CNN model
        """
        model = CWTCNN(n_classes, input_shape)
        model = model.to(self.device)
        return model
    
    def fit(self, X, y):
        """Fit the CNN classifier using wavelet transforms.
        
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
        
        log.info(f"Fitting CWTCNNClassifier with {len(self.classes_)} classes")
        log.info(f"Training data shape: {X.shape}")
        
        # Compute wavelet transforms
        print("Computing wavelet transforms for training data...", flush=True)
        wavelet_transforms = self._compute_wavelet_transforms(X)
        
        # Normalize wavelet transforms to [0, 1]
        log.info("Normalizing wavelet transforms...")
        transform_min = wavelet_transforms.min()
        transform_max = wavelet_transforms.max()
        self.transform_min_ = transform_min
        self.transform_max_ = transform_max
        
        if transform_max > transform_min:
            wavelet_transforms = (wavelet_transforms - transform_min) / (transform_max - transform_min)
        else:
            wavelet_transforms = np.zeros_like(wavelet_transforms)
        
        # Convert to PyTorch tensors and prepare data loader
        n_samples = wavelet_transforms.shape[0]
        
        # Shape: (n_samples, n_channels, nfreqs, n_timepoints)
        # Treat n_channels as channel dimension for CNN
        transform_tensors = torch.from_numpy(wavelet_transforms).float()
        y_tensors = torch.from_numpy(y_idx).long()
        
        dataset = TensorDataset(transform_tensors, y_tensors)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Build and train CNN model
        log.info(f"Building CNN model with input shape: {(wavelet_transforms.shape[1], wavelet_transforms.shape[2], wavelet_transforms.shape[3])}")
        self.model_ = self._build_cnn_model(
            (wavelet_transforms.shape[1], wavelet_transforms.shape[2], wavelet_transforms.shape[3]),
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
        
        # Compute wavelet transforms
        wavelet_transforms = self._compute_wavelet_transforms(X)
        
        # Normalize using fitted parameters
        if self.transform_max_ > self.transform_min_:
            wavelet_transforms = (wavelet_transforms - self.transform_min_) / (self.transform_max_ - self.transform_min_)
        else:
            wavelet_transforms = np.zeros_like(wavelet_transforms)
        
        # Convert to PyTorch tensor
        transform_tensors = torch.from_numpy(wavelet_transforms).float()
        transform_tensors = transform_tensors.to(self.device)
        
        # Make predictions
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(transform_tensors)
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
        
        # Compute wavelet transforms
        wavelet_transforms = self._compute_wavelet_transforms(X)
        
        # Normalize using fitted parameters
        if self.transform_max_ > self.transform_min_:
            wavelet_transforms = (wavelet_transforms - self.transform_min_) / (self.transform_max_ - self.transform_min_)
        else:
            wavelet_transforms = np.zeros_like(wavelet_transforms)
        
        # Convert to PyTorch tensor
        transform_tensors = torch.from_numpy(wavelet_transforms).float()
        transform_tensors = transform_tensors.to(self.device)
        
        # Get probabilities
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(transform_tensors)
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
