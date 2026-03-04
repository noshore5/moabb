"""EEGNet classifier for EEG signal classification.

Implementation of EEGNet from Lawhern et al. 2018:
"EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces"
https://arxiv.org/abs/1611.08024

This is an EEG-specific architecture optimized for small datasets with ~4,000 parameters.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset


log = logging.getLogger(__name__)


class EEGNetModel(nn.Module):
    """PyTorch EEGNet model for EEG classification."""

    def __init__(self, n_classes, n_channels=22, n_timepoints=1001, dropout_rate=0.5):
        super(EEGNetModel, self).__init__()

        F1 = 8  # Number of temporal filters
        F2 = 16  # Number of pointwise filters
        D = 2  # Depth multiplier for depthwise convolution

        # Block 1: Temporal convolution
        # (1, 51) kernel captures low-frequency temporal patterns
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, 51), padding=(0, 25), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 2: Depthwise convolution (spatial filters)
        # Learns spatial relationships between channels for each frequency band
        self.depthwise = nn.Conv2d(
            F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU(alpha=1.0)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 3: Separable convolution (pointwise + depthwise)
        # Efficient feature extraction
        self.sep_conv = nn.Conv2d(
            F1 * D, F2, kernel_size=(1, 15), padding=(0, 7), bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification layer
        self.fc = nn.Linear(F2, n_classes)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch_size, n_channels, n_timepoints)

        Returns
        -------
        output : torch.Tensor
            Logits of shape (batch_size, n_classes)
        """
        # Add channel dimension for conv2d: (batch, 1, n_channels, n_timepoints)
        x = x.unsqueeze(1)

        # Block 1: Temporal convolution
        x = self.conv1(x)
        x = self.bn1(x)

        # Block 2: Depthwise + spatial
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 3: Separable convolution
        x = self.sep_conv(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Global average pooling
        x = self.global_pool(x)

        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class EEGNetClassifier(BaseEstimator, ClassifierMixin):
    """EEGNet classifier for EEG-based classification tasks.

    EEGNet is a compact convolutional neural network designed for EEG signals.
    It uses temporal, depthwise, and separable convolutions to efficiently
    learn spatial-temporal patterns with minimal parameters (~4,000).

    Parameters
    ----------
    n_channels : int, default=22
        Number of EEG channels

    n_timepoints : int, default=1001
        Number of timepoints per sample

    epochs : int, default=100
        Number of training epochs

    batch_size : int, default=32
        Batch size for training

    learning_rate : float, default=0.001
        Learning rate for Adam optimizer

    dropout_rate : float, default=0.5
        Dropout rate for regularization

    device : str, default='cpu'
        Device to use for training ('cpu' or 'cuda')

    verbose : int, default=0
        Verbosity level

    Examples
    --------
    >>> clf = EEGNetClassifier(n_channels=22, n_timepoints=1001)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """

    def __init__(
        self,
        n_channels=22,
        n_timepoints=1001,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        dropout_rate=0.5,
        device="cpu",
        verbose=0,
    ):
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.device = device if torch.cuda.is_available() else "cpu"
        self.verbose = verbose
        self.model_ = None
        self.classes_ = None
        self.class_to_idx_ = None

    def fit(self, X, y):
        """Fit the EEGNet classifier.

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

        # Create label mapping
        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_idx = np.array([self.class_to_idx_[c] for c in y])

        log.info(f"Fitting EEGNetClassifier with {n_classes} classes")
        log.info(f"Training data shape: {X.shape}")

        # Normalize input data
        X_mean = X.mean()
        X_std = X.std()
        self.X_mean_ = X_mean
        self.X_std_ = X_std

        X_normalized = (X - X_mean) / (X_std + 1e-8)

        # Convert to PyTorch tensors
        X_tensor = torch.from_numpy(X_normalized).float()
        y_tensor = torch.from_numpy(y_idx).long()

        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build model
        self.model_ = EEGNetModel(
            n_classes=n_classes,
            n_channels=self.n_channels,
            n_timepoints=self.n_timepoints,
            dropout_rate=self.dropout_rate,
        )
        self.model_ = self.model_.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model_.parameters())
        log.info(f"EEGNet total parameters: {total_params}")
        print(f"   EEGNet total parameters: {total_params}", flush=True)

        # Training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        print("   Training EEGNet...", flush=True)
        self.model_.train()

        print_interval = max(1, self.epochs // 20)

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

            if (epoch + 1) % print_interval == 0 or epoch == 0:
                progress = 100 * (epoch + 1) / self.epochs
                print(
                    f"   Epoch [{epoch+1:3d}/{self.epochs}] ({progress:5.1f}%) | Loss: {avg_loss:.4f}",
                    flush=True,
                )

        log.info("EEGNet training complete")
        print(f"   ✓ Training complete! Final loss: {avg_loss:.4f}", flush=True)

        return self

    def predict(self, X):
        """Predict class labels.

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

        # Normalize using training statistics
        X_normalized = (X - self.X_mean_) / (self.X_std_ + 1e-8)

        X_tensor = torch.from_numpy(X_normalized).float()
        X_tensor = X_tensor.to(self.device)

        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            y_pred_idx = torch.argmax(outputs, dim=1).cpu().numpy()

        # Map back to original labels
        idx_to_class = {idx: cls for cls, idx in self.class_to_idx_.items()}
        y_pred = np.array([idx_to_class[idx] for idx in y_pred_idx])

        return y_pred

    def predict_proba(self, X):
        """Predict class probabilities.

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

        # Normalize using training statistics
        X_normalized = (X - self.X_mean_) / (self.X_std_ + 1e-8)

        X_tensor = torch.from_numpy(X_normalized).float()
        X_tensor = X_tensor.to(self.device)

        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()

        return proba

    def score(self, X, y):
        """Compute mean accuracy score.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_channels, n_timepoints)
            Test data
        y : array-like, shape (n_samples,)
            Target labels

        Returns
        -------
        score : float
            Mean accuracy
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
