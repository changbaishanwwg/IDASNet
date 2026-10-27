"""Shared input transformation for portable traditional-baseline pipelines."""
import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from data import NORMALIZATION


def normalize_spectra(X: torch.Tensor, epsilon: float = NORMALIZATION["epsilon"]) -> torch.Tensor:
    """Z-score each sample/method spectrum independently over wavelengths.

    X has shape (samples, preprocessing methods, wavelengths). The sample
    standard deviation uses correction=1. No dataset-level statistics are fitted.
    """
    normalized = (X - X.mean(dim=-1, keepdim=True)) / (
        X.std(dim=-1, keepdim=True, correction=NORMALIZATION["std_correction"]) + epsilon
    )
    if not torch.isfinite(normalized).all():
        raise ValueError("Non-finite values after spectral Z-score normalization")
    return normalized



class SamplewiseMethodZScore(TransformerMixin, BaseEstimator):
    """Apply the per-spectrum Z-score before flattening the model input."""

    def __init__(self, n_methods: int = 5, n_wavelengths: int = 151):
        self.n_methods = n_methods
        self.n_wavelengths = n_wavelengths

    def fit(self, X, y=None):
        values = check_array(X)
        if values.shape[1] != self.n_methods * self.n_wavelengths:
            raise ValueError("Feature count does not match methods x wavelengths")
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        values = check_array(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Unexpected input feature count")
        spectra = torch.as_tensor(values, dtype=torch.float32).reshape(
            len(values), self.n_methods, self.n_wavelengths
        )
        normalized = normalize_spectra(spectra)
        return normalized.reshape(len(values), -1).numpy().astype(np.float64)

