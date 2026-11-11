"""
Calibrated Neural Network (MLP + Temperature Scaling) and Logistic Regression Baseline
for MNIST Handwritten Digit Classification (Digits 1-9).

Implements:
- 2-hidden-layer Neural Network (784 -> 256 -> 128 -> 9) trained on MNIST digits 1..9.
- Post-hoc Temperature Scaling (Guo et al., 2017) fitted on the 10,000-image validation
  split to calibrate output probabilities before converting them to search step costs.
- Multinomial Logistic Regression baseline trained on the identical training split.
- Evaluation metrics: Accuracy, 9x9 Confusion Matrix, Expected Calibration Error (ECE),
  Negative Log-Likelihood (NLL), and Brier Score on the untouched test set.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

from data.mnist_loader import MNISTSubset, load_processed_mnist


@dataclass
class ClassifierMetrics:
    """Comprehensive test metrics for a 9-class digit classifier (digits 1..9)."""

    model_name: str
    accuracy: float
    ece_uncalibrated: float
    ece_calibrated: float
    nll_uncalibrated: float
    nll_calibrated: float
    brier_score: float
    temperature: float
    confusion_matrix: List[List[int]]


def compute_calibration_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    num_bins: int = 15,
) -> Tuple[float, float, float]:
    """
    Compute Expected Calibration Error (ECE), Negative Log-Likelihood (NLL),
    and multi-class Brier Score.

    Args:
      probs: (N, 9) array of predicted probabilities for digits 1..9.
      y_true: (N,) array of true digit labels in 1..9.
    """
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    n = len(y_true)
    true_idx = y_true - 1  # 0..8

    # Negative Log-Likelihood
    nll = float(-np.mean(np.log(probs[np.arange(n), true_idx])))

    # Multi-class Brier Score
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), true_idx] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    # Expected Calibration Error (ECE)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1) + 1
    accuracies = (predictions == y_true).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == 0:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences > low) & (confidences <= high)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += float(np.abs(acc_in_bin - conf_in_bin) * prop_in_bin)

    return ece, nll, brier


class CalibratedDigitClassifier:
    """
    Wraps a trained classifier (MLP or LogisticRegression) with optional
    post-hoc Temperature Scaling learned on the validation set.
    """

    def __init__(
        self,
        base_model,
        model_name: str = "CalibratedMLP",
        temperature: float = 1.0,
    ) -> None:
        self.base_model = base_model
        self.model_name = model_name
        self.temperature = float(temperature)

    def _compute_raw_logits(self, X_flat: np.ndarray) -> np.ndarray:
        """Compute raw pre-softmax logits (N, 9) from the underlying model."""
        if isinstance(self.base_model, MLPClassifier):
            # Forward pass through MLP layers up to pre-softmax output
            activation = X_flat
            for i in range(len(self.base_model.coefs_) - 1):
                activation = activation @ self.base_model.coefs_[i] + self.base_model.intercepts_[i]
                # ReLU hidden activation
                np.maximum(activation, 0, out=activation)
            logits = (
                activation @ self.base_model.coefs_[-1]
                + self.base_model.intercepts_[-1]
            )
            return logits
        elif isinstance(self.base_model, LogisticRegression):
            return self.base_model.decision_function(X_flat)
        else:
            probs = self.base_model.predict_proba(X_flat)
            return np.log(np.clip(probs, 1e-12, 1.0))

    def fit_temperature(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Optimize scalar temperature T > 0 on the validation set to minimize NLL.
        """
        X_flat = X_val.reshape(len(X_val), -1).astype(np.float64)
        logits = self._compute_raw_logits(X_flat)
        true_idx = y_val - 1

        def nll_objective(temp: float) -> float:
            scaled = logits / max(temp, 1e-3)
            log_probs = log_softmax(scaled, axis=1)
            return float(-np.mean(log_probs[np.arange(len(y_val)), true_idx]))

        res = minimize_scalar(nll_objective, bounds=(0.2, 5.0), method="bounded")
        self.temperature = float(res.x)
        return self.temperature

    def predict_proba(
        self,
        X: np.ndarray,
        use_temperature: bool = True,
    ) -> np.ndarray:
        """
        Return calibrated probability distribution over digits 1..9 of shape (N, 9).
        Accepts X of shape (N, 28, 28), (N, 784), or single image (28, 28).
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 2 and arr.shape == (28, 28):
            arr = arr.reshape(1, 784)
        elif arr.ndim == 3:
            arr = arr.reshape(len(arr), -1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)

        logits = self._compute_raw_logits(arr)
        temp = self.temperature if use_temperature else 1.0
        probs = softmax(logits / max(temp, 1e-3), axis=1)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted digit labels in 1..9."""
        probs = self.predict_proba(X, use_temperature=True)
        return (np.argmax(probs, axis=1) + 1).astype(np.int64)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> ClassifierMetrics:
        """Evaluate accuracy, confusion matrix, and calibration before/after scaling."""
        probs_uncal = self.predict_proba(X_test, use_temperature=False)
        probs_cal = self.predict_proba(X_test, use_temperature=True)
        preds = np.argmax(probs_cal, axis=1) + 1

        acc = float(accuracy_score(y_test, preds))
        ece_uncal, nll_uncal, _ = compute_calibration_metrics(probs_uncal, y_test)
        ece_cal, nll_cal, brier_cal = compute_calibration_metrics(probs_cal, y_test)
        cm = confusion_matrix(y_test, preds, labels=list(range(1, 10))).tolist()

        return ClassifierMetrics(
            model_name=self.model_name,
            accuracy=acc,
            ece_uncalibrated=ece_uncal,
            ece_calibrated=ece_cal,
            nll_uncalibrated=nll_uncal,
            nll_calibrated=nll_cal,
            brier_score=brier_cal,
            temperature=self.temperature,
            confusion_matrix=cm,
        )


def train_and_evaluate_classifiers(
    dataset: Optional[MNISTSubset] = None,
    models_dir: Path = Path("models"),
    results_dir: Path = Path("results"),
    seed: int = 42,
) -> Tuple[CalibratedDigitClassifier, CalibratedDigitClassifier, Dict[str, ClassifierMetrics]]:
    """
    Train both the Neural Network (MLP + Temperature Scaling) and the Logistic Regression
    baseline on the MNIST 1..9 training split, calibrate on the validation split, and
    evaluate on the untouched test split.
    """
    if dataset is None:
        dataset = load_processed_mnist(seed=seed)

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    X_tr_flat = dataset.X_train.reshape(len(dataset.X_train), -1)
    y_tr = dataset.y_train

    # 1. Train Neural Network (2-layer MLP: 784 -> 256 -> 128 -> 9)
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=25,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=seed,
    )
    mlp.fit(X_tr_flat, y_tr)

    nn_clf = CalibratedDigitClassifier(base_model=mlp, model_name="Neural Network (MLP + TempScaling)")
    nn_clf.fit_temperature(dataset.X_val, dataset.y_val)
    nn_metrics = nn_clf.evaluate(dataset.X_test, dataset.y_test)

    # 2. Train Baseline (Multinomial Logistic Regression)
    logreg = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=150,
        random_state=seed,
    )
    logreg.fit(X_tr_flat, y_tr)

    lr_clf = CalibratedDigitClassifier(base_model=logreg, model_name="Logistic Regression Baseline")
    lr_clf.fit_temperature(dataset.X_val, dataset.y_val)
    lr_metrics = lr_clf.evaluate(dataset.X_test, dataset.y_test)

    # Save models and metrics
    joblib.dump(nn_clf, models_dir / "calibrated_nn.joblib")
    joblib.dump(lr_clf, models_dir / "logreg_baseline.joblib")

    metrics_dict = {
        "nn": nn_metrics,
        "baseline_logreg": lr_metrics,
    }
    report_payload = {
        "seed": seed,
        "nn": asdict(nn_metrics),
        "baseline_logreg": asdict(lr_metrics),
    }
    (results_dir / "nn_vs_baseline_metrics.json").write_text(
        json.dumps(report_payload, indent=2), encoding="utf-8"
    )

    return nn_clf, lr_clf, metrics_dict


def load_or_train_models(
    dataset: Optional[MNISTSubset] = None,
    models_dir: Path = Path("models"),
    results_dir: Path = Path("results"),
    seed: int = 42,
) -> Tuple[CalibratedDigitClassifier, CalibratedDigitClassifier, Dict[str, ClassifierMetrics]]:
    """Load saved NN and Baseline models from disk, or train them if not yet saved."""
    nn_path = models_dir / "calibrated_nn.joblib"
    lr_path = models_dir / "logreg_baseline.joblib"
    metrics_path = results_dir / "nn_vs_baseline_metrics.json"

    if nn_path.exists() and lr_path.exists() and metrics_path.exists():
        nn_clf = joblib.load(nn_path)
        lr_clf = joblib.load(lr_path)
        raw_m = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = {
            "nn": ClassifierMetrics(**raw_m["nn"]),
            "baseline_logreg": ClassifierMetrics(**raw_m["baseline_logreg"]),
        }
        return nn_clf, lr_clf, metrics

    return train_and_evaluate_classifiers(
        dataset=dataset, models_dir=models_dir, results_dir=results_dir, seed=seed
    )

