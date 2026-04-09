"""
Disease Classification Models
Train and manage ML models for disease detection.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix
)

from .mimic_loader import MIMICLoader, get_mimic_loader

logger = logging.getLogger(__name__)

# Default model storage path
MODELS_PATH = Path(__file__).parent.parent / "models"


class DiseaseModel:
    """
    A trained disease classification model.
    """

    def __init__(
        self,
        disease_icd: str,
        disease_name: str,
        model: Any,
        scaler: StandardScaler,
        imputer: SimpleImputer,
        feature_names: List[str],
        metrics: Dict,
        threshold: float = 0.5
    ):
        self.disease_icd = disease_icd
        self.disease_name = disease_name
        self.model = model
        self.scaler = scaler
        self.imputer = imputer
        self.feature_names = feature_names
        self.metrics = metrics
        self.threshold = threshold
        self.created_at = datetime.utcnow().isoformat()

    def predict(self, features: Dict[str, float]) -> Dict:
        """
        Predict disease probability for a patient.

        Args:
            features: Dict of lab name -> value

        Returns:
            Dict with prediction results
        """
        # Build feature vector
        X = np.array([[features.get(f, np.nan) for f in self.feature_names]])

        # Impute missing values
        X = self.imputer.transform(X)

        # Scale features
        X = self.scaler.transform(X)

        # Predict
        prob = self.model.predict_proba(X)[0, 1]
        prediction = 1 if prob >= self.threshold else 0

        return {
            'disease_icd': self.disease_icd,
            'disease_name': self.disease_name,
            'probability': float(prob),
            'prediction': prediction,
            'confidence': float(prob) if prediction == 1 else float(1 - prob),
            'threshold': self.threshold,
            'source': 'ML_MIMIC'
        }

    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict for multiple patients."""
        # Ensure columns match
        X = features_df.reindex(columns=self.feature_names)

        # Impute and scale
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)

        # Predict
        probs = self.model.predict_proba(X)[:, 1]

        return pd.DataFrame({
            'probability': probs,
            'prediction': (probs >= self.threshold).astype(int)
        })

    def save(self, path: Path):
        """Save model to disk."""
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(self.model, path / "model.joblib")
        joblib.dump(self.scaler, path / "scaler.joblib")
        joblib.dump(self.imputer, path / "imputer.joblib")

        # Save metadata
        metadata = {
            'disease_icd': self.disease_icd,
            'disease_name': self.disease_name,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'threshold': self.threshold,
            'created_at': self.created_at
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: Path) -> 'DiseaseModel':
        """Load model from disk."""
        model = joblib.load(path / "model.joblib")
        scaler = joblib.load(path / "scaler.joblib")
        imputer = joblib.load(path / "imputer.joblib")

        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        return cls(
            disease_icd=metadata['disease_icd'],
            disease_name=metadata['disease_name'],
            model=model,
            scaler=scaler,
            imputer=imputer,
            feature_names=metadata['feature_names'],
            metrics=metadata['metrics'],
            threshold=metadata['threshold']
        )


class DiseaseModelManager:
    """
    Manage disease classification models.
    """

    def __init__(self, models_path: Path = MODELS_PATH):
        self.models_path = models_path
        self.models: Dict[str, DiseaseModel] = {}
        self.mimic_loader: Optional[MIMICLoader] = None

    def _get_mimic(self) -> MIMICLoader:
        """Get MIMIC loader (lazy load)."""
        if self.mimic_loader is None:
            self.mimic_loader = get_mimic_loader()
        return self.mimic_loader

    def train_model(
        self,
        disease_icd: str,
        disease_name: str = None,
        max_patients: int = 5000,
        model_type: str = 'gradient_boosting',
        test_size: float = 0.2
    ) -> DiseaseModel:
        """
        Train a model for a specific disease.

        Args:
            disease_icd: ICD code prefix (e.g., 'E11' for Type 2 Diabetes)
            disease_name: Human-readable name
            max_patients: Maximum patients per class
            model_type: 'gradient_boosting', 'random_forest', or 'logistic'
            test_size: Fraction for test set

        Returns:
            Trained DiseaseModel
        """
        logger.info(f"Training model for {disease_icd} ({disease_name})...")

        mimic = self._get_mimic()

        # Prepare data
        X, y = mimic.prepare_training_data(disease_icd, max_patients)

        if X.empty:
            raise ValueError(f"No training data found for {disease_icd}")

        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, stratify=y, random_state=42
        )

        # Create model
        if model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)

        # Train
        logger.info("Training model...")
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': float((y_pred == y_test).mean()),
            'roc_auc': float(roc_auc_score(y_test, y_prob)),
            'avg_precision': float(average_precision_score(y_test, y_prob)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'positive_rate': float(y.mean())
        }

        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        logger.info(f"Model trained - AUC: {metrics['roc_auc']:.3f}, "
                    f"Accuracy: {metrics['accuracy']:.3f}")

        # Create disease model
        disease_model = DiseaseModel(
            disease_icd=disease_icd,
            disease_name=disease_name or disease_icd,
            model=model,
            scaler=scaler,
            imputer=imputer,
            feature_names=list(X.columns),
            metrics=metrics,
            threshold=float(optimal_threshold)
        )

        # Save model
        model_path = self.models_path / disease_icd.replace('.', '_')
        disease_model.save(model_path)

        # Cache in memory
        self.models[disease_icd] = disease_model

        return disease_model

    def train_common_diseases(
        self,
        min_patients: int = 2000,
        max_diseases: int = 20
    ) -> List[DiseaseModel]:
        """
        Train models for the most common diseases.

        Args:
            min_patients: Minimum patients for a disease to be included
            max_diseases: Maximum number of diseases to train

        Returns:
            List of trained models
        """
        mimic = self._get_mimic()

        # Get common diseases
        common = mimic.get_common_diseases(min_patients)[:max_diseases]

        logger.info(f"Training models for {len(common)} common diseases...")

        trained = []
        for disease in common:
            try:
                model = self.train_model(
                    disease_icd=disease['icd_code'],
                    disease_name=disease['name'],
                    max_patients=5000
                )
                trained.append(model)
            except Exception as e:
                logger.error(f"Failed to train {disease['icd_code']}: {e}")

        logger.info(f"Successfully trained {len(trained)} models")
        return trained

    def load_models(self) -> int:
        """Load all saved models from disk."""
        if not self.models_path.exists():
            return 0

        count = 0
        for model_dir in self.models_path.iterdir():
            if model_dir.is_dir() and (model_dir / "metadata.json").exists():
                try:
                    model = DiseaseModel.load(model_dir)
                    self.models[model.disease_icd] = model
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to load model {model_dir}: {e}")

        logger.info(f"Loaded {count} models from {self.models_path}")
        return count

    def predict(self, features: Dict[str, float]) -> List[Dict]:
        """
        Run all models on patient features.

        Args:
            features: Dict of lab name -> value

        Returns:
            List of predictions from all models
        """
        predictions = []

        for disease_icd, model in self.models.items():
            try:
                pred = model.predict(features)
                if pred['probability'] >= 0.3:  # Only return significant predictions
                    predictions.append(pred)
            except Exception as e:
                logger.warning(f"Prediction failed for {disease_icd}: {e}")

        # Sort by probability
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        return predictions

    def get_model(self, disease_icd: str) -> Optional[DiseaseModel]:
        """Get a specific model."""
        return self.models.get(disease_icd)

    def list_models(self) -> List[Dict]:
        """List all available models."""
        return [
            {
                'disease_icd': m.disease_icd,
                'disease_name': m.disease_name,
                'metrics': m.metrics,
                'threshold': m.threshold
            }
            for m in self.models.values()
        ]


# Singleton instance
_model_manager: Optional[DiseaseModelManager] = None


def get_model_manager() -> DiseaseModelManager:
    """Get singleton model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = DiseaseModelManager()
        _model_manager.load_models()
    return _model_manager
