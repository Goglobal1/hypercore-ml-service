# ============================================================================
# HYPERCORE BULLETPROOF SAFETY LAYER
# ============================================================================
# Version: 5.19.0 (Major safety overhaul)
# Purpose: ZERO FAILURES - Universal protection for ALL 73 endpoints
# Target: 100+ hospital pilots, 1000s of computers, mission-critical healthcare
# ============================================================================
#
# THIS IS NOT A PATCH. THIS IS AN ARCHITECTURE.
#
# Every endpoint gets wrapped in the same bulletproof safety layer.
# No exceptions. No edge cases. It either works or returns a helpful error.
# It NEVER crashes. NEVER returns 500. NEVER leaves users confused.
#
# ============================================================================


"""
HYPERCORE SAFETY LAYER (ml_safety.py)

Add this file to the project and import it in main.py.
Then wrap ALL endpoints with the safety decorator.
"""

import numpy as np
import pandas as pd
import io
import traceback
import logging
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Callable
from fastapi import HTTPException
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hypercore_safety")


# ============================================================================
# SECTION 1: UNIVERSAL SAFETY DECORATOR
# ============================================================================
# This wraps EVERY endpoint and catches ALL exceptions

def bulletproof_endpoint(endpoint_name: str, min_rows: int = 1):
    """
    Universal safety decorator for ALL HyperCore endpoints.

    Usage:
        @app.post("/analyze")
        @bulletproof_endpoint("analyze", min_rows=5)
        def analyze(request):
            ...
    """
    def decorator(func: Callable):
        import asyncio

        @wraps(func)
        async def wrapper(*args, **kwargs):
            request_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")

            try:
                logger.info(f"[{request_id}] Starting {endpoint_name}")
                # Handle both sync and async functions
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                logger.info(f"[{request_id}] Completed {endpoint_name} successfully")
                return result

            except HTTPException as e:
                # Re-raise FastAPI HTTP exceptions (these are intentional)
                logger.warning(f"[{request_id}] {endpoint_name} returned {e.status_code}: {e.detail}")
                raise

            except ValueError as e:
                logger.error(f"[{request_id}] {endpoint_name} ValueError: {str(e)}")
                return JSONResponse(
                    status_code=422,
                    content=create_error_response(
                        error_type="validation_error",
                        message=str(e),
                        endpoint=endpoint_name,
                        request_id=request_id,
                        suggestions=["Check your input data format", "Ensure all required columns exist"]
                    )
                )

            except pd.errors.EmptyDataError:
                logger.error(f"[{request_id}] {endpoint_name} received empty data")
                return JSONResponse(
                    status_code=422,
                    content=create_error_response(
                        error_type="empty_data",
                        message="The provided CSV data is empty",
                        endpoint=endpoint_name,
                        request_id=request_id,
                        suggestions=["Provide a CSV with at least one row of data"]
                    )
                )

            except Exception as e:
                # CATCH EVERYTHING - Never let an exception become a 500 error
                error_trace = traceback.format_exc()
                logger.error(f"[{request_id}] {endpoint_name} UNEXPECTED ERROR: {str(e)}\n{error_trace}")

                # Classify the error and provide helpful response
                error_response = classify_and_respond(e, endpoint_name, request_id)
                return JSONResponse(status_code=422, content=error_response)

        return wrapper
    return decorator


def create_error_response(error_type: str, message: str, endpoint: str,
                         request_id: str, suggestions: List[str] = None,
                         details: Dict = None) -> Dict:
    """Create standardized error response."""
    return {
        "status": "error",
        "error": {
            "type": error_type,
            "message": message,
            "endpoint": endpoint,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat()
        },
        "suggestions": suggestions or [],
        "details": details or {},
        "support": "If this error persists, contact support with the request_id"
    }


def classify_and_respond(exception: Exception, endpoint: str, request_id: str) -> Dict:
    """Classify exception and return helpful error response."""
    error_str = str(exception).lower()
    error_type = type(exception).__name__

    # Pattern matching for common errors
    if "not enough values to unpack" in error_str:
        return create_error_response(
            error_type="insufficient_class_variety",
            message="Data doesn't have enough variety in outcomes for analysis",
            endpoint=endpoint,
            request_id=request_id,
            suggestions=[
                "Ensure your data has BOTH positive (1) and negative (0) outcomes",
                "Provide at least 5 samples of each outcome class",
                "Check that your label column contains varied values"
            ]
        )

    elif "could not convert string to float" in error_str:
        return create_error_response(
            error_type="data_type_error",
            message="Some numeric columns contain non-numeric values",
            endpoint=endpoint,
            request_id=request_id,
            suggestions=[
                "Check for text values in numeric columns",
                "Remove or encode categorical variables",
                "Ensure numbers don't have commas or currency symbols"
            ]
        )

    elif "key" in error_str and "not found" in error_str:
        return create_error_response(
            error_type="missing_column",
            message="A required column was not found in your data",
            endpoint=endpoint,
            request_id=request_id,
            suggestions=[
                "Check that all required column names are spelled correctly",
                "Column names are case-sensitive",
                "Verify your CSV has headers in the first row"
            ]
        )

    elif "division by zero" in error_str or "divide by zero" in error_str:
        return create_error_response(
            error_type="calculation_error",
            message="Unable to calculate metrics due to data distribution",
            endpoint=endpoint,
            request_id=request_id,
            suggestions=[
                "Ensure data has variety (not all same values)",
                "Check for columns with all zeros",
                "Provide more samples"
            ]
        )

    elif "memory" in error_str:
        return create_error_response(
            error_type="resource_limit",
            message="Dataset too large for processing",
            endpoint=endpoint,
            request_id=request_id,
            suggestions=[
                "Reduce dataset size",
                "Remove unnecessary columns",
                "Split into smaller batches"
            ]
        )

    elif "timeout" in error_str:
        return create_error_response(
            error_type="timeout",
            message="Analysis took too long to complete",
            endpoint=endpoint,
            request_id=request_id,
            suggestions=[
                "Reduce dataset size",
                "Reduce number of features",
                "Try again - server may have been busy"
            ]
        )

    else:
        # Generic fallback - still helpful
        return create_error_response(
            error_type=error_type,
            message=f"Analysis could not be completed: {str(exception)[:200]}",
            endpoint=endpoint,
            request_id=request_id,
            suggestions=[
                "Verify your data format matches the expected schema",
                "Ensure all required fields are provided",
                "Check that numeric columns contain only numbers",
                "Try with a smaller dataset first to verify format"
            ]
        )


# ============================================================================
# SECTION 2: UNIVERSAL INPUT VALIDATION
# ============================================================================
# Validate ALL inputs before any processing begins

class DataValidator:
    """Universal data validation for all endpoints."""

    @staticmethod
    def validate_csv(csv_string: str, min_rows: int = 1, max_rows: int = 1000000,
                    required_columns: List[str] = None) -> Tuple[pd.DataFrame, Optional[Dict]]:
        """
        Validate and parse CSV data.
        Returns (DataFrame, None) on success or (None, error_dict) on failure.
        """
        # Check if empty
        if not csv_string or len(csv_string.strip()) == 0:
            return None, {
                "error": "empty_csv",
                "message": "CSV data is empty",
                "suggestion": "Provide CSV data with headers and at least one row"
            }

        # Try to parse
        try:
            df = pd.read_csv(io.StringIO(csv_string))
        except pd.errors.EmptyDataError:
            return None, {
                "error": "empty_csv",
                "message": "CSV contains no data",
                "suggestion": "Provide CSV with headers and data rows"
            }
        except pd.errors.ParserError as e:
            return None, {
                "error": "csv_parse_error",
                "message": f"Could not parse CSV: {str(e)[:100]}",
                "suggestion": "Check CSV format - ensure consistent columns and proper escaping"
            }
        except Exception as e:
            return None, {
                "error": "csv_error",
                "message": f"CSV error: {str(e)[:100]}",
                "suggestion": "Verify CSV is properly formatted"
            }

        # Check row count
        if len(df) < min_rows:
            return None, {
                "error": "insufficient_rows",
                "message": f"Need at least {min_rows} rows, got {len(df)}",
                "suggestion": f"Provide dataset with at least {min_rows} rows"
            }

        if len(df) > max_rows:
            return None, {
                "error": "too_many_rows",
                "message": f"Maximum {max_rows} rows allowed, got {len(df)}",
                "suggestion": f"Reduce dataset to under {max_rows} rows"
            }

        # Check required columns
        if required_columns:
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                return None, {
                    "error": "missing_columns",
                    "message": f"Missing required columns: {missing}",
                    "available_columns": df.columns.tolist(),
                    "suggestion": "Check column names match exactly (case-sensitive)"
                }

        return df, None

    @staticmethod
    def validate_label_column(df: pd.DataFrame, label_column: str,
                             require_binary: bool = True,
                             min_per_class: int = 2) -> Optional[Dict]:
        """Validate label column for classification tasks."""

        if label_column not in df.columns:
            return {
                "error": "label_column_not_found",
                "message": f"Label column '{label_column}' not found",
                "available_columns": df.columns.tolist()
            }

        labels = df[label_column].dropna()

        if len(labels) == 0:
            return {
                "error": "empty_label_column",
                "message": f"Label column '{label_column}' has no values",
                "suggestion": "Ensure label column contains outcome values"
            }

        unique_labels = labels.unique()

        if require_binary and len(unique_labels) < 2:
            return {
                "error": "single_class",
                "message": f"Label column has only one unique value: {unique_labels[0]}",
                "suggestion": "Data must contain BOTH positive (1) and negative (0) outcomes"
            }

        # Check minimum samples per class
        class_counts = labels.value_counts()
        for cls, count in class_counts.items():
            if count < min_per_class:
                return {
                    "error": "insufficient_class_samples",
                    "message": f"Class '{cls}' has only {count} samples, need at least {min_per_class}",
                    "class_distribution": class_counts.to_dict(),
                    "suggestion": f"Provide at least {min_per_class} samples for each outcome class"
                }

        return None  # Valid

    @staticmethod
    def validate_treatment_column(df: pd.DataFrame, treatment_column: str,
                                  min_per_arm: int = 5) -> Optional[Dict]:
        """Validate treatment column for clinical trial endpoints."""

        if treatment_column not in df.columns:
            return {
                "error": "treatment_column_not_found",
                "message": f"Treatment column '{treatment_column}' not found",
                "available_columns": df.columns.tolist()
            }

        arms = df[treatment_column].dropna()
        unique_arms = arms.unique()

        if len(unique_arms) < 2:
            return {
                "error": "single_treatment_arm",
                "message": f"Need at least 2 treatment arms, found: {unique_arms.tolist()}",
                "suggestion": "Data should have both treatment and control groups"
            }

        arm_counts = arms.value_counts()
        for arm, count in arm_counts.items():
            if count < min_per_arm:
                return {
                    "error": "insufficient_arm_size",
                    "message": f"Arm '{arm}' has only {count} samples, need at least {min_per_arm}",
                    "arm_distribution": arm_counts.to_dict()
                }

        return None  # Valid

    @staticmethod
    def validate_numeric_columns(df: pd.DataFrame, columns: List[str]) -> Optional[Dict]:
        """Validate that specified columns are numeric."""
        non_numeric = []

        for col in columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    # Try to convert
                    try:
                        pd.to_numeric(df[col], errors='raise')
                    except:
                        non_numeric.append(col)

        if non_numeric:
            return {
                "error": "non_numeric_columns",
                "message": f"These columns should be numeric but contain non-numeric values: {non_numeric}",
                "suggestion": "Remove text values or encode categorical variables"
            }

        return None  # Valid


# ============================================================================
# SECTION 3: SAFE ML OPERATIONS
# ============================================================================
# Every ML operation is wrapped in safety handlers

class SafeML:
    """Safe wrappers for all ML operations."""

    @staticmethod
    def safe_train_test_split(X, y, test_size=0.2, random_state=42):
        """
        Train/test split that NEVER fails.
        Always returns usable data or clear error.
        """
        from sklearn.model_selection import train_test_split

        try:
            # Validate inputs
            if X is None or len(X) == 0:
                return None, None, None, None, {"error": "empty_features"}

            if y is None or len(y) == 0:
                return None, None, None, None, {"error": "empty_labels"}

            if len(X) != len(y):
                return None, None, None, None, {"error": "mismatched_lengths"}

            # Check minimum samples
            if len(y) < 4:  # Absolute minimum for any split
                return None, None, None, None, {
                    "error": "insufficient_samples",
                    "message": f"Need at least 4 samples, got {len(y)}"
                }

            # Check class distribution
            unique_classes, class_counts = np.unique(y, return_counts=True)

            if len(unique_classes) < 2:
                return None, None, None, None, {
                    "error": "single_class",
                    "message": f"All samples have same label: {unique_classes[0]}"
                }

            # Check if stratification is possible
            min_class_count = min(class_counts)

            if min_class_count < 2:
                # Can't stratify - use simple split
                logger.warning("Minority class has <2 samples, using non-stratified split")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
            else:
                # Use stratified split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )

            # Verify split has both classes in test
            test_classes = np.unique(y_test)
            if len(test_classes) < 2:
                # Fallback: adjust split to ensure both classes
                logger.warning("Test set missing class, adjusting split")
                # Use 50/50 split for small datasets
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.5, random_state=random_state, stratify=y
                )

            return X_train, X_test, y_train, y_test, None

        except ValueError as e:
            if "least populated class" in str(e):
                return None, None, None, None, {
                    "error": "stratification_failed",
                    "message": "Not enough samples in minority class"
                }
            return None, None, None, None, {"error": str(e)}
        except Exception as e:
            return None, None, None, None, {"error": str(e)}

    @staticmethod
    def safe_confusion_metrics(y_true, y_pred):
        """
        Compute confusion matrix metrics that NEVER fail.
        Always returns valid numbers.
        """
        from sklearn.metrics import confusion_matrix

        result = {
            "sensitivity": 0.0,
            "specificity": 0.0,
            "precision": 0.0,
            "npv": 0.0,
            "tp": 0, "tn": 0, "fp": 0, "fn": 0,
            "warning": None
        }

        try:
            if len(y_true) == 0 or len(y_pred) == 0:
                result["warning"] = "empty_arrays"
                return result

            unique_true = np.unique(y_true)
            unique_pred = np.unique(y_pred)

            if len(unique_true) < 2:
                result["warning"] = "single_class_in_truth"
                return result

            cm = confusion_matrix(y_true, y_pred)

            if cm.size == 1:
                result["warning"] = "single_class_predicted"
                return result
            elif cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                result["tn"] = int(tn)
                result["fp"] = int(fp)
                result["fn"] = int(fn)
                result["tp"] = int(tp)

                # Safe calculations
                result["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                result["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                result["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                result["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            else:
                result["warning"] = f"unexpected_matrix_shape_{cm.shape}"

            return result

        except Exception as e:
            result["warning"] = str(e)
            return result

    @staticmethod
    def safe_roc_auc(y_true, y_scores):
        """
        Compute ROC AUC that NEVER fails.
        Always returns valid curve data.
        """
        from sklearn.metrics import roc_curve, auc

        result = {
            "roc_auc": 0.5,  # Random baseline
            "fpr": [0.0, 1.0],
            "tpr": [0.0, 1.0],
            "thresholds": [1.0, 0.0],
            "warning": None
        }

        try:
            if len(y_true) == 0 or len(y_scores) == 0:
                result["warning"] = "empty_arrays"
                return result

            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                result["warning"] = "single_class"
                return result

            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc_score = auc(fpr, tpr)

            result["roc_auc"] = float(roc_auc_score)
            result["fpr"] = [float(x) for x in fpr]
            result["tpr"] = [float(x) for x in tpr]
            result["thresholds"] = [float(x) for x in thresholds]

            return result

        except Exception as e:
            result["warning"] = str(e)
            return result

    @staticmethod
    def safe_pr_auc(y_true, y_scores):
        """
        Compute Precision-Recall AUC that NEVER fails.
        """
        from sklearn.metrics import precision_recall_curve, auc

        result = {
            "pr_auc": 0.0,
            "precision": [0.0, 1.0],
            "recall": [1.0, 0.0],
            "warning": None
        }

        try:
            if len(y_true) == 0 or len(y_scores) == 0:
                result["warning"] = "empty_arrays"
                return result

            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                result["warning"] = "single_class"
                return result

            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc_score = auc(recall, precision)

            result["pr_auc"] = float(pr_auc_score)
            result["precision"] = [float(x) for x in precision]
            result["recall"] = [float(x) for x in recall]

            return result

        except Exception as e:
            result["warning"] = str(e)
            return result

    @staticmethod
    def safe_feature_importance(model, feature_names):
        """
        Extract feature importance that NEVER fails.
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
            else:
                return [{"feature": f, "importance": 0.0} for f in feature_names]

            # Normalize
            total = sum(importances)
            if total > 0:
                importances = importances / total

            # Create sorted list
            importance_list = [
                {"feature": f, "importance": float(imp)}
                for f, imp in zip(feature_names, importances)
            ]

            return sorted(importance_list, key=lambda x: x["importance"], reverse=True)

        except Exception:
            return [{"feature": f, "importance": 0.0} for f in feature_names]


# ============================================================================
# SECTION 4: SAFE NUMERIC OPERATIONS
# ============================================================================

class SafeMath:
    """Safe math operations that never fail."""

    @staticmethod
    def safe_divide(numerator, denominator, default=0.0):
        """Division that never fails."""
        try:
            if denominator == 0 or denominator is None:
                return default
            if numerator is None:
                return default
            return float(numerator) / float(denominator)
        except:
            return default

    @staticmethod
    def safe_mean(values, default=0.0):
        """Mean that handles empty/invalid arrays."""
        try:
            if values is None or len(values) == 0:
                return default
            result = np.nanmean(values)
            return default if np.isnan(result) else float(result)
        except:
            return default

    @staticmethod
    def safe_std(values, default=0.0):
        """Std deviation that handles edge cases."""
        try:
            if values is None or len(values) < 2:
                return default
            result = np.nanstd(values)
            return default if np.isnan(result) else float(result)
        except:
            return default

    @staticmethod
    def safe_percentile(values, percentile, default=0.0):
        """Percentile that handles edge cases."""
        try:
            if values is None or len(values) == 0:
                return default
            result = np.nanpercentile(values, percentile)
            return default if np.isnan(result) else float(result)
        except:
            return default

    @staticmethod
    def safe_correlation(x, y, default=0.0):
        """Correlation that handles edge cases."""
        try:
            if x is None or y is None or len(x) < 2 or len(y) < 2:
                return default
            if len(x) != len(y):
                return default
            result = np.corrcoef(x, y)[0, 1]
            return default if np.isnan(result) else float(result)
        except:
            return default


# ============================================================================
# SECTION 5: ENDPOINT-SPECIFIC REQUIREMENTS
# ============================================================================

ENDPOINT_REQUIREMENTS = {
    # Core Analysis
    "/analyze": {"min_rows": 5, "requires_label": True, "min_per_class": 2},
    "/early_risk_discovery": {"min_rows": 10, "requires_label": True, "requires_patient_id": True, "requires_time": True, "min_patients": 3},
    "/trial_rescue": {"min_rows": 20, "requires_label": True, "requires_treatment": True, "min_per_arm": 5},
    "/responder_prediction": {"min_rows": 20, "requires_label": True, "requires_treatment": True, "min_per_arm": 5},
    "/confounder_detection": {"min_rows": 10, "requires_label": True, "requires_treatment": True},
    "/confounder_analysis": {"min_rows": 10, "requires_label": True, "requires_treatment": True},
    "/predictive_modeling": {"min_rows": 10, "requires_label": True, "min_per_class": 2},
    "/emerging_phenotype": {"min_rows": 10, "requires_label": True},
    "/shap_explain": {"min_rows": 10, "requires_label": True},

    # Time Series
    "/change_point_detect": {"min_rows": 5},
    "/lead_time_analysis": {"min_rows": 10},

    # Surveillance
    "/outbreak_detection": {"min_rows": 5, "requires_region": True, "requires_time": True},
    "/surveillance/outbreak_detection": {"min_rows": 5, "requires_region": True, "requires_time": True},
    "/surveillance/unknown_diseases": {"min_rows": 10, "requires_label": True},
    "/surveillance/comprehensive": {"min_rows": 10},
    "/surveillance/multisite_synthesis": {"min_rows": 1},

    # Proteus
    "/proteus/generate_cohort": {"min_rows": 0},  # Generates data
    "/proteus/validate_model": {"min_rows": 5},
    "/proteus/ab_test": {"min_rows": 10},

    # Data Processing
    "/multi_omic_fusion": {"min_rows": 1},
    "/synthetic_cohort": {"min_rows": 0},  # Generates data
    "/population_risk": {"min_rows": 1},
    "/digital_twin_simulation": {"min_rows": 1},
    "/forecast_timeline": {"min_rows": 5},
    "/medication_interaction": {"min_rows": 0},  # Different input format

    # Security/PHI
    "/security/phi-scan": {"min_rows": 1},
    "/security/deidentify": {"min_rows": 1},

    # Predict
    "/predict": {"min_rows": 0},  # Flexible endpoint

    # Default for any endpoint not listed
    "_default": {"min_rows": 1}
}


def get_requirements(endpoint: str) -> Dict:
    """Get validation requirements for an endpoint."""
    return ENDPOINT_REQUIREMENTS.get(endpoint, ENDPOINT_REQUIREMENTS["_default"])


def validate_request(endpoint: str, df: pd.DataFrame, params: Dict) -> Optional[Dict]:
    """
    Validate request against endpoint requirements.
    Returns error dict if invalid, None if valid.
    """
    reqs = get_requirements(endpoint)

    # Check minimum rows
    if len(df) < reqs.get("min_rows", 1):
        return {
            "error": "insufficient_rows",
            "message": f"This endpoint requires at least {reqs['min_rows']} rows, got {len(df)}",
            "endpoint": endpoint
        }

    # Check label column
    if reqs.get("requires_label"):
        label_col = params.get("label_column")
        error = DataValidator.validate_label_column(
            df, label_col,
            require_binary=True,
            min_per_class=reqs.get("min_per_class", 2)
        )
        if error:
            return error

    # Check treatment column
    if reqs.get("requires_treatment"):
        treatment_col = params.get("treatment_column")
        error = DataValidator.validate_treatment_column(
            df, treatment_col,
            min_per_arm=reqs.get("min_per_arm", 5)
        )
        if error:
            return error

    # Check patient ID for longitudinal data
    if reqs.get("requires_patient_id"):
        pid_col = params.get("patient_id_column")
        if pid_col not in df.columns:
            return {
                "error": "missing_patient_id_column",
                "message": f"Patient ID column '{pid_col}' not found"
            }
        n_patients = df[pid_col].nunique()
        min_patients = reqs.get("min_patients", 1)
        if n_patients < min_patients:
            return {
                "error": "insufficient_patients",
                "message": f"Need at least {min_patients} patients, found {n_patients}"
            }

    # Check time column
    if reqs.get("requires_time"):
        time_col = params.get("time_column")
        if time_col not in df.columns:
            return {
                "error": "missing_time_column",
                "message": f"Time column '{time_col}' not found"
            }

    # Check region column
    if reqs.get("requires_region"):
        region_col = params.get("region_column")
        if region_col not in df.columns:
            return {
                "error": "missing_region_column",
                "message": f"Region column '{region_col}' not found"
            }

    return None  # All valid


# ============================================================================
# EXPORT ALL
# ============================================================================

__all__ = [
    'bulletproof_endpoint',
    'create_error_response',
    'classify_and_respond',
    'DataValidator',
    'SafeML',
    'SafeMath',
    'ENDPOINT_REQUIREMENTS',
    'get_requirements',
    'validate_request'
]
