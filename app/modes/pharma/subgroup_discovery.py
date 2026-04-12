"""
Subgroup Discovery - Step 1 of Trial Rescue
============================================

Find hidden responder subgroups in clinical trial data.

Uses clustering algorithms (KMeans, DBSCAN) to identify patient
clusters with significantly better treatment response than the
overall population.

Reference: HyperCore Implementation Guide - Appendix E, Section E.7.1
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ResponderSubgroup:
    """A discovered subgroup with better treatment response."""
    subgroup_id: str
    subgroup_name: str
    description: str

    # Subgroup characteristics
    n_patients: int
    percentage_of_total: float
    defining_features: Dict[str, Any]  # Feature ranges that define this subgroup

    # Response metrics
    response_rate: float  # Response rate in this subgroup
    overall_response_rate: float  # Response rate in full population
    relative_response_improvement: float  # (subgroup - overall) / overall
    absolute_response_improvement: float  # subgroup - overall

    # Statistical measures
    pvalue: float
    effect_size: float  # Cohen's d or similar
    confidence_interval: Tuple[float, float]

    # Plausibility
    biological_plausibility: float  # 0-1 score
    clinical_actionability: float  # 0-1 score

    # Clustering info
    cluster_method: str
    cluster_id: int

    # Raw data for downstream use
    patient_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SubgroupDiscovery:
    """
    Discovers hidden responder subgroups in clinical trial data.

    Uses multiple clustering approaches to find patient clusters
    with significantly better outcomes than the overall population.
    """

    MIN_SUBGROUP_SIZE = 10  # Minimum patients for valid subgroup
    MIN_IMPROVEMENT_THRESHOLD = 0.10  # 10% relative improvement minimum
    SIGNIFICANCE_THRESHOLD = 0.10  # p < 0.10 for interesting subgroups

    def __init__(self, random_state: int = 42):
        """Initialize subgroup discovery."""
        self.random_state = random_state
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None

    def find_subgroups(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: List[str],
        min_subgroup_size: Optional[int] = None,
    ) -> List[ResponderSubgroup]:
        """
        Find hidden responder subgroups in trial data.

        Args:
            df: DataFrame with trial data
            treatment_col: Column name for treatment assignment (0/1)
            outcome_col: Column name for outcome (0/1 or continuous)
            covariate_cols: List of covariate column names for clustering
            min_subgroup_size: Minimum patients per subgroup

        Returns:
            List of ResponderSubgroup objects sorted by improvement
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available for subgroup discovery")
            return []

        min_size = min_subgroup_size or self.MIN_SUBGROUP_SIZE
        subgroups = []

        try:
            return self._find_subgroups_impl(df, treatment_col, outcome_col, covariate_cols, min_size)
        except Exception as e:
            logger.error(f"Subgroup discovery failed: {e}", exc_info=True)
            return []

    def _find_subgroups_impl(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: List[str],
        min_size: int,
    ) -> List[ResponderSubgroup]:
        """Internal implementation of subgroup discovery."""
        subgroups = []

        # Separate treatment and control groups
        treated = df[df[treatment_col] == 1].copy()
        control = df[df[treatment_col] == 0].copy()

        if len(treated) < min_size or len(control) < min_size:
            logger.warning(f"Insufficient data for subgroup analysis: {len(treated)} treated, {len(control)} control")
            return []

        # Calculate overall response rates
        overall_treated_response = treated[outcome_col].mean()
        overall_control_response = control[outcome_col].mean()
        overall_effect = overall_treated_response - overall_control_response

        logger.info(f"Overall treated response: {overall_treated_response:.3f}")
        logger.info(f"Overall control response: {overall_control_response:.3f}")
        logger.info(f"Overall treatment effect: {overall_effect:.3f}")

        # Prepare features for clustering
        feature_cols = [c for c in covariate_cols if c in df.columns]
        if not feature_cols:
            logger.warning("No valid covariate columns for clustering")
            return []

        # Handle missing values and prepare features
        X = df[feature_cols].copy()
        X = X.fillna(X.median())

        # Convert categorical to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes

        # Scale features with error handling
        try:
            X_scaled = self.scaler.fit_transform(X)

            # Check for NaN/Inf values after scaling
            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                logger.warning("Scaled features contain NaN/Inf, replacing with 0")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            logger.warning(f"Feature scaling failed: {e}")
            return []

        n_samples = X_scaled.shape[0]
        n_features = X_scaled.shape[1]
        logger.info(f"Clustering {n_samples} samples with {n_features} features")

        # Try multiple clustering methods
        clustering_results = []

        # Method 1: KMeans with different k values
        # Only try k values that make sense for the sample size
        max_clusters = min(6, n_samples // (min_size // 2 + 1))  # Ensure enough samples per cluster
        k_values = [k for k in [2, 3, 4, 5, 6] if k <= max_clusters]

        for n_clusters in k_values:
            try:
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.random_state,
                    n_init=10,
                    max_iter=300
                )
                labels = kmeans.fit_predict(X_scaled)
                clustering_results.append(("kmeans", n_clusters, labels))
                logger.debug(f"KMeans k={n_clusters} succeeded")
            except Exception as e:
                logger.debug(f"KMeans k={n_clusters} failed: {e}")

        # Method 2: DBSCAN with different eps values
        # Adjust min_samples based on dataset size
        dbscan_min_samples = max(3, min(min_size, n_samples // 10))

        for eps in [0.3, 0.5, 1.0, 1.5, 2.0]:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=dbscan_min_samples)
                labels = dbscan.fit_predict(X_scaled)
                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters_found >= 2:  # At least 2 real clusters
                    clustering_results.append(("dbscan", eps, labels))
                    logger.debug(f"DBSCAN eps={eps} found {n_clusters_found} clusters")
            except Exception as e:
                logger.debug(f"DBSCAN eps={eps} failed: {e}")

        logger.info(f"Generated {len(clustering_results)} clustering results")

        # Analyze each clustering result with error handling
        for method, param, labels in clustering_results:
            try:
                df_temp = df.copy()
                df_temp["_cluster"] = labels

                for cluster_id in set(labels):
                    if cluster_id == -1:  # Skip DBSCAN noise
                        continue

                    try:
                        cluster_mask = df_temp["_cluster"] == cluster_id
                        cluster_df = df_temp[cluster_mask]

                        if len(cluster_df) < min_size:
                            continue

                        # Split cluster into treated and control
                        cluster_treated = cluster_df[cluster_df[treatment_col] == 1]
                        cluster_control = cluster_df[cluster_df[treatment_col] == 0]

                        if len(cluster_treated) < 5 or len(cluster_control) < 5:
                            continue

                        # Calculate cluster response rates
                        cluster_treated_response = cluster_treated[outcome_col].mean()
                        cluster_control_response = cluster_control[outcome_col].mean()
                        cluster_effect = cluster_treated_response - cluster_control_response

                        # Calculate improvement over overall
                        if overall_effect != 0:
                            relative_improvement = (cluster_effect - overall_effect) / abs(overall_effect)
                        else:
                            relative_improvement = cluster_effect

                        absolute_improvement = cluster_effect - overall_effect

                        # Only consider subgroups with meaningful improvement
                        if relative_improvement < self.MIN_IMPROVEMENT_THRESHOLD:
                            continue

                        # Statistical test (two-sample t-test on treatment effect)
                        try:
                            # Compare cluster vs non-cluster treatment effects
                            non_cluster = df_temp[~cluster_mask]
                            non_cluster_treated = non_cluster[non_cluster[treatment_col] == 1]
                            non_cluster_control = non_cluster[non_cluster[treatment_col] == 0]

                            if len(non_cluster_treated) >= 5 and len(non_cluster_control) >= 5:
                                cluster_outcomes = cluster_treated[outcome_col].values - cluster_control[outcome_col].mean()
                                non_cluster_outcomes = non_cluster_treated[outcome_col].values - non_cluster_control[outcome_col].mean()

                                t_stat, pvalue = stats.ttest_ind(cluster_outcomes, non_cluster_outcomes)
                            else:
                                pvalue = 1.0
                        except Exception:
                            pvalue = 1.0

                        # Calculate effect size (Cohen's d)
                        try:
                            pooled_std = np.sqrt(
                                (cluster_treated[outcome_col].var() + cluster_control[outcome_col].var()) / 2
                            )
                            effect_size = cluster_effect / pooled_std if pooled_std > 0 else 0.0
                        except Exception:
                            effect_size = 0.0

                        # Calculate confidence interval (bootstrap)
                        ci_lower, ci_upper = self._bootstrap_ci(
                            cluster_treated[outcome_col].values,
                            cluster_control[outcome_col].values
                        )

                        # Identify defining features
                        defining_features = self._identify_defining_features(
                            df, cluster_mask, feature_cols
                        )

                        # Calculate plausibility scores
                        biological_plausibility = self._estimate_biological_plausibility(defining_features)
                        clinical_actionability = self._estimate_clinical_actionability(
                            n_patients=len(cluster_df),
                            effect_size=effect_size,
                            defining_features=defining_features
                        )

                        # Generate subgroup name and description
                        subgroup_name = self._generate_subgroup_name(defining_features)
                        description = self._generate_description(
                            defining_features, cluster_treated_response, relative_improvement
                        )

                        # Get patient IDs
                        patient_ids = cluster_df.index.tolist() if "patient_id" not in cluster_df.columns else cluster_df["patient_id"].tolist()

                        subgroup = ResponderSubgroup(
                            subgroup_id=f"{method}_{param}_{cluster_id}",
                            subgroup_name=subgroup_name,
                            description=description,
                            n_patients=len(cluster_df),
                            percentage_of_total=len(cluster_df) / len(df) * 100,
                            defining_features=defining_features,
                            response_rate=cluster_treated_response,
                            overall_response_rate=overall_treated_response,
                            relative_response_improvement=relative_improvement,
                            absolute_response_improvement=absolute_improvement,
                            pvalue=pvalue,
                            effect_size=effect_size,
                            confidence_interval=(ci_lower, ci_upper),
                            biological_plausibility=biological_plausibility,
                            clinical_actionability=clinical_actionability,
                            cluster_method=method,
                            cluster_id=cluster_id,
                            patient_ids=patient_ids,
                            metadata={
                                "method_param": param,
                                "cluster_effect": cluster_effect,
                                "overall_effect": overall_effect,
                            }
                        )
                        subgroups.append(subgroup)

                    except Exception as e:
                        logger.debug(f"Error analyzing cluster {cluster_id} from {method}: {e}")
                        continue

            except Exception as e:
                logger.warning(f"Error processing {method} clustering: {e}")
                continue

        # Remove duplicates (similar subgroups from different methods)
        subgroups = self._deduplicate_subgroups(subgroups)

        # Sort by relative improvement (descending)
        subgroups.sort(key=lambda x: x.relative_response_improvement, reverse=True)

        logger.info(f"Found {len(subgroups)} distinct responder subgroups")
        return subgroups

    def _bootstrap_ci(
        self,
        treated_outcomes: np.ndarray,
        control_outcomes: np.ndarray,
        n_bootstrap: int = 1000,
        ci: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for treatment effect."""
        np.random.seed(self.random_state)
        effects = []

        for _ in range(n_bootstrap):
            t_sample = np.random.choice(treated_outcomes, size=len(treated_outcomes), replace=True)
            c_sample = np.random.choice(control_outcomes, size=len(control_outcomes), replace=True)
            effects.append(t_sample.mean() - c_sample.mean())

        alpha = (1 - ci) / 2
        return np.percentile(effects, alpha * 100), np.percentile(effects, (1 - alpha) * 100)

    def _identify_defining_features(
        self,
        df: pd.DataFrame,
        cluster_mask: pd.Series,
        feature_cols: List[str]
    ) -> Dict[str, Any]:
        """Identify features that define a subgroup."""
        defining = {}

        for col in feature_cols:
            cluster_values = df.loc[cluster_mask, col]
            non_cluster_values = df.loc[~cluster_mask, col]

            if cluster_values.dtype in ['float64', 'int64']:
                # Numeric: check if range is significantly different
                cluster_mean = cluster_values.mean()
                cluster_std = cluster_values.std()
                non_cluster_mean = non_cluster_values.mean()

                # Use effect size to determine if feature is defining
                pooled_std = np.sqrt((cluster_std**2 + non_cluster_values.std()**2) / 2)
                if pooled_std > 0:
                    effect = abs(cluster_mean - non_cluster_mean) / pooled_std
                    if effect > 0.5:  # Medium effect size
                        defining[col] = {
                            "type": "numeric",
                            "cluster_mean": cluster_mean,
                            "cluster_range": (cluster_values.min(), cluster_values.max()),
                            "effect_size": effect,
                        }
            else:
                # Categorical: check if distribution is different
                cluster_mode = cluster_values.mode().iloc[0] if len(cluster_values.mode()) > 0 else None
                cluster_freq = (cluster_values == cluster_mode).mean() if cluster_mode else 0

                non_cluster_freq = (non_cluster_values == cluster_mode).mean() if cluster_mode else 0

                if cluster_freq - non_cluster_freq > 0.2:  # 20% difference
                    defining[col] = {
                        "type": "categorical",
                        "dominant_value": cluster_mode,
                        "cluster_frequency": cluster_freq,
                        "difference": cluster_freq - non_cluster_freq,
                    }

        return defining

    def _estimate_biological_plausibility(self, defining_features: Dict[str, Any]) -> float:
        """Estimate biological plausibility of subgroup (0-1)."""
        # Higher score for features with known biological relevance
        plausible_markers = [
            "age", "bmi", "weight", "egfr", "creatinine",
            "biomarker", "gene", "mutation", "expression",
            "baseline", "severity", "stage"
        ]

        score = 0.5  # Base score
        n_features = len(defining_features)

        for feature in defining_features:
            feature_lower = feature.lower()
            if any(marker in feature_lower for marker in plausible_markers):
                score += 0.1

        return min(1.0, score)

    def _estimate_clinical_actionability(
        self,
        n_patients: int,
        effect_size: float,
        defining_features: Dict[str, Any]
    ) -> float:
        """Estimate clinical actionability (0-1)."""
        score = 0.5

        # Larger subgroups are more actionable
        if n_patients >= 100:
            score += 0.2
        elif n_patients >= 50:
            score += 0.1

        # Larger effect sizes are more actionable
        if effect_size >= 0.8:
            score += 0.2
        elif effect_size >= 0.5:
            score += 0.1

        # Simple defining criteria are more actionable
        if len(defining_features) <= 3:
            score += 0.1

        return min(1.0, score)

    def _generate_subgroup_name(self, defining_features: Dict[str, Any]) -> str:
        """Generate a descriptive name for the subgroup."""
        if not defining_features:
            return "Uncharacterized Responders"

        parts = []
        for feature, info in list(defining_features.items())[:2]:  # Top 2 features
            if info["type"] == "numeric":
                parts.append(f"{feature}={info['cluster_mean']:.1f}")
            else:
                parts.append(f"{feature}={info['dominant_value']}")

        return " + ".join(parts) if parts else "Clustered Responders"

    def _generate_description(
        self,
        defining_features: Dict[str, Any],
        response_rate: float,
        relative_improvement: float
    ) -> str:
        """Generate description of the subgroup."""
        feature_desc = []
        for feature, info in defining_features.items():
            if info["type"] == "numeric":
                feature_desc.append(f"{feature} around {info['cluster_mean']:.1f}")
            else:
                feature_desc.append(f"{feature}={info['dominant_value']}")

        features_str = ", ".join(feature_desc) if feature_desc else "mixed characteristics"

        return (
            f"Patients with {features_str} show {response_rate:.1%} response rate, "
            f"representing {relative_improvement:.1%} relative improvement over overall population."
        )

    def _deduplicate_subgroups(
        self,
        subgroups: List[ResponderSubgroup],
        overlap_threshold: float = 0.7
    ) -> List[ResponderSubgroup]:
        """Remove duplicate/highly overlapping subgroups."""
        if len(subgroups) <= 1:
            return subgroups

        unique = []
        for sg in subgroups:
            is_duplicate = False
            sg_set = set(sg.patient_ids)

            for existing in unique:
                existing_set = set(existing.patient_ids)
                overlap = len(sg_set & existing_set) / max(len(sg_set), len(existing_set))

                if overlap > overlap_threshold:
                    # Keep the one with better improvement
                    if sg.relative_response_improvement > existing.relative_response_improvement:
                        unique.remove(existing)
                        unique.append(sg)
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(sg)

        return unique
