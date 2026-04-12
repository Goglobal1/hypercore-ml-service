"""
Confounder Detector - Step 2 of Trial Rescue
=============================================

Detect confounding variables that mask treatment effect.

Identifies variables that:
1. Correlate with both treatment and outcome
2. Create imbalance between treatment arms
3. Mask the true treatment effect when not controlled

Reference: HyperCore Implementation Guide - Appendix E, Section E.7.2
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Confounder:
    """A detected confounding variable."""
    variable_name: str
    confounder_type: str  # "imbalance", "interaction", "mediator", "collider"
    description: str

    # Confounder metrics
    impact_score: float  # 0-1 score of confounding impact
    treatment_correlation: float  # Correlation with treatment assignment
    outcome_correlation: float  # Correlation with outcome
    imbalance_ratio: float  # Imbalance between arms

    # Effect adjustment
    unadjusted_effect: float  # Treatment effect without controlling
    adjusted_effect: float  # Treatment effect after controlling
    effect_change: float  # Difference (adjusted - unadjusted)
    effect_change_percentage: float  # Percentage change

    # Statistical
    pvalue: float
    confidence: float

    # Recommendations
    adjustment_method: str  # "stratification", "matching", "regression", "ipw"
    recommendation: str

    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfounderDetector:
    """
    Detects confounding variables in clinical trial data.

    A confounder is a variable that:
    1. Is associated with the treatment (Z -> X)
    2. Is associated with the outcome (Z -> Y)
    3. Is NOT on the causal path from treatment to outcome

    Controlling for confounders can reveal hidden treatment effects.
    """

    IMPACT_THRESHOLD = 0.05  # 5% effect change to flag as confounder
    CORRELATION_THRESHOLD = 0.1  # Minimum correlation to consider

    def __init__(self):
        """Initialize confounder detector."""
        pass

    def detect_confounders(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: List[str],
    ) -> List[Confounder]:
        """
        Detect confounding variables in trial data.

        Args:
            df: DataFrame with trial data
            treatment_col: Column name for treatment (0/1)
            outcome_col: Column name for outcome
            covariate_cols: List of potential confounder columns

        Returns:
            List of Confounder objects sorted by impact
        """
        try:
            return self._detect_confounders_impl(df, treatment_col, outcome_col, covariate_cols)
        except Exception as e:
            logger.error(f"Confounder detection failed: {e}", exc_info=True)
            return []

    def _detect_confounders_impl(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: List[str],
    ) -> List[Confounder]:
        """Internal implementation of confounder detection."""
        confounders = []

        # Calculate unadjusted treatment effect
        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]
        unadjusted_effect = treated[outcome_col].mean() - control[outcome_col].mean()

        logger.info(f"Unadjusted treatment effect: {unadjusted_effect:.4f}")

        # Analyze each covariate
        for col in covariate_cols:
            if col not in df.columns:
                continue

            try:
                confounder = self._analyze_potential_confounder(
                    df=df,
                    covariate_col=col,
                    treatment_col=treatment_col,
                    outcome_col=outcome_col,
                    unadjusted_effect=unadjusted_effect,
                )

                if confounder and confounder.impact_score >= self.IMPACT_THRESHOLD:
                    confounders.append(confounder)

            except Exception as e:
                logger.debug(f"Error analyzing {col}: {e}")

        # Sort by impact score (descending)
        confounders.sort(key=lambda x: x.impact_score, reverse=True)

        logger.info(f"Detected {len(confounders)} significant confounders")
        return confounders

    def _analyze_potential_confounder(
        self,
        df: pd.DataFrame,
        covariate_col: str,
        treatment_col: str,
        outcome_col: str,
        unadjusted_effect: float,
    ) -> Optional[Confounder]:
        """Analyze a single variable for confounding."""
        # Handle missing values
        df_clean = df[[covariate_col, treatment_col, outcome_col]].dropna()

        if len(df_clean) < 20:
            return None

        covariate = df_clean[covariate_col]
        treatment = df_clean[treatment_col]
        outcome = df_clean[outcome_col]

        # Convert categorical to numeric if needed
        if covariate.dtype == 'object':
            covariate = pd.Categorical(covariate).codes

        # 1. Calculate correlation with treatment
        try:
            treatment_corr, _ = stats.pointbiserialr(treatment, covariate)
        except Exception:
            treatment_corr = 0.0

        # 2. Calculate correlation with outcome
        try:
            if outcome.dtype in ['float64', 'int64'] and len(outcome.unique()) > 2:
                outcome_corr, _ = stats.pearsonr(covariate, outcome)
            else:
                outcome_corr, _ = stats.pointbiserialr(outcome, covariate)
        except Exception:
            outcome_corr = 0.0

        # 3. Calculate imbalance between arms
        treated_cov = covariate[treatment == 1]
        control_cov = covariate[treatment == 0]

        try:
            t_stat, imbalance_pvalue = stats.ttest_ind(treated_cov, control_cov)
            imbalance_ratio = abs(treated_cov.mean() - control_cov.mean()) / (covariate.std() + 1e-10)
        except Exception:
            imbalance_ratio = 0.0
            imbalance_pvalue = 1.0

        # Skip if correlations are too weak
        if abs(treatment_corr) < self.CORRELATION_THRESHOLD or abs(outcome_corr) < self.CORRELATION_THRESHOLD:
            return None

        # 4. Calculate adjusted effect (controlling for this variable)
        adjusted_effect = self._calculate_adjusted_effect(
            df_clean, covariate_col, treatment_col, outcome_col
        )

        # 5. Calculate effect change
        effect_change = adjusted_effect - unadjusted_effect
        if unadjusted_effect != 0:
            effect_change_pct = abs(effect_change / unadjusted_effect) * 100
        else:
            effect_change_pct = abs(effect_change) * 100

        # 6. Determine confounder type
        confounder_type = self._determine_confounder_type(
            treatment_corr, outcome_corr, imbalance_ratio, effect_change
        )

        # 7. Calculate impact score
        impact_score = self._calculate_impact_score(
            treatment_corr, outcome_corr, imbalance_ratio, effect_change_pct
        )

        # 8. Determine adjustment method
        adjustment_method = self._recommend_adjustment_method(
            df_clean, covariate_col, treatment_col, confounder_type
        )

        # 9. Generate recommendation
        recommendation = self._generate_recommendation(
            covariate_col, confounder_type, effect_change, adjusted_effect
        )

        # 10. Calculate confidence
        confidence = 1.0 - imbalance_pvalue

        return Confounder(
            variable_name=covariate_col,
            confounder_type=confounder_type,
            description=f"{covariate_col} shows {confounder_type} confounding pattern",
            impact_score=impact_score,
            treatment_correlation=treatment_corr,
            outcome_correlation=outcome_corr,
            imbalance_ratio=imbalance_ratio,
            unadjusted_effect=unadjusted_effect,
            adjusted_effect=adjusted_effect,
            effect_change=effect_change,
            effect_change_percentage=effect_change_pct,
            pvalue=imbalance_pvalue,
            confidence=confidence,
            adjustment_method=adjustment_method,
            recommendation=recommendation,
            metadata={
                "treated_mean": float(treated_cov.mean()),
                "control_mean": float(control_cov.mean()),
                "n_complete": len(df_clean),
            }
        )

    def _calculate_adjusted_effect(
        self,
        df: pd.DataFrame,
        covariate_col: str,
        treatment_col: str,
        outcome_col: str,
    ) -> float:
        """Calculate treatment effect controlling for covariate."""
        if not SKLEARN_AVAILABLE:
            # Fallback: stratified analysis
            return self._stratified_effect(df, covariate_col, treatment_col, outcome_col)

        try:
            # Use regression adjustment
            X = df[[treatment_col, covariate_col]].copy()

            # Handle categorical covariates
            if X[covariate_col].dtype == 'object':
                X[covariate_col] = pd.Categorical(X[covariate_col]).codes

            y = df[outcome_col]

            if len(y.unique()) == 2:
                # Binary outcome: logistic regression
                model = LogisticRegression(max_iter=1000)
                model.fit(X, y)
                # Treatment coefficient
                return model.coef_[0][0]
            else:
                # Continuous outcome: linear regression
                model = LinearRegression()
                model.fit(X, y)
                return model.coef_[0]  # Treatment coefficient

        except Exception as e:
            logger.debug(f"Regression adjustment failed: {e}")
            return self._stratified_effect(df, covariate_col, treatment_col, outcome_col)

    def _stratified_effect(
        self,
        df: pd.DataFrame,
        covariate_col: str,
        treatment_col: str,
        outcome_col: str,
    ) -> float:
        """Calculate stratified treatment effect."""
        covariate = df[covariate_col]

        # Create strata (tertiles for continuous, categories for categorical)
        if covariate.dtype in ['float64', 'int64']:
            try:
                strata = pd.qcut(covariate, 3, labels=['low', 'mid', 'high'], duplicates='drop')
            except Exception:
                strata = pd.cut(covariate, 3, labels=['low', 'mid', 'high'])
        else:
            strata = covariate

        # Calculate weighted average effect across strata
        effects = []
        weights = []

        for stratum in strata.unique():
            stratum_df = df[strata == stratum]
            if len(stratum_df) < 10:
                continue

            treated = stratum_df[stratum_df[treatment_col] == 1]
            control = stratum_df[stratum_df[treatment_col] == 0]

            if len(treated) >= 2 and len(control) >= 2:
                effect = treated[outcome_col].mean() - control[outcome_col].mean()
                weight = len(stratum_df)
                effects.append(effect)
                weights.append(weight)

        if effects:
            return np.average(effects, weights=weights)
        return 0.0

    def _determine_confounder_type(
        self,
        treatment_corr: float,
        outcome_corr: float,
        imbalance_ratio: float,
        effect_change: float,
    ) -> str:
        """Determine the type of confounding."""
        # Classic confounder: correlated with both, creates bias
        if abs(treatment_corr) > 0.2 and abs(outcome_corr) > 0.2:
            if effect_change > 0:
                return "suppressor"  # Masks positive effect
            else:
                return "amplifier"  # Amplifies apparent effect

        # Imbalance confounder: uneven distribution between arms
        if imbalance_ratio > 0.3:
            return "imbalance"

        # Interaction: modifies treatment effect
        if abs(treatment_corr) < 0.1 and abs(outcome_corr) > 0.3:
            return "interaction"

        return "weak"

    def _calculate_impact_score(
        self,
        treatment_corr: float,
        outcome_corr: float,
        imbalance_ratio: float,
        effect_change_pct: float,
    ) -> float:
        """Calculate overall impact score (0-1)."""
        # Weighted combination of confounding indicators
        score = 0.0

        # Correlation component (max 0.3)
        corr_component = min(0.3, abs(treatment_corr * outcome_corr))
        score += corr_component

        # Imbalance component (max 0.3)
        imbalance_component = min(0.3, imbalance_ratio * 0.5)
        score += imbalance_component

        # Effect change component (max 0.4)
        effect_component = min(0.4, effect_change_pct / 100)
        score += effect_component

        return min(1.0, score)

    def _recommend_adjustment_method(
        self,
        df: pd.DataFrame,
        covariate_col: str,
        treatment_col: str,
        confounder_type: str,
    ) -> str:
        """Recommend adjustment method."""
        covariate = df[covariate_col]

        # For categorical with few levels: stratification
        if covariate.dtype == 'object' or len(covariate.unique()) <= 5:
            return "stratification"

        # For severe imbalance: propensity score matching or IPW
        if confounder_type == "imbalance":
            return "ipw"  # Inverse probability weighting

        # Default: regression adjustment
        return "regression"

    def _generate_recommendation(
        self,
        variable_name: str,
        confounder_type: str,
        effect_change: float,
        adjusted_effect: float,
    ) -> str:
        """Generate actionable recommendation."""
        direction = "increases" if effect_change > 0 else "decreases"

        if confounder_type == "suppressor":
            return (
                f"Controlling for {variable_name} {direction} the observed treatment effect. "
                f"Consider subgroup analysis or stratification to reveal true effect "
                f"(adjusted effect: {adjusted_effect:.4f})."
            )

        if confounder_type == "imbalance":
            return (
                f"Baseline imbalance in {variable_name} between arms may bias results. "
                f"Consider propensity score adjustment or matching to correct."
            )

        if confounder_type == "interaction":
            return (
                f"{variable_name} may modify treatment effect. "
                f"Consider interaction analysis or subgroup-specific effects."
            )

        return (
            f"Adjusting for {variable_name} changes effect by {abs(effect_change):.4f}. "
            f"Consider including in adjusted analysis."
        )


def detect_confounders_in_trial(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: Optional[List[str]] = None,
) -> List[Confounder]:
    """
    Convenience function to detect confounders.

    Args:
        df: Trial data
        treatment_col: Treatment column name
        outcome_col: Outcome column name
        covariate_cols: Optional list of covariates (defaults to all other columns)

    Returns:
        List of detected confounders
    """
    if covariate_cols is None:
        covariate_cols = [
            c for c in df.columns
            if c not in [treatment_col, outcome_col]
        ]

    detector = ConfounderDetector()
    return detector.detect_confounders(df, treatment_col, outcome_col, covariate_cols)
