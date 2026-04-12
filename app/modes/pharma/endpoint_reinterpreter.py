"""
Endpoint Reinterpreter - Step 3 of Trial Rescue
================================================

Find alternative endpoints where treatment shows effect.

When the primary endpoint fails, this module searches for:
1. Secondary endpoints with significant effects
2. Composite endpoints combining multiple measures
3. Time-to-event endpoints with different cutoffs
4. Biomarker-based surrogate endpoints

Reference: HyperCore Implementation Guide - Appendix E, Section E.7.3
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class AlternativeEndpoint:
    """An alternative endpoint showing treatment effect."""
    endpoint_id: str
    endpoint_name: str
    endpoint_type: str  # "secondary", "composite", "surrogate", "time_to_event", "biomarker"
    description: str

    # Effect metrics
    treatment_effect: float
    effect_size: float  # Standardized effect size
    pvalue: float
    confidence_interval: Tuple[float, float]

    # Comparison to primary
    primary_endpoint_effect: float
    improvement_over_primary: float  # How much better than primary

    # Validation metrics (required)
    clinical_relevance: float  # 0-1 score
    regulatory_acceptability: float  # 0-1 score (FDA/EMA likelihood)
    biological_plausibility: float  # 0-1 score

    # Fields with defaults
    # Composite details (if applicable)
    components: List[str] = field(default_factory=list)
    component_weights: Dict[str, float] = field(default_factory=dict)

    # Sample size implications
    n_treated: int = 0
    n_control: int = 0
    power_estimate: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)


class EndpointReinterpreter:
    """
    Finds alternative endpoints where treatment shows effect.

    When primary endpoint fails, searches for:
    - Secondary endpoints with better results
    - Composite endpoints combining outcomes
    - Different timepoints or definitions
    - Biomarker surrogates
    """

    MIN_EFFECT_SIZE = 0.1  # Minimum Cohen's d to consider
    MAX_PVALUE = 0.10  # Maximum p-value to surface

    def __init__(self):
        """Initialize endpoint reinterpreter."""
        pass

    def find_alternatives(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        primary_outcome_col: str,
        secondary_outcome_cols: Optional[List[str]] = None,
        biomarker_cols: Optional[List[str]] = None,
    ) -> List[AlternativeEndpoint]:
        """
        Find alternative endpoints with treatment effect.

        Args:
            df: DataFrame with trial data
            treatment_col: Treatment assignment column (0/1)
            primary_outcome_col: Primary endpoint column
            secondary_outcome_cols: Optional list of secondary outcomes
            biomarker_cols: Optional list of biomarker columns

        Returns:
            List of AlternativeEndpoint objects sorted by effect size
        """
        alternatives = []

        # Calculate primary endpoint effect
        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]

        primary_effect = self._calculate_effect(
            treated[primary_outcome_col],
            control[primary_outcome_col]
        )

        logger.info(f"Primary endpoint effect: {primary_effect['effect']:.4f} (p={primary_effect['pvalue']:.4f})")

        # Identify potential secondary outcomes
        if secondary_outcome_cols is None:
            # Auto-detect: numeric columns that could be outcomes
            secondary_outcome_cols = [
                c for c in df.columns
                if c not in [treatment_col, primary_outcome_col]
                and df[c].dtype in ['float64', 'int64']
                and df[c].nunique() > 1
            ]

        # Identify biomarker columns
        if biomarker_cols is None:
            biomarker_cols = [
                c for c in df.columns
                if any(marker in c.lower() for marker in [
                    'biomarker', 'marker', 'level', 'concentration',
                    'expression', 'protein', 'gene', 'cytokine'
                ])
                and df[c].dtype in ['float64', 'int64']
            ]

        # 1. Analyze secondary endpoints
        for col in secondary_outcome_cols[:20]:  # Limit to avoid multiple testing issues
            try:
                alt = self._analyze_secondary_endpoint(
                    df, treatment_col, col, primary_effect
                )
                if alt and alt.pvalue <= self.MAX_PVALUE:
                    alternatives.append(alt)
            except Exception as e:
                logger.debug(f"Error analyzing {col}: {e}")

        # 2. Generate and analyze composite endpoints
        if len(secondary_outcome_cols) >= 2:
            composite_alts = self._find_composite_endpoints(
                df, treatment_col, primary_outcome_col,
                secondary_outcome_cols[:10], primary_effect
            )
            alternatives.extend(composite_alts)

        # 3. Analyze biomarker surrogates
        for col in biomarker_cols[:10]:
            try:
                alt = self._analyze_biomarker_endpoint(
                    df, treatment_col, col, primary_effect
                )
                if alt and alt.pvalue <= self.MAX_PVALUE:
                    alternatives.append(alt)
            except Exception as e:
                logger.debug(f"Error analyzing biomarker {col}: {e}")

        # 4. Try different outcome definitions (if binary outcome)
        if df[primary_outcome_col].nunique() <= 10:
            threshold_alts = self._try_threshold_variations(
                df, treatment_col, primary_outcome_col, primary_effect
            )
            alternatives.extend(threshold_alts)

        # Remove duplicates and sort by effect size
        alternatives = self._deduplicate_alternatives(alternatives)
        alternatives.sort(key=lambda x: abs(x.effect_size), reverse=True)

        logger.info(f"Found {len(alternatives)} alternative endpoints")
        return alternatives

    def _calculate_effect(
        self,
        treated_values: pd.Series,
        control_values: pd.Series,
    ) -> Dict[str, float]:
        """Calculate treatment effect with statistics."""
        treated = treated_values.dropna()
        control = control_values.dropna()

        if len(treated) < 5 or len(control) < 5:
            return {"effect": 0.0, "effect_size": 0.0, "pvalue": 1.0, "ci": (0.0, 0.0)}

        # Raw effect
        effect = treated.mean() - control.mean()

        # Pooled standard deviation
        pooled_std = np.sqrt(
            ((len(treated) - 1) * treated.var() + (len(control) - 1) * control.var()) /
            (len(treated) + len(control) - 2)
        )

        # Cohen's d
        effect_size = effect / pooled_std if pooled_std > 0 else 0.0

        # T-test
        try:
            t_stat, pvalue = stats.ttest_ind(treated, control)
        except Exception:
            pvalue = 1.0

        # Confidence interval (95%)
        se = pooled_std * np.sqrt(1/len(treated) + 1/len(control))
        ci = (effect - 1.96 * se, effect + 1.96 * se)

        return {
            "effect": effect,
            "effect_size": effect_size,
            "pvalue": pvalue,
            "ci": ci,
            "n_treated": len(treated),
            "n_control": len(control),
        }

    def _analyze_secondary_endpoint(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        primary_effect: Dict[str, float],
    ) -> Optional[AlternativeEndpoint]:
        """Analyze a secondary endpoint."""
        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]

        effect = self._calculate_effect(treated[outcome_col], control[outcome_col])

        if abs(effect["effect_size"]) < self.MIN_EFFECT_SIZE:
            return None

        # Calculate improvement over primary
        if primary_effect["effect_size"] != 0:
            improvement = (abs(effect["effect_size"]) - abs(primary_effect["effect_size"])) / abs(primary_effect["effect_size"])
        else:
            improvement = abs(effect["effect_size"])

        # Estimate clinical relevance
        clinical_relevance = self._estimate_clinical_relevance(outcome_col, effect["effect_size"])

        # Estimate regulatory acceptability
        regulatory = self._estimate_regulatory_acceptability(outcome_col, effect["pvalue"])

        # Estimate power
        power = self._estimate_power(effect["effect_size"], effect["n_treated"], effect["n_control"])

        return AlternativeEndpoint(
            endpoint_id=f"secondary_{outcome_col}",
            endpoint_name=outcome_col,
            endpoint_type="secondary",
            description=f"Secondary endpoint {outcome_col} shows effect size {effect['effect_size']:.3f}",
            treatment_effect=effect["effect"],
            effect_size=effect["effect_size"],
            pvalue=effect["pvalue"],
            confidence_interval=effect["ci"],
            primary_endpoint_effect=primary_effect["effect"],
            improvement_over_primary=improvement,
            clinical_relevance=clinical_relevance,
            regulatory_acceptability=regulatory,
            biological_plausibility=0.7,  # Default for measured outcome
            n_treated=effect["n_treated"],
            n_control=effect["n_control"],
            power_estimate=power,
        )

    def _find_composite_endpoints(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        primary_col: str,
        secondary_cols: List[str],
        primary_effect: Dict[str, float],
    ) -> List[AlternativeEndpoint]:
        """Find effective composite endpoints."""
        composites = []

        # Try 2-component and 3-component composites
        for n_components in [2, 3]:
            for combo in combinations(secondary_cols, n_components):
                try:
                    composite = self._create_composite_endpoint(
                        df, treatment_col, list(combo), primary_effect
                    )
                    if composite and composite.pvalue <= self.MAX_PVALUE:
                        composites.append(composite)
                except Exception as e:
                    logger.debug(f"Composite {combo} failed: {e}")

        # Limit number of composites
        composites.sort(key=lambda x: x.pvalue)
        return composites[:5]

    def _create_composite_endpoint(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        component_cols: List[str],
        primary_effect: Dict[str, float],
    ) -> Optional[AlternativeEndpoint]:
        """Create and analyze a composite endpoint."""
        # Standardize each component
        composite_values = pd.DataFrame()

        for col in component_cols:
            values = df[col].copy()
            if values.std() > 0:
                composite_values[col] = (values - values.mean()) / values.std()
            else:
                return None

        # Average composite (equal weights)
        df_temp = df.copy()
        df_temp["_composite"] = composite_values.mean(axis=1)

        treated = df_temp[df_temp[treatment_col] == 1]
        control = df_temp[df_temp[treatment_col] == 0]

        effect = self._calculate_effect(treated["_composite"], control["_composite"])

        if abs(effect["effect_size"]) < self.MIN_EFFECT_SIZE:
            return None

        # Calculate improvement
        if primary_effect["effect_size"] != 0:
            improvement = (abs(effect["effect_size"]) - abs(primary_effect["effect_size"])) / abs(primary_effect["effect_size"])
        else:
            improvement = abs(effect["effect_size"])

        composite_name = " + ".join(component_cols)

        return AlternativeEndpoint(
            endpoint_id=f"composite_{'_'.join(c[:5] for c in component_cols)}",
            endpoint_name=f"Composite: {composite_name}",
            endpoint_type="composite",
            description=f"Composite of {len(component_cols)} endpoints with effect size {effect['effect_size']:.3f}",
            treatment_effect=effect["effect"],
            effect_size=effect["effect_size"],
            pvalue=effect["pvalue"],
            confidence_interval=effect["ci"],
            primary_endpoint_effect=primary_effect["effect"],
            improvement_over_primary=improvement,
            components=component_cols,
            component_weights={c: 1.0/len(component_cols) for c in component_cols},
            clinical_relevance=0.6,  # Composites are moderately relevant
            regulatory_acceptability=0.5,  # Need pre-specification
            biological_plausibility=0.6,
            n_treated=effect["n_treated"],
            n_control=effect["n_control"],
            power_estimate=self._estimate_power(effect["effect_size"], effect["n_treated"], effect["n_control"]),
        )

    def _analyze_biomarker_endpoint(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        biomarker_col: str,
        primary_effect: Dict[str, float],
    ) -> Optional[AlternativeEndpoint]:
        """Analyze a biomarker as surrogate endpoint."""
        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]

        effect = self._calculate_effect(treated[biomarker_col], control[biomarker_col])

        if abs(effect["effect_size"]) < self.MIN_EFFECT_SIZE:
            return None

        # Improvement over primary
        if primary_effect["effect_size"] != 0:
            improvement = (abs(effect["effect_size"]) - abs(primary_effect["effect_size"])) / abs(primary_effect["effect_size"])
        else:
            improvement = abs(effect["effect_size"])

        return AlternativeEndpoint(
            endpoint_id=f"biomarker_{biomarker_col}",
            endpoint_name=f"Biomarker: {biomarker_col}",
            endpoint_type="biomarker",
            description=f"Biomarker {biomarker_col} shows effect size {effect['effect_size']:.3f}",
            treatment_effect=effect["effect"],
            effect_size=effect["effect_size"],
            pvalue=effect["pvalue"],
            confidence_interval=effect["ci"],
            primary_endpoint_effect=primary_effect["effect"],
            improvement_over_primary=improvement,
            clinical_relevance=0.5,  # Biomarkers need validation
            regulatory_acceptability=0.4,  # Surrogates need qualification
            biological_plausibility=0.8,  # Direct measurement
            n_treated=effect["n_treated"],
            n_control=effect["n_control"],
            power_estimate=self._estimate_power(effect["effect_size"], effect["n_treated"], effect["n_control"]),
        )

    def _try_threshold_variations(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        primary_effect: Dict[str, float],
    ) -> List[AlternativeEndpoint]:
        """Try different threshold definitions for outcome."""
        alternatives = []
        values = df[outcome_col].dropna()

        if values.dtype not in ['float64', 'int64']:
            return alternatives

        # Try different percentile thresholds
        for pct in [25, 50, 75]:
            threshold = np.percentile(values, pct)

            df_temp = df.copy()
            df_temp["_threshold_outcome"] = (df_temp[outcome_col] > threshold).astype(int)

            treated = df_temp[df_temp[treatment_col] == 1]
            control = df_temp[df_temp[treatment_col] == 0]

            # Chi-square test for binary outcome
            try:
                treated_success = treated["_threshold_outcome"].sum()
                treated_fail = len(treated) - treated_success
                control_success = control["_threshold_outcome"].sum()
                control_fail = len(control) - control_success

                contingency = [[treated_success, treated_fail], [control_success, control_fail]]
                chi2, pvalue, dof, expected = stats.chi2_contingency(contingency)

                # Risk difference
                risk_diff = treated_success/len(treated) - control_success/len(control)

                # Effect size (phi coefficient)
                n = len(df_temp)
                effect_size = np.sqrt(chi2 / n)

                if pvalue <= self.MAX_PVALUE and abs(effect_size) >= self.MIN_EFFECT_SIZE:
                    # Improvement
                    if primary_effect["effect_size"] != 0:
                        improvement = (abs(effect_size) - abs(primary_effect["effect_size"])) / abs(primary_effect["effect_size"])
                    else:
                        improvement = abs(effect_size)

                    alternatives.append(AlternativeEndpoint(
                        endpoint_id=f"threshold_{outcome_col}_p{pct}",
                        endpoint_name=f"{outcome_col} > {threshold:.2f} (p{pct})",
                        endpoint_type="threshold",
                        description=f"Binary threshold at {pct}th percentile shows effect",
                        treatment_effect=risk_diff,
                        effect_size=effect_size,
                        pvalue=pvalue,
                        confidence_interval=(0.0, 0.0),  # Would need bootstrap
                        primary_endpoint_effect=primary_effect["effect"],
                        improvement_over_primary=improvement,
                        clinical_relevance=0.6,
                        regulatory_acceptability=0.5,
                        biological_plausibility=0.6,
                        n_treated=len(treated),
                        n_control=len(control),
                        power_estimate=0.0,
                        metadata={"threshold": threshold, "percentile": pct},
                    ))
            except Exception as e:
                logger.debug(f"Threshold analysis failed at p{pct}: {e}")

        return alternatives

    def _estimate_clinical_relevance(self, endpoint_name: str, effect_size: float) -> float:
        """Estimate clinical relevance (0-1)."""
        # Higher effect sizes are more relevant
        size_score = min(1.0, abs(effect_size) / 0.8)

        # Some endpoints are more clinically meaningful
        high_value_terms = ["survival", "mortality", "death", "event", "hospitalization", "response"]
        endpoint_lower = endpoint_name.lower()

        if any(term in endpoint_lower for term in high_value_terms):
            return min(1.0, 0.3 + size_score * 0.7)

        return size_score * 0.7

    def _estimate_regulatory_acceptability(self, endpoint_name: str, pvalue: float) -> float:
        """Estimate regulatory acceptability (0-1)."""
        # P-value component
        pvalue_score = 1.0 - min(1.0, pvalue / 0.05)

        # Endpoint type component
        accepted_terms = ["survival", "response", "remission", "cure", "death", "progression"]
        endpoint_lower = endpoint_name.lower()

        if any(term in endpoint_lower for term in accepted_terms):
            return min(1.0, 0.4 + pvalue_score * 0.6)

        return pvalue_score * 0.5

    def _estimate_power(self, effect_size: float, n_treated: int, n_control: int) -> float:
        """Estimate statistical power."""
        try:
            from scipy.stats import norm

            n = (n_treated + n_control) / 2
            se = np.sqrt(2 / n)
            z_alpha = 1.96  # Two-sided alpha = 0.05

            z_power = abs(effect_size) / se - z_alpha
            power = norm.cdf(z_power)

            return power
        except Exception:
            return 0.5

    def _deduplicate_alternatives(
        self,
        alternatives: List[AlternativeEndpoint]
    ) -> List[AlternativeEndpoint]:
        """Remove duplicate alternatives."""
        seen = set()
        unique = []

        for alt in alternatives:
            key = (alt.endpoint_type, alt.endpoint_name)
            if key not in seen:
                seen.add(key)
                unique.append(alt)

        return unique
