"""
Inflection Point Detection - Layer 2 of Trajectory System

Detects when biomarker trajectories change from stable to trending.
This is the KEY innovation: catching the moment things start going wrong,
not when they've already gone wrong.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class InflectionPoint:
    biomarker: str
    day_index: int
    days_ago: float
    value_at_inflection: float
    current_value: float
    change_since_inflection: float  # Percentage
    inflection_type: str  # 'acceleration', 'trend_reversal', 'breakout'
    significance: float


@dataclass
class TrajectoryPhase:
    phase: str  # 'baseline', 'early_rise', 'acceleration', 'critical', 'peak', 'recovery'
    started_day: int
    duration_days: int
    slope: float
    confidence: float


class InflectionDetector:
    """
    Detects when biomarker trajectories change from stable to trending.
    """

    def detect_inflection_points(
        self,
        patient_data: Dict[str, List[float]],
        timestamps: List[float]
    ) -> Dict[str, List[InflectionPoint]]:
        """
        Find all significant inflection points in patient trajectory.
        """
        all_inflections = {}

        for biomarker, values in patient_data.items():
            if biomarker.lower() in ['patient_id', 'id', 'day', 'time', 'timestamp', 'outcome', 'label']:
                continue

            if len(values) < 5:
                continue

            try:
                inflections = self._find_inflections(biomarker, values, timestamps)
                if inflections:
                    all_inflections[biomarker] = inflections
            except:
                continue

        return all_inflections

    def _find_inflections(
        self,
        biomarker: str,
        values: List[float],
        timestamps: List[float]
    ) -> List[InflectionPoint]:
        """
        Use multiple methods to find inflection points.
        """
        try:
            values = np.array([float(v) for v in values if v is not None])
            timestamps = np.array([float(t) for t in timestamps[:len(values)]])
        except:
            return []

        if len(values) < 5:
            return []

        inflections = []

        # Method 1: Second derivative analysis
        accel_inflections = self._second_derivative_method(biomarker, values, timestamps)
        inflections.extend(accel_inflections)

        # Method 2: Changepoint detection
        changepoint_inflections = self._changepoint_method(biomarker, values, timestamps)
        inflections.extend(changepoint_inflections)

        # Method 3: Moving average crossover
        ma_inflections = self._moving_average_method(biomarker, values, timestamps)
        inflections.extend(ma_inflections)

        # Deduplicate
        inflections = self._deduplicate_inflections(inflections)

        return sorted(inflections, key=lambda x: x.significance, reverse=True)

    def _second_derivative_method(
        self,
        biomarker: str,
        values: np.ndarray,
        timestamps: np.ndarray
    ) -> List[InflectionPoint]:
        """Find points where acceleration changes sign."""
        inflections = []

        if len(values) < 5:
            return inflections

        # Smooth the data
        smoothed = gaussian_filter1d(values, sigma=1)

        # First derivative
        d1 = np.gradient(smoothed, timestamps)

        # Second derivative
        d2 = np.gradient(d1, timestamps)

        # Find zero crossings
        for i in range(1, len(d2)):
            if d2[i-1] * d2[i] < 0:
                days_ago = timestamps[-1] - timestamps[i]
                change_since = (values[-1] - values[i]) / values[i] if values[i] != 0 else 0
                significance = min(abs(change_since), 1.0)

                inflections.append(InflectionPoint(
                    biomarker=biomarker,
                    day_index=i,
                    days_ago=float(days_ago),
                    value_at_inflection=float(values[i]),
                    current_value=float(values[-1]),
                    change_since_inflection=float(change_since * 100),
                    inflection_type='acceleration',
                    significance=float(significance)
                ))

        return inflections

    def _changepoint_method(
        self,
        biomarker: str,
        values: np.ndarray,
        timestamps: np.ndarray
    ) -> List[InflectionPoint]:
        """Detect statistical changepoints."""
        inflections = []

        if len(values) < 7:
            return inflections

        # CUSUM approach
        mean_val = np.mean(values[:len(values)//2])
        std_val = np.std(values[:len(values)//2]) or 1.0

        cusum = np.cumsum((values - mean_val) / std_val)

        max_idx = np.argmax(np.abs(cusum[len(values)//2:]))
        max_idx += len(values)//2

        if abs(cusum[max_idx]) > 2.0:
            days_ago = timestamps[-1] - timestamps[max_idx]
            change_since = (values[-1] - values[max_idx]) / values[max_idx] if values[max_idx] != 0 else 0

            inflections.append(InflectionPoint(
                biomarker=biomarker,
                day_index=int(max_idx),
                days_ago=float(days_ago),
                value_at_inflection=float(values[max_idx]),
                current_value=float(values[-1]),
                change_since_inflection=float(change_since * 100),
                inflection_type='breakout',
                significance=float(min(abs(cusum[max_idx]) / 5.0, 1.0))
            ))

        return inflections

    def _moving_average_method(
        self,
        biomarker: str,
        values: np.ndarray,
        timestamps: np.ndarray
    ) -> List[InflectionPoint]:
        """Detect trend crossovers."""
        inflections = []

        if len(values) < 7:
            return inflections

        short_window = min(3, len(values)//2)
        long_window = min(7, len(values))

        short_ma = np.convolve(values, np.ones(short_window)/short_window, mode='valid')
        long_ma = np.convolve(values, np.ones(long_window)/long_window, mode='valid')

        min_len = min(len(short_ma), len(long_ma))
        short_ma = short_ma[-min_len:]
        long_ma = long_ma[-min_len:]

        diff = short_ma - long_ma
        for i in range(1, len(diff)):
            if diff[i-1] * diff[i] < 0:
                orig_idx = len(values) - min_len + i
                days_ago = timestamps[-1] - timestamps[orig_idx]
                change_since = (values[-1] - values[orig_idx]) / values[orig_idx] if values[orig_idx] != 0 else 0

                inflections.append(InflectionPoint(
                    biomarker=biomarker,
                    day_index=int(orig_idx),
                    days_ago=float(days_ago),
                    value_at_inflection=float(values[orig_idx]),
                    current_value=float(values[-1]),
                    change_since_inflection=float(change_since * 100),
                    inflection_type='trend_reversal',
                    significance=float(min(abs(change_since), 1.0) * 0.8)
                ))

        return inflections

    def _deduplicate_inflections(
        self,
        inflections: List[InflectionPoint],
        day_tolerance: float = 1.0
    ) -> List[InflectionPoint]:
        """Merge nearby inflection points."""
        if not inflections:
            return []

        sorted_inf = sorted(inflections, key=lambda x: x.day_index)

        merged = [sorted_inf[0]]
        for inf in sorted_inf[1:]:
            if abs(inf.day_index - merged[-1].day_index) <= day_tolerance:
                if inf.significance > merged[-1].significance:
                    merged[-1] = inf
            else:
                merged.append(inf)

        return merged

    def identify_trajectory_phase(
        self,
        values: List[float],
        timestamps: List[float],
        inflection_points: List[InflectionPoint]
    ) -> TrajectoryPhase:
        """Determine what phase of disease progression the patient is in."""
        values = np.array(values)
        timestamps = np.array(timestamps)

        if len(values) >= 3:
            recent_slope = np.polyfit(timestamps[-3:], values[-3:], 1)[0]
        else:
            recent_slope = 0

        overall_slope = np.polyfit(timestamps, values, 1)[0]

        if not inflection_points:
            return TrajectoryPhase(
                phase='baseline',
                started_day=0,
                duration_days=int(timestamps[-1] - timestamps[0]),
                slope=float(overall_slope),
                confidence=0.8
            )

        recent_inflection = max(inflection_points, key=lambda x: -x.days_ago)

        if recent_slope > 0 and abs(recent_slope) > abs(overall_slope) * 1.5:
            phase = 'acceleration'
        elif recent_slope > 0:
            phase = 'early_rise'
        elif recent_slope < 0 and values[-1] > np.mean(values):
            phase = 'peak'
        elif recent_slope < 0:
            phase = 'recovery'
        else:
            phase = 'baseline'

        return TrajectoryPhase(
            phase=phase,
            started_day=recent_inflection.day_index,
            duration_days=int(recent_inflection.days_ago),
            slope=float(recent_slope),
            confidence=float(recent_inflection.significance)
        )
