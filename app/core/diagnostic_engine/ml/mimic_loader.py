"""
MIMIC-IV Data Loader for ML Training
Loads and prepares MIMIC-IV data for disease classification model training.
"""

import gzip
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# Default MIMIC-IV path
MIMIC_PATH = Path("F:/mimic-iv-3.1/mimic-iv-3.1")


class MIMICLoader:
    """
    Load MIMIC-IV data for ML training.

    Key files:
    - hosp/labevents.csv.gz: Lab test results
    - hosp/diagnoses_icd.csv.gz: ICD diagnoses
    - hosp/d_labitems.csv.gz: Lab item definitions
    - hosp/d_icd_diagnoses.csv.gz: ICD code definitions
    - hosp/patients.csv.gz: Patient demographics
    - hosp/admissions.csv.gz: Hospital admissions
    """

    def __init__(self, mimic_path: Path = MIMIC_PATH):
        self.mimic_path = mimic_path
        self.hosp_path = mimic_path / "hosp"
        self.icu_path = mimic_path / "icu"

        # Cached data
        self._labitems: Optional[pd.DataFrame] = None
        self._icd_codes: Optional[pd.DataFrame] = None
        self._patients: Optional[pd.DataFrame] = None

        # Lab item ID to name mapping
        self._labitem_map: Dict[int, str] = {}

        # Common lab items we care about for disease detection
        self.target_labs = {
            # Metabolic
            'glucose': [50931, 50809, 52027],
            'hba1c': [50852],
            # Renal
            'creatinine': [50912, 52546],
            'bun': [51006, 52024],
            'egfr': [],  # Calculated
            # Hepatic
            'alt': [50861],
            'ast': [50878],
            'bilirubin': [50885, 50883],
            'albumin': [50862],
            'alkaline_phosphatase': [50863],
            # Cardiac
            'troponin': [51002, 51003, 52642],
            'bnp': [50963],
            'ck_mb': [50911],
            # Hematologic
            'hemoglobin': [51222, 50811],
            'hematocrit': [51221, 50810],
            'wbc': [51301, 51300],
            'platelets': [51265],
            # Inflammatory
            'crp': [50889],
            'procalcitonin': [50976],
            'lactate': [50813],
            # Electrolytes
            'sodium': [50983, 50824],
            'potassium': [50971, 50822],
            'calcium': [50893],
            'magnesium': [50960],
            # Coagulation
            'inr': [51237],
            'pt': [51274],
            'ptt': [51275],
        }

        # Flatten lab IDs for filtering
        self._target_lab_ids = set()
        for ids in self.target_labs.values():
            self._target_lab_ids.update(ids)

    def check_availability(self) -> Dict[str, bool]:
        """Check which MIMIC files are available."""
        files = {
            'labevents': self.hosp_path / "labevents.csv.gz",
            'diagnoses_icd': self.hosp_path / "diagnoses_icd.csv.gz",
            'd_labitems': self.hosp_path / "d_labitems.csv.gz",
            'd_icd_diagnoses': self.hosp_path / "d_icd_diagnoses.csv.gz",
            'patients': self.hosp_path / "patients.csv.gz",
            'admissions': self.hosp_path / "admissions.csv.gz",
        }
        return {name: path.exists() for name, path in files.items()}

    def load_lab_definitions(self) -> pd.DataFrame:
        """Load lab item definitions."""
        if self._labitems is not None:
            return self._labitems

        path = self.hosp_path / "d_labitems.csv.gz"
        logger.info(f"Loading lab definitions from {path}")

        self._labitems = pd.read_csv(path, compression='gzip')

        # Build ID to name map
        for _, row in self._labitems.iterrows():
            self._labitem_map[row['itemid']] = row['label']

        logger.info(f"Loaded {len(self._labitems)} lab item definitions")
        return self._labitems

    def load_icd_definitions(self) -> pd.DataFrame:
        """Load ICD code definitions."""
        if self._icd_codes is not None:
            return self._icd_codes

        path = self.hosp_path / "d_icd_diagnoses.csv.gz"
        logger.info(f"Loading ICD definitions from {path}")

        self._icd_codes = pd.read_csv(path, compression='gzip')
        logger.info(f"Loaded {len(self._icd_codes)} ICD code definitions")
        return self._icd_codes

    def load_patients(self) -> pd.DataFrame:
        """Load patient demographics."""
        if self._patients is not None:
            return self._patients

        path = self.hosp_path / "patients.csv.gz"
        logger.info(f"Loading patients from {path}")

        self._patients = pd.read_csv(path, compression='gzip')
        logger.info(f"Loaded {len(self._patients)} patients")
        return self._patients

    def load_diagnoses(self, icd_prefix: str = None) -> pd.DataFrame:
        """
        Load diagnoses, optionally filtered by ICD prefix.

        Args:
            icd_prefix: Filter to ICD codes starting with this (e.g., 'E11' for diabetes)
        """
        path = self.hosp_path / "diagnoses_icd.csv.gz"
        logger.info(f"Loading diagnoses from {path}")

        df = pd.read_csv(path, compression='gzip')

        if icd_prefix:
            df = df[df['icd_code'].str.startswith(icd_prefix, na=False)]
            logger.info(f"Filtered to {len(df)} diagnoses with prefix '{icd_prefix}'")
        else:
            logger.info(f"Loaded {len(df)} total diagnoses")

        return df

    def load_labs_for_patients(
        self,
        subject_ids: List[int],
        lab_names: List[str] = None,
        chunk_size: int = 1000000
    ) -> pd.DataFrame:
        """
        Load lab events for specific patients.

        Args:
            subject_ids: List of patient IDs
            lab_names: List of lab names to include (uses target_labs if None)
            chunk_size: Rows to read per chunk

        Returns:
            DataFrame with lab results
        """
        path = self.hosp_path / "labevents.csv.gz"
        logger.info(f"Loading labs for {len(subject_ids)} patients...")

        subject_set = set(subject_ids)

        # Determine lab IDs to filter
        if lab_names:
            lab_ids = set()
            for name in lab_names:
                if name in self.target_labs:
                    lab_ids.update(self.target_labs[name])
        else:
            lab_ids = self._target_lab_ids

        # Load in chunks to handle large file
        chunks = []
        total_rows = 0

        for chunk in pd.read_csv(
            path,
            compression='gzip',
            chunksize=chunk_size,
            usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'value', 'valuenum', 'valueuom']
        ):
            # Filter to target patients and labs
            filtered = chunk[
                (chunk['subject_id'].isin(subject_set)) &
                (chunk['itemid'].isin(lab_ids))
            ]

            if len(filtered) > 0:
                chunks.append(filtered)
                total_rows += len(filtered)

            if total_rows % 1000000 == 0 and total_rows > 0:
                logger.info(f"  Processed {total_rows:,} matching lab rows...")

        if not chunks:
            return pd.DataFrame()

        result = pd.concat(chunks, ignore_index=True)

        # Add lab name column
        result['lab_name'] = result['itemid'].map(self._labitem_map)

        logger.info(f"Loaded {len(result):,} lab events for {len(subject_ids)} patients")
        return result

    def prepare_training_data(
        self,
        disease_icd: str,
        max_patients: int = 10000,
        control_ratio: float = 1.0
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data for a specific disease.

        Args:
            disease_icd: ICD code prefix for the target disease
            max_patients: Maximum patients per class
            control_ratio: Ratio of controls to cases

        Returns:
            (features_df, labels_series)
        """
        logger.info(f"Preparing training data for disease: {disease_icd}")

        # Load definitions first
        self.load_lab_definitions()

        # Get patients with the disease
        diagnoses = self.load_diagnoses(disease_icd)
        case_subjects = diagnoses['subject_id'].unique()
        logger.info(f"Found {len(case_subjects)} patients with {disease_icd}")

        # Limit cases
        if len(case_subjects) > max_patients:
            case_subjects = np.random.choice(case_subjects, max_patients, replace=False)

        # Get control patients (without this diagnosis)
        all_diagnoses = self.load_diagnoses()
        disease_subjects = set(diagnoses['subject_id'].unique())
        all_subjects = set(all_diagnoses['subject_id'].unique())
        control_subjects = list(all_subjects - disease_subjects)

        # Sample controls
        n_controls = int(len(case_subjects) * control_ratio)
        if len(control_subjects) > n_controls:
            control_subjects = np.random.choice(control_subjects, n_controls, replace=False)

        logger.info(f"Using {len(case_subjects)} cases, {len(control_subjects)} controls")

        # Load labs for all patients
        all_subjects = list(case_subjects) + list(control_subjects)
        labs = self.load_labs_for_patients(all_subjects)

        if labs.empty:
            logger.error("No lab data found!")
            return pd.DataFrame(), pd.Series()

        # Pivot to wide format (one row per patient)
        features = self._pivot_labs(labs)

        # Create labels
        labels = pd.Series(0, index=features.index)
        labels.loc[labels.index.isin(case_subjects)] = 1

        logger.info(f"Training data: {features.shape[0]} samples, {features.shape[1]} features")
        logger.info(f"Class distribution: {labels.value_counts().to_dict()}")

        return features, labels

    def _pivot_labs(self, labs: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot lab data to wide format with one row per patient.
        Uses the most recent value for each lab.
        """
        # Get most recent lab value per patient per lab type
        labs = labs.sort_values('charttime')
        latest = labs.groupby(['subject_id', 'lab_name']).last().reset_index()

        # Pivot to wide format
        pivoted = latest.pivot(
            index='subject_id',
            columns='lab_name',
            values='valuenum'
        )

        # Standardize column names
        pivoted.columns = [c.lower().replace(' ', '_') for c in pivoted.columns]

        return pivoted

    def get_common_diseases(self, min_patients: int = 1000) -> List[Dict]:
        """
        Get list of common diseases for model training.

        Returns:
            List of dicts with icd_code, name, patient_count
        """
        diagnoses = self.load_diagnoses()
        icd_defs = self.load_icd_definitions()

        # Count patients per ICD code (first 3 chars)
        diagnoses['icd_prefix'] = diagnoses['icd_code'].str[:3]
        counts = diagnoses.groupby('icd_prefix')['subject_id'].nunique()

        # Filter by minimum patients
        common = counts[counts >= min_patients].sort_values(ascending=False)

        # Build result with names
        icd_names = dict(zip(
            icd_defs['icd_code'].str[:3],
            icd_defs['long_title']
        ))

        results = []
        for icd, count in common.items():
            results.append({
                'icd_code': icd,
                'name': icd_names.get(icd, 'Unknown'),
                'patient_count': count
            })

        return results


# Singleton instance
_mimic_loader: Optional[MIMICLoader] = None


def get_mimic_loader() -> MIMICLoader:
    """Get singleton MIMIC loader instance."""
    global _mimic_loader
    if _mimic_loader is None:
        _mimic_loader = MIMICLoader()
    return _mimic_loader
