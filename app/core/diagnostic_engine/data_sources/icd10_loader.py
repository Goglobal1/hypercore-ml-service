"""
ICD-10 Code Loader for Diagnostic Engine
Loads and indexes ICD-10 codes from MIMIC-IV for disease classification.

Contains 97,000+ ICD-10 diagnosis codes with hierarchical structure.
"""

import gzip
import logging
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

# Default path to MIMIC ICD data - uses environment variable or relative path
MIMIC_ICD_PATH = Path(os.environ.get(
    'MIMIC_ICD_PATH',
    os.environ.get('MIMIC_PATH', './data/mimic-iv') + '/hosp/d_icd_diagnoses.csv.gz'
))


class ICD10Loader:
    """
    Load and index ICD-10 codes for disease detection.

    Provides:
    - Code → Description lookup
    - Description search
    - Hierarchical category navigation
    - Code validation
    """

    _instance = None

    def __init__(self, icd_path: Path = MIMIC_ICD_PATH):
        self.icd_path = icd_path

        # Main data
        self.codes: Dict[str, Dict] = {}  # code -> {code, title, category}
        self.categories: Dict[str, List[str]] = defaultdict(list)  # category -> codes

        # Search indices
        self._title_index: Dict[str, List[str]] = defaultdict(list)  # word -> codes
        self._code_prefix_index: Dict[str, List[str]] = defaultdict(list)  # prefix -> codes

        # Statistics
        self.total_codes = 0
        self.icd10_codes = 0
        self.icd9_codes = 0

        self._loaded = False
        self._load_error = None

    def load(self) -> bool:
        """Load ICD-10 data (thread-safe singleton pattern)."""
        if self._loaded:
            return True

        if not self.icd_path.exists():
            self._load_error = f"ICD file not found: {self.icd_path}"
            logger.warning(self._load_error)
            return False

        logger.info(f"[ICD-10] Loading from {self.icd_path}...")

        try:
            df = pd.read_csv(self.icd_path, compression='gzip')

            # Filter to ICD-10 only
            df_icd10 = df[df['icd_version'] == 10].copy()

            self.total_codes = len(df)
            self.icd10_codes = len(df_icd10)
            self.icd9_codes = len(df) - len(df_icd10)

            # Build main code dictionary
            for _, row in df_icd10.iterrows():
                code = str(row['icd_code']).strip()
                title = str(row['long_title']).strip()

                # Determine category (first letter or first 3 chars)
                category = self._get_category(code)

                self.codes[code] = {
                    'code': code,
                    'title': title,
                    'category': category,
                    'category_name': self._get_category_name(category)
                }

                # Add to category index
                self.categories[category].append(code)

                # Build search index (words in title)
                words = title.lower().split()
                for word in words:
                    if len(word) >= 3:  # Skip short words
                        self._title_index[word].append(code)

                # Build prefix index
                for i in range(1, min(len(code) + 1, 5)):
                    prefix = code[:i]
                    self._code_prefix_index[prefix].append(code)

            self._loaded = True
            logger.info(f"[ICD-10] Loaded {self.icd10_codes:,} ICD-10 codes in {len(self.categories)} categories")

            return True

        except Exception as e:
            self._load_error = str(e)
            logger.error(f"[ICD-10] Load error: {e}")
            return False

    def _get_category(self, code: str) -> str:
        """Get category prefix for an ICD-10 code."""
        if len(code) >= 3:
            return code[:3]
        return code

    def _get_category_name(self, category: str) -> str:
        """Get human-readable category name."""
        # ICD-10 chapter mappings (simplified)
        chapter_map = {
            'A': 'Infectious diseases',
            'B': 'Infectious diseases',
            'C': 'Neoplasms',
            'D': 'Blood/immune disorders',
            'E': 'Endocrine/metabolic',
            'F': 'Mental/behavioral',
            'G': 'Nervous system',
            'H': 'Eye/ear',
            'I': 'Circulatory system',
            'J': 'Respiratory system',
            'K': 'Digestive system',
            'L': 'Skin/subcutaneous',
            'M': 'Musculoskeletal',
            'N': 'Genitourinary',
            'O': 'Pregnancy/childbirth',
            'P': 'Perinatal conditions',
            'Q': 'Congenital malformations',
            'R': 'Symptoms/signs/abnormal findings',
            'S': 'Injury/poisoning',
            'T': 'Injury/poisoning',
            'V': 'External causes',
            'W': 'External causes',
            'X': 'External causes',
            'Y': 'External causes',
            'Z': 'Health status/services',
        }

        if category and category[0].upper() in chapter_map:
            return chapter_map[category[0].upper()]
        return 'Other'

    def get_code(self, code: str) -> Optional[Dict]:
        """
        Get details for a specific ICD-10 code.

        Args:
            code: ICD-10 code (e.g., 'E11.9')

        Returns:
            Dict with code details or None
        """
        if not self._loaded:
            self.load()

        # Normalize code (remove dots, uppercase)
        normalized = code.replace('.', '').upper()
        return self.codes.get(normalized)

    def search(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        Search for ICD-10 codes by description.

        Args:
            query: Search term
            max_results: Maximum results to return

        Returns:
            List of matching codes with details
        """
        if not self._loaded:
            self.load()

        query_lower = query.lower()
        query_words = query_lower.split()

        # Score codes by match quality
        scores: Dict[str, float] = defaultdict(float)

        # Check for code prefix match
        query_upper = query.upper().replace('.', '')
        if query_upper in self._code_prefix_index:
            for code in self._code_prefix_index[query_upper]:
                scores[code] += 10.0  # High score for code match

        # Check for word matches in title
        for word in query_words:
            if len(word) >= 3 and word in self._title_index:
                for code in self._title_index[word]:
                    scores[code] += 1.0

        # Sort by score and return top results
        sorted_codes = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)

        results = []
        for code in sorted_codes[:max_results]:
            if code in self.codes:
                result = self.codes[code].copy()
                result['score'] = scores[code]
                results.append(result)

        return results

    def get_children(self, code: str) -> List[Dict]:
        """
        Get all sub-codes under a category.

        Args:
            code: ICD-10 code or prefix (e.g., 'E11' for all diabetes codes)

        Returns:
            List of child codes
        """
        if not self._loaded:
            self.load()

        prefix = code.replace('.', '').upper()
        children = []

        for full_code, details in self.codes.items():
            if full_code.startswith(prefix) and full_code != prefix:
                children.append(details)

        return sorted(children, key=lambda x: x['code'])

    def get_category_codes(self, category: str) -> List[Dict]:
        """Get all codes in a category."""
        if not self._loaded:
            self.load()

        codes = self.categories.get(category.upper(), [])
        return [self.codes[c] for c in codes if c in self.codes]

    def validate_code(self, code: str) -> bool:
        """Check if a code is valid ICD-10."""
        if not self._loaded:
            self.load()

        normalized = code.replace('.', '').upper()
        return normalized in self.codes

    def get_all_categories(self) -> List[Dict]:
        """Get list of all ICD-10 categories."""
        if not self._loaded:
            self.load()

        result = []
        for category in sorted(self.categories.keys()):
            result.append({
                'category': category,
                'name': self._get_category_name(category),
                'code_count': len(self.categories[category])
            })
        return result

    def get_stats(self) -> Dict:
        """Get loader statistics."""
        return {
            'loaded': self._loaded,
            'error': self._load_error,
            'total_codes': self.total_codes,
            'icd10_codes': self.icd10_codes,
            'icd9_codes': self.icd9_codes,
            'categories': len(self.categories),
            'icd_path': str(self.icd_path)
        }

    def map_disease_to_icd(self, disease_name: str) -> Optional[str]:
        """
        Attempt to map a disease name to an ICD-10 code.

        Args:
            disease_name: Disease name to map

        Returns:
            Best matching ICD-10 code or None
        """
        results = self.search(disease_name, max_results=5)

        if results:
            # Return the best match
            return results[0]['code']

        return None


# Singleton instance
_icd10_loader: Optional[ICD10Loader] = None


def get_icd10_loader() -> ICD10Loader:
    """Get singleton ICD-10 loader instance."""
    global _icd10_loader
    if _icd10_loader is None:
        _icd10_loader = ICD10Loader()
    return _icd10_loader
