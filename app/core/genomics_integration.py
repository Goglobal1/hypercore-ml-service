"""
Genomics Integration Pipeline for HyperCore

This module provides:
1. GEO series_matrix.txt.gz parser
2. ClinVar variant_summary.txt.gz loader
3. Gene-to-phenotype mapper
4. Clinical trajectory correlation analysis

Data paths:
- GEO: F:/DATASETS/GENE_EXPRESSION/GEO_Datasets/
- ClinVar: F:/DATASETS/GENETICS/ClinVar/variant_summary.txt.gz
- Cohorts: pattern_library/
"""

import gzip
import json
import os
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

# Configuration
GEO_DATA_PATH = Path("F:/DATASETS/GENE_EXPRESSION/GEO_Datasets")
CLINVAR_PATH = Path("F:/DATASETS/GENETICS/ClinVar/variant_summary.txt.gz")
COHORT_PATH = Path("C:/Users/letsa/Documents/hypercore-ml-service/pattern_library")

# Gene symbol aliases for common clinical genes
GENE_ALIASES = {
    "APOE": ["APOE", "APO-E", "APOLIPOPROTEIN E"],
    "BRCA1": ["BRCA1", "BRCA-1", "BREAST CANCER 1"],
    "BRCA2": ["BRCA2", "BRCA-2", "BREAST CANCER 2"],
    "TP53": ["TP53", "P53", "TUMOR PROTEIN P53"],
    "EGFR": ["EGFR", "ERBB1", "HER1"],
    "KRAS": ["KRAS", "K-RAS", "KIRSTEN RAT SARCOMA"],
    "IL6": ["IL6", "IL-6", "INTERLEUKIN 6"],
    "TNF": ["TNF", "TNFA", "TNF-ALPHA"],
    "CRP": ["CRP", "C-REACTIVE PROTEIN"],
}

# Phenotype to ICD-10 mapping (geriatric-focused)
PHENOTYPE_ICD10_MAP = {
    "alzheimer": ["G30", "F00"],
    "dementia": ["F01", "F02", "F03"],
    "parkinson": ["G20", "G21"],
    "diabetes": ["E10", "E11", "E13", "E14"],
    "hypertension": ["I10", "I11", "I12", "I13"],
    "heart failure": ["I50"],
    "cardiomyopathy": ["I42", "I43"],
    "atrial fibrillation": ["I48"],
    "stroke": ["I60", "I61", "I62", "I63", "I64"],
    "chronic kidney": ["N18", "N19"],
    "osteoporosis": ["M80", "M81"],
    "sepsis": ["A40", "A41", "R65.2"],
    "pneumonia": ["J12", "J13", "J14", "J15", "J18"],
    "cancer": ["C00-C97"],
    "breast cancer": ["C50"],
    "lung cancer": ["C34"],
    "colon cancer": ["C18", "C19", "C20"],
    "prostate cancer": ["C61"],
    "macular degeneration": ["H35.3"],
    "glaucoma": ["H40"],
    "hearing loss": ["H90", "H91"],
    "frailty": ["R54"],
}


@dataclass
class GEOParseResult:
    """Result from parsing a GEO series_matrix file."""
    series_id: str
    title: str
    summary: str
    platform_id: str
    sample_ids: List[str]
    sample_metadata: Dict[str, Dict[str, str]]
    expression_matrix: Dict[str, Dict[str, float]]  # probe_id -> sample_id -> value
    probe_count: int
    sample_count: int
    pubmed_id: Optional[str] = None


@dataclass
class ClinVarVariantRecord:
    """Parsed ClinVar variant record."""
    allele_id: int
    gene_symbol: str
    gene_id: int
    variant_name: str
    variant_type: str
    clinical_significance: str
    phenotypes: List[str]
    review_status: str
    chromosome: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    rs_id: Optional[str] = None


class GEOParser:
    """Parser for GEO series_matrix.txt.gz files."""

    def __init__(self, data_path: Path = GEO_DATA_PATH):
        self.data_path = data_path
        self._file_cache: Dict[str, GEOParseResult] = {}

    def list_available_series(self) -> List[str]:
        """List all available GEO series files."""
        if not self.data_path.exists():
            logger.warning(f"GEO data path does not exist: {self.data_path}")
            return []

        series_files = []
        for f in self.data_path.glob("*_series_matrix.txt.gz"):
            # Extract series ID (e.g., GSE101709)
            match = re.match(r"(GSE\d+)", f.name)
            if match:
                series_files.append(match.group(1))
        return sorted(set(series_files))

    def parse_series_matrix(self, file_path: Path) -> Optional[GEOParseResult]:
        """
        Parse a GEO series_matrix.txt.gz file.

        Returns structured data including metadata and expression matrix.
        """
        cache_key = str(file_path)
        if cache_key in self._file_cache:
            return self._file_cache[cache_key]

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        try:
            series_metadata = {}
            sample_metadata = defaultdict(dict)
            sample_ids = []
            expression_matrix = {}
            in_matrix = False

            with gzip.open(file_path, 'rt', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()

                    if line == "!series_matrix_table_begin":
                        in_matrix = True
                        continue
                    elif line == "!series_matrix_table_end":
                        in_matrix = False
                        continue

                    if in_matrix:
                        # Expression data row
                        parts = line.split('\t')
                        if len(parts) > 1:
                            probe_id = parts[0].strip('"')
                            if probe_id == "ID_REF":
                                # Header row - get sample IDs
                                sample_ids = [s.strip('"') for s in parts[1:]]
                            else:
                                # Expression values
                                expression_matrix[probe_id] = {}
                                for i, val in enumerate(parts[1:]):
                                    if i < len(sample_ids):
                                        try:
                                            expression_matrix[probe_id][sample_ids[i]] = float(val)
                                        except (ValueError, TypeError):
                                            pass

                    elif line.startswith("!Series_"):
                        # Series metadata
                        key = line.split('\t')[0].replace("!Series_", "")
                        value = '\t'.join(line.split('\t')[1:]).strip('"')
                        series_metadata[key] = value

                    elif line.startswith("!Sample_"):
                        # Sample metadata
                        parts = line.split('\t')
                        key = parts[0].replace("!Sample_", "")
                        values = [v.strip('"') for v in parts[1:]]

                        # Get sample IDs from geo_accession if not yet set
                        if key == "geo_accession" and not sample_ids:
                            sample_ids = values

                        for i, val in enumerate(values):
                            if i < len(sample_ids) if sample_ids else True:
                                sample_id = sample_ids[i] if sample_ids else f"sample_{i}"
                                sample_metadata[sample_id][key] = val

            result = GEOParseResult(
                series_id=series_metadata.get("geo_accession", ""),
                title=series_metadata.get("title", ""),
                summary=series_metadata.get("summary", ""),
                platform_id=series_metadata.get("platform_id", ""),
                sample_ids=sample_ids,
                sample_metadata=dict(sample_metadata),
                expression_matrix=expression_matrix,
                probe_count=len(expression_matrix),
                sample_count=len(sample_ids),
                pubmed_id=series_metadata.get("pubmed_id")
            )

            self._file_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

    def search_gene_expression(
        self,
        gene_symbol: str,
        max_files: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for gene expression across available GEO datasets.

        Note: GEO series_matrix files use probe IDs, not gene symbols.
        This function searches probe IDs that contain the gene symbol.
        For accurate mapping, a platform annotation file would be needed.
        """
        results = []
        series_files = list(self.data_path.glob("*_series_matrix.txt.gz"))[:max_files]

        gene_upper = gene_symbol.upper()
        aliases = GENE_ALIASES.get(gene_upper, [gene_upper])

        for file_path in series_files:
            parsed = self.parse_series_matrix(file_path)
            if not parsed:
                continue

            # Search probe IDs (limited - ideally would use platform annotations)
            for probe_id, expression_data in parsed.expression_matrix.items():
                probe_upper = probe_id.upper()
                if any(alias in probe_upper for alias in aliases):
                    values = list(expression_data.values())
                    if values:
                        results.append({
                            "series_id": parsed.series_id,
                            "probe_id": probe_id,
                            "sample_count": len(values),
                            "mean_expression": sum(values) / len(values),
                            "min_expression": min(values),
                            "max_expression": max(values),
                            "platform": parsed.platform_id
                        })

        return results


class ClinVarLoader:
    """Loader for ClinVar variant_summary.txt.gz."""

    _load_lock = threading.Lock()  # Class-level lock for thread-safe loading

    def __init__(self, clinvar_path: Path = CLINVAR_PATH):
        self.clinvar_path = clinvar_path
        self._variant_cache: Dict[str, List[ClinVarVariantRecord]] = {}
        self._loaded = False
        self._gene_index: Dict[str, List[int]] = defaultdict(list)
        self._all_variants: List[ClinVarVariantRecord] = []

    def _load_variants(self):
        """Load and index all variants from ClinVar (thread-safe)."""
        # Quick check without lock
        if self._loaded:
            return

        # Thread-safe loading with lock - entire loading happens inside lock
        with ClinVarLoader._load_lock:
            # Double-check after acquiring lock (another thread may have loaded)
            if self._loaded:
                return

            if not self.clinvar_path.exists():
                logger.warning(f"ClinVar file not found: {self.clinvar_path}")
                self._loaded = True
                return

            logger.info(f"Loading ClinVar variants from {self.clinvar_path}")

            try:
                with gzip.open(self.clinvar_path, 'rt', encoding='utf-8', errors='replace') as f:
                    header = None
                    for line_num, line in enumerate(f):
                        if line_num == 0:
                            header = line.strip().split('\t')
                            continue

                        parts = line.strip().split('\t')
                        if len(parts) < 15:
                            continue

                        try:
                            record = ClinVarVariantRecord(
                                allele_id=int(parts[0]) if parts[0].isdigit() else 0,
                                gene_symbol=parts[4],
                                gene_id=int(parts[3]) if parts[3].isdigit() else 0,
                                variant_name=parts[2],
                                variant_type=parts[1],
                                clinical_significance=parts[6],
                                phenotypes=parts[13].split('|') if parts[13] else [],
                                review_status=parts[24] if len(parts) > 24 else "",
                                chromosome=parts[18] if len(parts) > 18 else None,
                                start=int(parts[19]) if len(parts) > 19 and parts[19].isdigit() else None,
                                end=int(parts[20]) if len(parts) > 20 and parts[20].isdigit() else None,
                                rs_id=parts[9] if len(parts) > 9 and parts[9] != "-" else None
                            )

                            idx = len(self._all_variants)
                            self._all_variants.append(record)
                            self._gene_index[record.gene_symbol.upper()].append(idx)

                        except Exception as e:
                            continue

                logger.info(f"Loaded {len(self._all_variants)} ClinVar variants")
                self._loaded = True

            except Exception as e:
                logger.error(f"Error loading ClinVar: {e}")
                self._loaded = True

    def get_variants_for_gene(
        self,
        gene_symbol: str,
        pathogenic_only: bool = True,
        max_variants: int = 100
    ) -> List[ClinVarVariantRecord]:
        """Get ClinVar variants for a gene."""
        self._load_variants()

        gene_upper = gene_symbol.upper()

        # Check cache
        cache_key = f"{gene_upper}_{pathogenic_only}"
        if cache_key in self._variant_cache:
            return self._variant_cache[cache_key][:max_variants]

        indices = self._gene_index.get(gene_upper, [])
        variants = [self._all_variants[i] for i in indices]

        if pathogenic_only:
            variants = [
                v for v in variants
                if "pathogenic" in v.clinical_significance.lower()
            ]

        # Sort by clinical significance
        variants.sort(key=lambda v: (
            0 if "pathogenic" in v.clinical_significance.lower() else 1,
            v.allele_id
        ))

        self._variant_cache[cache_key] = variants
        return variants[:max_variants]

    def get_phenotypes_for_gene(self, gene_symbol: str) -> List[str]:
        """Get all phenotypes associated with a gene's pathogenic variants."""
        variants = self.get_variants_for_gene(gene_symbol, pathogenic_only=True)

        phenotypes = set()
        for v in variants:
            for p in v.phenotypes:
                if p and p.lower() != "not provided":
                    phenotypes.add(p)

        return sorted(phenotypes)


class GenomicsIntegration:
    """
    Main integration engine connecting genomics data to clinical outcomes.
    """

    def __init__(self):
        self.geo_parser = GEOParser()
        self.clinvar_loader = ClinVarLoader()
        self.cohort_path = COHORT_PATH

    def get_gene_phenotype_associations(
        self,
        genes: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get phenotype associations for a list of genes.

        Returns associations with ICD-10 mappings where available.
        """
        associations = []

        for gene in genes:
            phenotypes = self.clinvar_loader.get_phenotypes_for_gene(gene)
            variants = self.clinvar_loader.get_variants_for_gene(gene)

            # Map phenotypes to ICD-10 codes
            icd10_codes = []
            for phenotype in phenotypes:
                phenotype_lower = phenotype.lower()
                for key, codes in PHENOTYPE_ICD10_MAP.items():
                    if key in phenotype_lower:
                        icd10_codes.extend(codes)

            associations.append({
                "gene": gene,
                "phenotypes": phenotypes[:20],  # Top 20
                "phenotype_count": len(phenotypes),
                "pathogenic_variant_count": len(variants),
                "icd10_codes": list(set(icd10_codes)),
                "top_variants": [
                    {
                        "name": v.variant_name[:100],
                        "significance": v.clinical_significance,
                        "type": v.variant_type
                    }
                    for v in variants[:5]
                ]
            })

        return associations

    def get_expression_patterns(
        self,
        genes: List[str],
        max_files: int = 10
    ) -> List[Dict[str, Any]]:
        """Get expression patterns for genes across GEO datasets."""
        patterns = []

        for gene in genes:
            results = self.geo_parser.search_gene_expression(gene, max_files)

            if results:
                all_means = [r["mean_expression"] for r in results]
                patterns.append({
                    "gene": gene,
                    "datasets_found": len(results),
                    "overall_mean": sum(all_means) / len(all_means) if all_means else 0,
                    "expression_range": [
                        min(r["min_expression"] for r in results),
                        max(r["max_expression"] for r in results)
                    ] if results else [0, 0],
                    "datasets": results[:5]  # Top 5 datasets
                })
            else:
                patterns.append({
                    "gene": gene,
                    "datasets_found": 0,
                    "overall_mean": 0,
                    "expression_range": [0, 0],
                    "datasets": []
                })

        return patterns

    def load_cohort_trajectories(
        self,
        cohort: str = "geriatric"
    ) -> Dict[str, Any]:
        """Load trajectory data for a cohort."""
        cohort_dir = self.cohort_path / cohort
        trajectories_dir = cohort_dir / "trajectories"

        if not trajectories_dir.exists():
            logger.warning(f"Trajectories not found for cohort: {cohort}")
            return {"trajectories": [], "count": 0}

        all_traj_file = trajectories_dir / "all_trajectories.json"
        if all_traj_file.exists():
            with open(all_traj_file, 'r') as f:
                data = json.load(f)
                return {
                    "trajectories": data if isinstance(data, list) else data.get("trajectories", []),
                    "count": len(data) if isinstance(data, list) else len(data.get("trajectories", []))
                }

        return {"trajectories": [], "count": 0}

    def analyze_gene_clinical_correlation(
        self,
        genes: List[str],
        cohort: str = "geriatric"
    ) -> Dict[str, Any]:
        """
        Analyze correlation between gene-associated phenotypes and
        clinical outcomes in a trajectory cohort.
        """
        # Get gene-phenotype associations
        associations = self.get_gene_phenotype_associations(genes)

        # Get ICD-10 codes for these genes
        gene_icd10 = {}
        for assoc in associations:
            gene_icd10[assoc["gene"]] = set(assoc["icd10_codes"])

        # Load cohort diagnoses
        cohort_dir = self.cohort_path / cohort
        diagnoses_file = cohort_dir / "diagnoses.csv"

        correlation_results = {
            "cohort": cohort,
            "genes_analyzed": genes,
            "correlations": []
        }

        if diagnoses_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(diagnoses_file)

                # Check for ICD code column
                icd_col = None
                for col in ["icd_code", "icd9_code", "icd10_code", "diagnosis_code"]:
                    if col in df.columns:
                        icd_col = col
                        break

                if icd_col:
                    all_codes = df[icd_col].dropna().astype(str).tolist()

                    for gene in genes:
                        gene_codes = gene_icd10.get(gene, set())
                        if not gene_codes:
                            continue

                        # Count matches
                        matches = sum(
                            1 for code in all_codes
                            if any(code.startswith(gc) for gc in gene_codes)
                        )

                        correlation_results["correlations"].append({
                            "gene": gene,
                            "icd10_codes": list(gene_codes),
                            "matched_diagnoses": matches,
                            "total_diagnoses": len(all_codes),
                            "prevalence": matches / len(all_codes) if all_codes else 0
                        })

            except Exception as e:
                logger.error(f"Error analyzing diagnoses: {e}")

        return correlation_results

    def comprehensive_analysis(
        self,
        genes: List[str],
        cohort: str = "geriatric",
        include_expression: bool = True,
        include_variants: bool = True,
        max_expression_files: int = 10
    ) -> Dict[str, Any]:
        """
        Perform comprehensive genomics analysis.

        Combines:
        - Gene-phenotype associations from ClinVar
        - Expression patterns from GEO
        - Clinical correlations from trajectory data
        """
        result = {
            "genes_analyzed": genes,
            "cohort": cohort,
            "gene_phenotype_associations": [],
            "expression_patterns": [],
            "variant_impacts": [],
            "clinical_correlations": {},
            "analysis_metadata": {
                "geo_path": str(GEO_DATA_PATH),
                "clinvar_path": str(CLINVAR_PATH),
                "cohort_path": str(self.cohort_path / cohort)
            }
        }

        # Get phenotype associations
        if include_variants:
            result["gene_phenotype_associations"] = self.get_gene_phenotype_associations(genes)

            # Extract variant impacts
            for assoc in result["gene_phenotype_associations"]:
                result["variant_impacts"].extend([
                    {
                        "gene": assoc["gene"],
                        **v
                    }
                    for v in assoc.get("top_variants", [])
                ])

        # Get expression patterns
        if include_expression:
            result["expression_patterns"] = self.get_expression_patterns(
                genes, max_expression_files
            )

        # Get clinical correlations
        result["clinical_correlations"] = self.analyze_gene_clinical_correlation(
            genes, cohort
        )

        # Calculate cohort overlap statistics
        traj_data = self.load_cohort_trajectories(cohort)
        result["cohort_overlap"] = {
            "cohort": cohort,
            "trajectory_count": traj_data["count"],
            "genes_with_expression_data": sum(
                1 for p in result["expression_patterns"]
                if p.get("datasets_found", 0) > 0
            ),
            "genes_with_variants": sum(
                1 for a in result["gene_phenotype_associations"]
                if a.get("pathogenic_variant_count", 0) > 0
            )
        }

        return result


# Singleton instances
_geo_parser = None
_clinvar_loader = None
_integration = None


def get_geo_parser() -> GEOParser:
    """Get singleton GEO parser instance."""
    global _geo_parser
    if _geo_parser is None:
        _geo_parser = GEOParser()
    return _geo_parser


def get_clinvar_loader() -> ClinVarLoader:
    """Get singleton ClinVar loader instance."""
    global _clinvar_loader
    if _clinvar_loader is None:
        _clinvar_loader = ClinVarLoader()
    return _clinvar_loader


def get_genomics_integration() -> GenomicsIntegration:
    """Get singleton genomics integration instance."""
    global _integration
    if _integration is None:
        _integration = GenomicsIntegration()
    return _integration


# API-friendly functions
def get_gene_expression(
    gene: str,
    max_files: int = 5
) -> Dict[str, Any]:
    """API function to get gene expression data."""
    parser = get_geo_parser()
    results = parser.search_gene_expression(gene, max_files)

    return {
        "gene": gene,
        "probe_ids": list(set(r["probe_id"] for r in results)),
        "samples": results,
        "statistics": {
            "mean_expression": (
                sum(r["mean_expression"] for r in results) / len(results)
                if results else 0
            ),
            "dataset_count": len(results)
        },
        "series_count": len(set(r["series_id"] for r in results)),
        "sample_count": sum(r["sample_count"] for r in results)
    }


def get_gene_variants(
    gene: str,
    pathogenic_only: bool = True,
    max_variants: int = 100
) -> Dict[str, Any]:
    """API function to get gene variants."""
    loader = get_clinvar_loader()
    variants = loader.get_variants_for_gene(gene, pathogenic_only, max_variants)

    phenotypes = set()
    for v in variants:
        phenotypes.update(v.phenotypes)

    return {
        "gene": gene,
        "variants": [asdict(v) for v in variants],
        "pathogenic_count": sum(
            1 for v in variants
            if "pathogenic" in v.clinical_significance.lower()
        ),
        "total_count": len(variants),
        "phenotypes": sorted(p for p in phenotypes if p and p.lower() != "not provided")
    }


def analyze_genomics(
    genes: List[str],
    cohort: str = "geriatric",
    include_variants: bool = True,
    include_expression: bool = True,
    max_expression_files: int = 10
) -> Dict[str, Any]:
    """API function for comprehensive genomics analysis."""
    integration = get_genomics_integration()
    return integration.comprehensive_analysis(
        genes=genes,
        cohort=cohort,
        include_expression=include_expression,
        include_variants=include_variants,
        max_expression_files=max_expression_files
    )
