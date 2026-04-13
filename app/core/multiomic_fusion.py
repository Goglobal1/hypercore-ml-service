"""
Multi-Omic Fusion Engine for HyperCore

Provides unified query capabilities across multiple data sources:
- Genomics: GEO expression, ClinVar variants
- Proteomics: Human Protein Atlas
- Clinical: MIMIC-IV trajectories, eICU, Northwestern ICU
- Pharmacological: FDA FAERS, ClinicalTrials AACT
- Population: NHANES
- Surveillance: WHO, CDC WONDER

Key capabilities:
1. Unified cross-source queries
2. Gene-centric, disease-centric, drug-centric lookups
3. Cross-layer correlation analysis
4. Data source status monitoring
"""

import os
import gzip
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from collections import defaultdict

logger = logging.getLogger(__name__)

# Base directory for relative fallback paths
# Go up 3 levels: multiomic_fusion.py -> core -> app -> project_root
_BASE_DIR = Path(__file__).parent.parent.parent

# Data source paths configuration - use environment variables with fallbacks
DATA_PATHS = {
    "geo": Path(os.environ.get('GEO_DATA_PATH', _BASE_DIR / 'data' / 'geo')),
    "clinvar": Path(os.environ.get('CLINVAR_PATH', _BASE_DIR / 'data' / 'clinvar')),
    "hpa": Path(os.environ.get('HPA_PATH', _BASE_DIR / 'data' / 'hpa')),
    "mimic": Path(os.environ.get('MIMIC_PATH', _BASE_DIR / 'data' / 'mimic')),
    "eicu": Path(os.environ.get('EICU_PATH', _BASE_DIR / 'data' / 'eicu')),
    "northwestern": Path(os.environ.get('NORTHWESTERN_PATH', _BASE_DIR / 'data' / 'northwestern')),
    "faers": Path(os.environ.get('FAERS_PATH', _BASE_DIR / 'data' / 'faers')),
    "aact": Path(os.environ.get('AACT_PATH', _BASE_DIR / 'data' / 'aact')),
    "nhanes": Path(os.environ.get('NHANES_PATH', _BASE_DIR / 'data' / 'nhanes')),
    "who": Path(os.environ.get('WHO_PATH', _BASE_DIR / 'data' / 'who')),
    "cdc_wonder": Path(os.environ.get('CDC_WONDER_PATH', _BASE_DIR / 'data' / 'cdc_wonder')),
    "cohorts": Path(os.environ.get('COHORT_PATH', _BASE_DIR.parent / 'pattern_library')),
}

# Omic layer mapping
SOURCE_LAYERS = {
    "geo": "transcriptomic",
    "clinvar": "genomic",
    "hpa": "proteomic",
    "mimic": "clinical",
    "eicu": "clinical",
    "northwestern": "clinical",
    "faers": "pharmacological",
    "aact": "pharmacological",
    "nhanes": "epidemiological",
    "who": "epidemiological",
    "cdc_wonder": "epidemiological",
}

# Gene-disease associations for cross-referencing
GENE_DISEASE_MAP = {
    "APOE": ["alzheimer", "dementia", "hyperlipidemia", "cardiovascular"],
    "BRCA1": ["breast cancer", "ovarian cancer", "hereditary cancer"],
    "BRCA2": ["breast cancer", "ovarian cancer", "pancreatic cancer"],
    "TP53": ["cancer", "li-fraumeni syndrome", "tumor"],
    "EGFR": ["lung cancer", "glioblastoma", "colorectal cancer"],
    "KRAS": ["pancreatic cancer", "colorectal cancer", "lung cancer"],
    "IL6": ["inflammation", "sepsis", "rheumatoid arthritis", "covid"],
    "TNF": ["inflammation", "autoimmune", "sepsis", "crohn"],
    "ACE2": ["covid", "hypertension", "heart failure"],
    "CFTR": ["cystic fibrosis", "pancreatitis"],
    "HBB": ["sickle cell", "thalassemia", "anemia"],
    "F5": ["thrombophilia", "dvt", "pulmonary embolism"],
    "MTHFR": ["homocystinuria", "neural tube defects", "cardiovascular"],
}

# Drug-gene associations
DRUG_GENE_MAP = {
    "warfarin": ["CYP2C9", "VKORC1", "CYP4F2"],
    "clopidogrel": ["CYP2C19", "ABCB1"],
    "metformin": ["SLC22A1", "SLC22A2", "SLC47A1"],
    "statins": ["SLCO1B1", "APOE", "HMGCR"],
    "tamoxifen": ["CYP2D6", "ESR1"],
    "irinotecan": ["UGT1A1"],
    "azathioprine": ["TPMT", "NUDT15"],
    "carbamazepine": ["HLA-B", "HLA-A"],
    "abacavir": ["HLA-B*57:01"],
}


@dataclass
class SourceStatus:
    """Status of a data source."""
    source: str
    available: bool
    path: str
    file_count: int
    layer: str
    description: str


class DataSourceConnector:
    """Base connector for data sources."""

    def __init__(self, source_name: str, path: Path):
        self.source_name = source_name
        self.path = path
        self.layer = SOURCE_LAYERS.get(source_name, "unknown")

    def is_available(self) -> bool:
        return self.path.exists()

    def get_file_count(self) -> int:
        if not self.is_available():
            return 0
        return sum(1 for _ in self.path.rglob("*") if _.is_file())

    def get_status(self) -> SourceStatus:
        return SourceStatus(
            source=self.source_name,
            available=self.is_available(),
            path=str(self.path),
            file_count=self.get_file_count() if self.is_available() else 0,
            layer=self.layer,
            description=f"{self.source_name.upper()} data source"
        )


class GEOConnector(DataSourceConnector):
    """Connector for GEO expression data."""

    def __init__(self):
        super().__init__("geo", DATA_PATHS["geo"])

    def search_gene(self, gene: str, max_files: int = 5) -> List[Dict[str, Any]]:
        """Search for gene expression across GEO datasets."""
        if not self.is_available():
            return []

        # Delegate to genomics_integration module
        try:
            from app.core.genomics_integration import get_gene_expression
            result = get_gene_expression(gene, max_files)
            return result.get("samples", [])
        except ImportError:
            return []


class ClinVarConnector(DataSourceConnector):
    """Connector for ClinVar variant data."""

    def __init__(self):
        super().__init__("clinvar", DATA_PATHS["clinvar"])

    def search_gene(self, gene: str, max_variants: int = 50) -> List[Dict[str, Any]]:
        """Search for variants in a gene."""
        if not self.is_available():
            return []

        try:
            from app.core.genomics_integration import get_gene_variants
            result = get_gene_variants(gene, pathogenic_only=True, max_variants=max_variants)
            return result.get("variants", [])
        except ImportError:
            return []


class HPAConnector(DataSourceConnector):
    """Connector for Human Protein Atlas data."""

    def __init__(self):
        super().__init__("hpa", DATA_PATHS["hpa"])
        self._data_cache = {}

    def search_gene(self, gene: str) -> List[Dict[str, Any]]:
        """Search for protein expression data for a gene."""
        if not self.is_available():
            return []

        results = []
        gene_upper = gene.upper()

        # Search through HPA files
        for f in self.path.glob("*.tsv"):
            try:
                with open(f, 'r', encoding='utf-8', errors='replace') as file:
                    header = file.readline().strip().split('\t')
                    gene_col = None
                    for i, col in enumerate(header):
                        if 'gene' in col.lower():
                            gene_col = i
                            break

                    if gene_col is None:
                        continue

                    for line in file:
                        parts = line.strip().split('\t')
                        if len(parts) > gene_col and gene_upper in parts[gene_col].upper():
                            results.append({
                                "source_file": f.name,
                                "gene": parts[gene_col] if gene_col < len(parts) else "",
                                "data": dict(zip(header[:len(parts)], parts))
                            })
                            if len(results) >= 50:
                                return results
            except Exception as e:
                logger.warning(f"Error reading HPA file {f}: {e}")
                continue

        return results


class FAERSConnector(DataSourceConnector):
    """Connector for FDA FAERS adverse event data."""

    def __init__(self):
        super().__init__("faers", DATA_PATHS["faers"])

    def search_drug(self, drug: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for adverse events related to a drug."""
        if not self.is_available():
            return []

        results = []
        drug_upper = drug.upper()

        # Search FAERS files
        for f in self.path.glob("*.txt"):
            if "drug" not in f.name.lower():
                continue

            try:
                with open(f, 'r', encoding='utf-8', errors='replace') as file:
                    header = file.readline().strip().split('$')
                    drugname_col = None
                    for i, col in enumerate(header):
                        if 'drugname' in col.lower() or 'drug_name' in col.lower():
                            drugname_col = i
                            break

                    if drugname_col is None:
                        continue

                    for line in file:
                        parts = line.strip().split('$')
                        if len(parts) > drugname_col:
                            if drug_upper in parts[drugname_col].upper():
                                results.append({
                                    "source_file": f.name,
                                    "drug": parts[drugname_col],
                                    "data": dict(zip(header[:len(parts)], parts))
                                })
                                if len(results) >= max_results:
                                    return results
            except Exception as e:
                logger.warning(f"Error reading FAERS file {f}: {e}")
                continue

        return results


class AACTConnector(DataSourceConnector):
    """Connector for ClinicalTrials.gov AACT data."""

    def __init__(self):
        super().__init__("aact", DATA_PATHS["aact"])

    def search_condition(self, condition: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for clinical trials by condition."""
        if not self.is_available():
            return []

        results = []
        condition_lower = condition.lower()

        # Look for conditions or studies files
        for pattern in ["*conditions*", "*studies*", "*browse_conditions*"]:
            for f in self.path.glob(pattern):
                if not f.suffix in ['.txt', '.csv', '.tsv']:
                    continue

                try:
                    delimiter = ',' if f.suffix == '.csv' else '\t'
                    with open(f, 'r', encoding='utf-8', errors='replace') as file:
                        header = file.readline().strip().split(delimiter)

                        for line in file:
                            if condition_lower in line.lower():
                                parts = line.strip().split(delimiter)
                                results.append({
                                    "source_file": f.name,
                                    "data": dict(zip(header[:len(parts)], parts))
                                })
                                if len(results) >= max_results:
                                    return results
                except Exception as e:
                    logger.warning(f"Error reading AACT file {f}: {e}")
                    continue

        return results


class NHANESConnector(DataSourceConnector):
    """Connector for NHANES population data."""

    def __init__(self):
        super().__init__("nhanes", DATA_PATHS["nhanes"])

    def list_datasets(self) -> List[str]:
        """List available NHANES datasets."""
        if not self.is_available():
            return []

        return [f.stem for f in self.path.glob("*.xpt")]

    def search_variable(self, variable: str) -> List[Dict[str, Any]]:
        """Search for a variable across NHANES datasets."""
        # NHANES uses SAS XPT format - would need pyreadstat to read
        # For now, return dataset list containing the variable name
        results = []
        var_upper = variable.upper()

        for f in self.path.glob("*.xpt"):
            if var_upper in f.stem.upper():
                results.append({
                    "dataset": f.stem,
                    "file": f.name,
                    "size_mb": f.stat().st_size / (1024 * 1024)
                })

        return results


class WHOConnector(DataSourceConnector):
    """Connector for WHO surveillance data."""

    def __init__(self):
        super().__init__("who", DATA_PATHS["who"])

    def list_indicators(self) -> List[str]:
        """List available WHO indicators."""
        if not self.is_available():
            return []

        return [f.stem.replace("_ALL_LATEST", "") for f in self.path.glob("*_ALL_LATEST.csv")]

    def search_indicator(self, indicator: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search WHO data for an indicator."""
        if not self.is_available():
            return []

        results = []
        indicator_upper = indicator.upper()

        for f in self.path.glob("*.csv"):
            if indicator_upper not in f.stem.upper():
                continue

            try:
                with open(f, 'r', encoding='utf-8', errors='replace') as file:
                    header = file.readline().strip().split(',')

                    for line in file:
                        parts = line.strip().split(',')
                        results.append({
                            "indicator": f.stem,
                            "data": dict(zip(header[:len(parts)], parts))
                        })
                        if len(results) >= max_results:
                            return results
            except Exception as e:
                logger.warning(f"Error reading WHO file {f}: {e}")
                continue

        return results


class MultiOmicFusionEngine:
    """
    Main fusion engine for cross-source queries and analysis.
    """

    def __init__(self):
        self.connectors = {
            "geo": GEOConnector(),
            "clinvar": ClinVarConnector(),
            "hpa": HPAConnector(),
            "faers": FAERSConnector(),
            "aact": AACTConnector(),
            "nhanes": NHANESConnector(),
            "who": WHOConnector(),
        }

    def get_all_source_status(self) -> List[Dict[str, Any]]:
        """Get status of all data sources."""
        statuses = []
        for name, connector in self.connectors.items():
            status = connector.get_status()
            statuses.append(asdict(status))
        return statuses

    def gene_centric_query(
        self,
        genes: List[str],
        sources: Optional[List[str]] = None,
        max_results_per_source: int = 50
    ) -> Dict[str, Any]:
        """
        Perform gene-centric query across multiple sources.

        Aggregates data about genes from:
        - GEO (expression)
        - ClinVar (variants)
        - HPA (protein expression)
        """
        start_time = time.time()
        results = {
            "query_type": "gene_centric",
            "genes": genes,
            "sources_queried": [],
            "results_by_source": {},
            "cross_references": [],
            "summary": {}
        }

        # Determine which sources to query
        gene_sources = ["geo", "clinvar", "hpa"]
        if sources:
            gene_sources = [s for s in sources if s in gene_sources]

        for gene in genes:
            gene_upper = gene.upper()
            gene_results = {}

            # GEO expression
            if "geo" in gene_sources:
                connector = self.connectors["geo"]
                if connector.is_available():
                    geo_data = connector.search_gene(gene_upper, max_files=5)
                    gene_results["geo"] = {
                        "source": "geo",
                        "layer": "transcriptomic",
                        "record_count": len(geo_data),
                        "data": geo_data[:max_results_per_source]
                    }
                    if "geo" not in results["sources_queried"]:
                        results["sources_queried"].append("geo")

            # ClinVar variants
            if "clinvar" in gene_sources:
                connector = self.connectors["clinvar"]
                if connector.is_available():
                    clinvar_data = connector.search_gene(gene_upper, max_variants=max_results_per_source)
                    gene_results["clinvar"] = {
                        "source": "clinvar",
                        "layer": "genomic",
                        "record_count": len(clinvar_data),
                        "data": clinvar_data[:max_results_per_source]
                    }
                    if "clinvar" not in results["sources_queried"]:
                        results["sources_queried"].append("clinvar")

            # HPA protein expression
            if "hpa" in gene_sources:
                connector = self.connectors["hpa"]
                if connector.is_available():
                    hpa_data = connector.search_gene(gene_upper)
                    gene_results["hpa"] = {
                        "source": "hpa",
                        "layer": "proteomic",
                        "record_count": len(hpa_data),
                        "data": hpa_data[:max_results_per_source]
                    }
                    if "hpa" not in results["sources_queried"]:
                        results["sources_queried"].append("hpa")

            results["results_by_source"][gene_upper] = gene_results

            # Add cross-references (gene-disease associations)
            if gene_upper in GENE_DISEASE_MAP:
                results["cross_references"].append({
                    "gene": gene_upper,
                    "associated_diseases": GENE_DISEASE_MAP[gene_upper],
                    "evidence": "curated_association"
                })

        # Summary
        results["summary"] = {
            "genes_queried": len(genes),
            "sources_queried": len(results["sources_queried"]),
            "total_records": sum(
                sum(s.get("record_count", 0) for s in gene_data.values())
                for gene_data in results["results_by_source"].values()
            )
        }
        results["execution_time_ms"] = (time.time() - start_time) * 1000

        return results

    def disease_centric_query(
        self,
        disease: str,
        icd_codes: Optional[List[str]] = None,
        max_results_per_source: int = 50
    ) -> Dict[str, Any]:
        """
        Perform disease-centric query across sources.

        Finds:
        - Related genes (via gene-disease map)
        - Clinical trials (AACT)
        - Adverse events (FAERS)
        - WHO surveillance data
        """
        start_time = time.time()
        disease_lower = disease.lower()

        results = {
            "query_type": "disease_centric",
            "disease": disease,
            "icd_codes": icd_codes or [],
            "sources_queried": [],
            "related_genes": [],
            "results_by_source": {},
            "summary": {}
        }

        # Find related genes
        for gene, diseases in GENE_DISEASE_MAP.items():
            if any(disease_lower in d.lower() for d in diseases):
                results["related_genes"].append(gene)

        # Search AACT clinical trials
        aact_connector = self.connectors["aact"]
        if aact_connector.is_available():
            aact_data = aact_connector.search_condition(disease, max_results_per_source)
            results["results_by_source"]["aact"] = {
                "source": "aact",
                "layer": "pharmacological",
                "record_count": len(aact_data),
                "data": aact_data
            }
            results["sources_queried"].append("aact")

        # Search WHO data
        who_connector = self.connectors["who"]
        if who_connector.is_available():
            who_data = who_connector.search_indicator(disease, max_results_per_source)
            results["results_by_source"]["who"] = {
                "source": "who",
                "layer": "epidemiological",
                "record_count": len(who_data),
                "data": who_data
            }
            results["sources_queried"].append("who")

        # Summary
        results["summary"] = {
            "disease": disease,
            "related_genes_count": len(results["related_genes"]),
            "sources_queried": len(results["sources_queried"]),
            "total_records": sum(
                s.get("record_count", 0)
                for s in results["results_by_source"].values()
            )
        }
        results["execution_time_ms"] = (time.time() - start_time) * 1000

        return results

    def drug_centric_query(
        self,
        drug: str,
        max_results_per_source: int = 50
    ) -> Dict[str, Any]:
        """
        Perform drug-centric query across sources.

        Finds:
        - Related genes (pharmacogenomics)
        - Adverse events (FAERS)
        - Clinical trials (AACT)
        """
        start_time = time.time()
        drug_lower = drug.lower()

        results = {
            "query_type": "drug_centric",
            "drug": drug,
            "sources_queried": [],
            "pharmacogenomic_genes": [],
            "results_by_source": {},
            "summary": {}
        }

        # Find pharmacogenomic genes
        for drug_name, genes in DRUG_GENE_MAP.items():
            if drug_lower in drug_name.lower():
                results["pharmacogenomic_genes"].extend(genes)

        # Search FAERS
        faers_connector = self.connectors["faers"]
        if faers_connector.is_available():
            faers_data = faers_connector.search_drug(drug, max_results_per_source)
            results["results_by_source"]["faers"] = {
                "source": "faers",
                "layer": "pharmacological",
                "record_count": len(faers_data),
                "data": faers_data
            }
            results["sources_queried"].append("faers")

        # Search AACT
        aact_connector = self.connectors["aact"]
        if aact_connector.is_available():
            aact_data = aact_connector.search_condition(drug, max_results_per_source)
            results["results_by_source"]["aact"] = {
                "source": "aact",
                "layer": "pharmacological",
                "record_count": len(aact_data),
                "data": aact_data
            }
            results["sources_queried"].append("aact")

        # Summary
        results["summary"] = {
            "drug": drug,
            "pharmacogenomic_genes_count": len(set(results["pharmacogenomic_genes"])),
            "sources_queried": len(results["sources_queried"]),
            "total_records": sum(
                s.get("record_count", 0)
                for s in results["results_by_source"].values()
            )
        }
        results["execution_time_ms"] = (time.time() - start_time) * 1000

        return results

    def fusion_analysis(
        self,
        target_gene: Optional[str] = None,
        target_disease: Optional[str] = None,
        include_genomic: bool = True,
        include_proteomic: bool = True,
        include_clinical: bool = True,
        include_pharmacological: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive multi-omic fusion analysis.

        Integrates data across layers to provide:
        - Cross-layer correlations
        - Biomarker candidates
        - Drug candidates
        - Clinical implications
        """
        start_time = time.time()

        result = {
            "target_gene": target_gene,
            "target_disease": target_disease,
            "layers_analyzed": [],
            "genomic_summary": None,
            "proteomic_summary": None,
            "clinical_summary": None,
            "pharmacological_summary": None,
            "cross_layer_correlations": [],
            "integrated_score": 0.0,
            "confidence": 0.0,
            "biomarker_candidates": [],
            "drug_candidates": [],
            "clinical_implications": []
        }

        # Gene-centric analysis if gene provided
        if target_gene:
            gene_upper = target_gene.upper()

            if include_genomic:
                # ClinVar data
                clinvar_data = self.connectors["clinvar"].search_gene(gene_upper, max_variants=20)
                if clinvar_data:
                    result["genomic_summary"] = {
                        "gene": gene_upper,
                        "variant_count": len(clinvar_data),
                        "pathogenic_variants": [
                            v.get("variant_name", "")[:50]
                            for v in clinvar_data[:5]
                        ]
                    }
                    result["layers_analyzed"].append("genomic")

                # GEO expression
                geo_data = self.connectors["geo"].search_gene(gene_upper, max_files=3)
                if geo_data:
                    if "transcriptomic" not in result["layers_analyzed"]:
                        result["layers_analyzed"].append("transcriptomic")

            if include_proteomic:
                hpa_data = self.connectors["hpa"].search_gene(gene_upper)
                if hpa_data:
                    result["proteomic_summary"] = {
                        "gene": gene_upper,
                        "protein_data_count": len(hpa_data),
                        "sources": list(set(d.get("source_file", "") for d in hpa_data[:5]))
                    }
                    result["layers_analyzed"].append("proteomic")

            # Find associated drugs
            if include_pharmacological:
                for drug, genes in DRUG_GENE_MAP.items():
                    if gene_upper in genes:
                        result["drug_candidates"].append(drug)

                result["pharmacological_summary"] = {
                    "gene": gene_upper,
                    "associated_drugs": result["drug_candidates"][:10]
                }
                if result["drug_candidates"]:
                    result["layers_analyzed"].append("pharmacological")

            # Biomarker candidates
            if gene_upper in GENE_DISEASE_MAP:
                result["biomarker_candidates"] = [gene_upper]
                result["clinical_implications"] = [
                    f"{gene_upper} associated with: {', '.join(GENE_DISEASE_MAP[gene_upper])}"
                ]

        # Disease-centric analysis if disease provided
        if target_disease:
            disease_results = self.disease_centric_query(target_disease)

            if include_clinical:
                result["clinical_summary"] = {
                    "disease": target_disease,
                    "related_genes": disease_results.get("related_genes", []),
                    "clinical_trials": disease_results.get("results_by_source", {}).get("aact", {}).get("record_count", 0)
                }
                if result["clinical_summary"]["related_genes"]:
                    result["layers_analyzed"].append("clinical")

            result["biomarker_candidates"].extend(disease_results.get("related_genes", []))

        # Calculate integrated score
        layer_count = len(result["layers_analyzed"])
        if layer_count > 0:
            result["integrated_score"] = min(1.0, layer_count * 0.25)
            result["confidence"] = min(0.95, 0.3 + (layer_count * 0.15))

        result["execution_time_ms"] = (time.time() - start_time) * 1000

        return result


# Singleton instance
_fusion_engine = None


def get_fusion_engine() -> MultiOmicFusionEngine:
    """Get singleton fusion engine instance."""
    global _fusion_engine
    if _fusion_engine is None:
        _fusion_engine = MultiOmicFusionEngine()
    return _fusion_engine


# API-friendly functions
def get_source_status() -> List[Dict[str, Any]]:
    """Get status of all data sources."""
    return get_fusion_engine().get_all_source_status()


def unified_query(
    query_type: str,
    genes: Optional[List[str]] = None,
    disease: Optional[str] = None,
    drug: Optional[str] = None,
    sources: Optional[List[str]] = None,
    max_results_per_source: int = 50
) -> Dict[str, Any]:
    """
    Perform unified multi-omic query.

    Args:
        query_type: "gene_centric", "disease_centric", or "drug_centric"
        genes: List of gene symbols (for gene_centric)
        disease: Disease name (for disease_centric)
        drug: Drug name (for drug_centric)
        sources: Limit to specific sources
        max_results_per_source: Max results per source

    Returns:
        Unified query results
    """
    engine = get_fusion_engine()

    if query_type == "gene_centric" and genes:
        return engine.gene_centric_query(genes, sources, max_results_per_source)
    elif query_type == "disease_centric" and disease:
        return engine.disease_centric_query(disease, None, max_results_per_source)
    elif query_type == "drug_centric" and drug:
        return engine.drug_centric_query(drug, max_results_per_source)
    else:
        return {"error": "Invalid query type or missing parameters"}


def fusion_analysis(
    target_gene: Optional[str] = None,
    target_disease: Optional[str] = None,
    include_genomic: bool = True,
    include_proteomic: bool = True,
    include_clinical: bool = True,
    include_pharmacological: bool = True
) -> Dict[str, Any]:
    """Perform comprehensive multi-omic fusion analysis."""
    return get_fusion_engine().fusion_analysis(
        target_gene=target_gene,
        target_disease=target_disease,
        include_genomic=include_genomic,
        include_proteomic=include_proteomic,
        include_clinical=include_clinical,
        include_pharmacological=include_pharmacological
    )
