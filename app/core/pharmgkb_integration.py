"""
PharmGKB Integration Module

Provides pharmacogenomics data integration for HyperCore:
- Gene-drug relationships
- Clinical variant annotations
- Drug labels and guidelines
- Haplotype frequencies by population

Data source: PharmGKB (https://www.pharmgkb.org/)
"""

import os
import csv
import json
import logging
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path
from functools import lru_cache

# Increase CSV field size limit for large PharmGKB fields
csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)

# PharmGKB data path - use environment variable with fallback
# Go up 3 levels: pharmgkb_integration.py -> core -> app -> project_root
_BASE_DIR = Path(__file__).parent.parent.parent
PHARMGKB_PATH = Path(os.environ.get('PHARMGKB_PATH', _BASE_DIR / 'data' / 'pharmgkb'))

# Data caches
_relationships_cache: Dict[str, List[Dict]] = {}
_drugs_cache: Dict[str, Dict] = {}
_genes_cache: Dict[str, Dict] = {}
_clinical_variants_cache: Dict[str, List[Dict]] = {}
_drug_labels_cache: Dict[str, List[Dict]] = {}
_haplotype_cache: Dict[str, Dict] = {}


def _load_tsv(file_path: Path) -> List[Dict]:
    """Load TSV file into list of dicts."""
    if not file_path.exists():
        logger.warning(f"PharmGKB file not found: {file_path}")
        return []

    rows = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                rows.append(row)
    except Exception as e:
        logger.error(f"Error loading PharmGKB TSV {file_path}: {e}")

    return rows


def _ensure_relationships_loaded():
    """Load relationships.tsv into cache."""
    global _relationships_cache

    if _relationships_cache:
        return

    rel_path = PHARMGKB_PATH / "relationships" / "relationships.tsv"
    rows = _load_tsv(rel_path)

    # Index by both gene and chemical/drug
    for row in rows:
        entity1_name = row.get("Entity1_name", "").lower()
        entity2_name = row.get("Entity2_name", "").lower()
        entity1_type = row.get("Entity1_type", "")
        entity2_type = row.get("Entity2_type", "")

        # Index genes
        if entity1_type == "Gene":
            if entity1_name not in _relationships_cache:
                _relationships_cache[entity1_name] = []
            _relationships_cache[entity1_name].append(row)

        # Index drugs/chemicals
        if entity2_type == "Chemical":
            if entity2_name not in _relationships_cache:
                _relationships_cache[entity2_name] = []
            _relationships_cache[entity2_name].append(row)

    logger.info(f"Loaded {len(rows)} PharmGKB relationships")


def _ensure_drugs_loaded():
    """Load drugs.tsv into cache."""
    global _drugs_cache

    if _drugs_cache:
        return

    drugs_path = PHARMGKB_PATH / "drugs" / "drugs.tsv"
    rows = _load_tsv(drugs_path)

    for row in rows:
        name = row.get("Name", "").lower()
        if name:
            _drugs_cache[name] = row

        # Also index by generic names
        generic = row.get("Generic Names", "")
        if generic:
            for gname in generic.split(","):
                gname = gname.strip().lower()
                if gname and gname not in _drugs_cache:
                    _drugs_cache[gname] = row

    logger.info(f"Loaded {len(rows)} PharmGKB drugs")


def _ensure_genes_loaded():
    """Load genes.tsv into cache."""
    global _genes_cache

    if _genes_cache:
        return

    genes_path = PHARMGKB_PATH / "genes" / "genes.tsv"
    rows = _load_tsv(genes_path)

    for row in rows:
        symbol = row.get("Symbol", "").upper()
        if symbol:
            _genes_cache[symbol] = row

        # Also index by alternate symbols
        alt = row.get("Alternate Symbols", "")
        if alt:
            for asym in alt.split(","):
                asym = asym.strip().upper()
                if asym and asym not in _genes_cache:
                    _genes_cache[asym] = row

    logger.info(f"Loaded {len(rows)} PharmGKB genes")


def _ensure_clinical_variants_loaded():
    """Load clinicalVariants.tsv into cache."""
    global _clinical_variants_cache

    if _clinical_variants_cache:
        return

    cv_path = PHARMGKB_PATH / "clinicalVariants" / "clinicalVariants.tsv"
    rows = _load_tsv(cv_path)

    for row in rows:
        gene = row.get("gene", "").upper()
        if gene:
            if gene not in _clinical_variants_cache:
                _clinical_variants_cache[gene] = []
            _clinical_variants_cache[gene].append(row)

    logger.info(f"Loaded {len(rows)} PharmGKB clinical variants")


def _ensure_drug_labels_loaded():
    """Load drugLabels.tsv into cache."""
    global _drug_labels_cache

    if _drug_labels_cache:
        return

    labels_path = PHARMGKB_PATH / "drugLabels" / "drugLabels.tsv"
    rows = _load_tsv(labels_path)

    for row in rows:
        # Index by chemicals mentioned
        chemicals = row.get("Chemicals", "")
        for chem in chemicals.split(";"):
            chem = chem.strip().lower()
            if chem:
                if chem not in _drug_labels_cache:
                    _drug_labels_cache[chem] = []
                _drug_labels_cache[chem].append(row)

        # Also index by genes mentioned
        genes = row.get("Genes", "")
        for gene in genes.split(";"):
            gene = gene.strip().upper()
            if gene:
                key = f"gene:{gene}"
                if key not in _drug_labels_cache:
                    _drug_labels_cache[key] = []
                _drug_labels_cache[key].append(row)

    logger.info(f"Loaded {len(rows)} PharmGKB drug labels")


def get_drug_gene_interactions(drug_name: str) -> Dict[str, Any]:
    """
    Get gene-drug interactions for a drug.

    Args:
        drug_name: Name of the drug to lookup

    Returns:
        Dict with drug info and gene interactions
    """
    _ensure_relationships_loaded()
    _ensure_drugs_loaded()

    drug_lower = drug_name.lower()

    # Get drug info
    drug_info = _drugs_cache.get(drug_lower, {})

    # Get relationships where this drug is involved
    relationships = _relationships_cache.get(drug_lower, [])

    # Extract gene interactions
    gene_interactions = []
    for rel in relationships:
        if rel.get("Entity1_type") == "Gene":
            gene_interactions.append({
                "gene_id": rel.get("Entity1_id"),
                "gene_name": rel.get("Entity1_name"),
                "evidence": rel.get("Evidence"),
                "association": rel.get("Association"),
                "pk": rel.get("PK") == "PK",
                "pd": rel.get("PD") == "PD",
                "pmids": rel.get("PMIDs", "").split(",") if rel.get("PMIDs") else [],
            })

    return {
        "drug_name": drug_name,
        "pharmgkb_id": drug_info.get("PharmGKB Accession Id", ""),
        "generic_names": drug_info.get("Generic Names", "").split(",") if drug_info.get("Generic Names") else [],
        "trade_names": drug_info.get("Trade Names", "").split(",") if drug_info.get("Trade Names") else [],
        "drug_type": drug_info.get("Type", ""),
        "smiles": drug_info.get("SMILES", ""),
        "clinical_annotation_count": int(drug_info.get("Clinical Annotation Count", 0) or 0),
        "variant_annotation_count": int(drug_info.get("Variant Annotation Count", 0) or 0),
        "dosing_guideline": drug_info.get("Dosing Guideline", "") == "Yes",
        "top_clinical_level": drug_info.get("Top Clinical Annotation Level", ""),
        "fda_label_testing": drug_info.get("Top FDA Label Testing Level", ""),
        "gene_interactions": gene_interactions,
        "total_interactions": len(gene_interactions),
    }


def get_variant_annotations(gene_symbol: str) -> Dict[str, Any]:
    """
    Get clinical variant annotations for a gene.

    Args:
        gene_symbol: Gene symbol (e.g., CYP2D6, BRCA1)

    Returns:
        Dict with gene info and variant annotations
    """
    _ensure_clinical_variants_loaded()
    _ensure_genes_loaded()

    gene_upper = gene_symbol.upper()

    # Get gene info
    gene_info = _genes_cache.get(gene_upper, {})

    # Get clinical variants
    variants = _clinical_variants_cache.get(gene_upper, [])

    # Format variant annotations
    annotations = []
    for var in variants:
        annotations.append({
            "variant": var.get("variant", ""),
            "type": var.get("type", ""),
            "level_of_evidence": var.get("level of evidence", ""),
            "chemicals": var.get("chemicals", "").split(",") if var.get("chemicals") else [],
            "phenotypes": var.get("phenotypes", "").split(",") if var.get("phenotypes") else [],
        })

    # Sort by evidence level
    evidence_order = {"1A": 0, "1B": 1, "2A": 2, "2B": 3, "3": 4, "4": 5}
    annotations.sort(key=lambda x: evidence_order.get(x.get("level_of_evidence", ""), 99))

    return {
        "gene_symbol": gene_symbol,
        "pharmgkb_id": gene_info.get("PharmGKB Accession Id", ""),
        "gene_name": gene_info.get("Name", ""),
        "ncbi_gene_id": gene_info.get("NCBI Gene ID", ""),
        "hgnc_id": gene_info.get("HGNC ID", ""),
        "ensembl_id": gene_info.get("Ensembl Id", ""),
        "chromosome": gene_info.get("Chromosome", ""),
        "is_vip": gene_info.get("Is VIP", "") == "Yes",
        "has_cpic_guideline": gene_info.get("Has CPIC Dosing Guideline", "") == "Yes",
        "variant_annotations": annotations,
        "total_annotations": len(annotations),
    }


def get_clinical_guidelines(drug_name: str) -> Dict[str, Any]:
    """
    Get clinical guidelines for a drug.

    Args:
        drug_name: Name of the drug

    Returns:
        Dict with drug labels and guidelines
    """
    _ensure_drug_labels_loaded()

    drug_lower = drug_name.lower()

    # Get drug labels
    labels = _drug_labels_cache.get(drug_lower, [])

    # Load guideline annotations if available
    guidelines = []
    guidelines_dir = PHARMGKB_PATH / "guidelineAnnotations.json"

    # Format labels
    formatted_labels = []
    for label in labels:
        formatted_labels.append({
            "pharmgkb_id": label.get("PharmGKB ID", ""),
            "name": label.get("Name", ""),
            "source": label.get("Source", ""),
            "testing_level": label.get("Testing Level", ""),
            "has_prescribing_info": label.get("Has Prescribing Info", "") == "Prescribing Info",
            "has_dosing_info": label.get("Has Dosing Info", "") == "Dosing Info",
            "has_alternate_drug": label.get("Has Alternate Drug", "") == "Alternate Drug",
            "genes": label.get("Genes", "").split(";") if label.get("Genes") else [],
            "chemicals": label.get("Chemicals", "").split(";") if label.get("Chemicals") else [],
            "latest_update": label.get("Latest History Date (YYYY-MM-DD)", ""),
        })

    # Try to load JSON guidelines
    if guidelines_dir.exists():
        try:
            for json_file in guidelines_dir.glob("PA*.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    guideline = data.get("guideline", {})

                    # Check if this guideline is for our drug
                    related_chemicals = guideline.get("relatedChemicals", [])
                    for chem in related_chemicals:
                        if chem.get("name", "").lower() == drug_lower:
                            guidelines.append({
                                "id": guideline.get("id", ""),
                                "name": guideline.get("name", ""),
                                "source": guideline.get("source", ""),
                                "summary": guideline.get("summaryMarkdown", {}).get("html", ""),
                                "related_genes": [g.get("symbol") for g in guideline.get("relatedGenes", [])],
                                "has_dosing_info": guideline.get("dosingInformation", False),
                                "alternate_drug_available": guideline.get("alternateDrugAvailable", False),
                            })
                            break
        except Exception as e:
            logger.debug(f"Error loading guideline annotations: {e}")

    return {
        "drug_name": drug_name,
        "drug_labels": formatted_labels,
        "total_labels": len(formatted_labels),
        "guidelines": guidelines[:10],  # Limit to first 10
        "total_guidelines": len(guidelines),
    }


def get_haplotype_frequencies(
    gene_symbol: str,
    population: str = "all"
) -> Dict[str, Any]:
    """
    Get haplotype frequencies for a gene.

    Args:
        gene_symbol: Gene symbol (e.g., CYP2D6, CYP2C19)
        population: Population/biogeographic group or "all"

    Returns:
        Dict with haplotype frequency data
    """
    gene_upper = gene_symbol.upper()

    # Try AllOfUs data first
    allofus_path = PHARMGKB_PATH / "pharmgkb_haplotype_frequencies_AllOfUs" / "AllOfUs_Frequencies_v7" / "allele"
    ukbb_path = PHARMGKB_PATH / "pharmgkb_haplotype_frequencies_UKBB" / "ukbb_allele_frequencies"

    frequencies = []
    sources = []

    # Load AllOfUs data
    allofus_file = allofus_path / f"{gene_upper}_allele.tsv"
    if allofus_file.exists():
        rows = _load_tsv(allofus_file)
        for row in rows:
            pop = row.get("biogeographic_group", "")
            if population.lower() == "all" or pop.lower() == population.lower():
                freq_val = row.get("frequencies", "0")
                try:
                    freq = float(freq_val) if freq_val else 0.0
                except ValueError:
                    freq = 0.0

                frequencies.append({
                    "source": "AllOfUs",
                    "population": pop,
                    "allele": row.get("allele", ""),
                    "frequency": freq,
                    "n_haplotypes": int(row.get("n_haplotype", 0) or 0),
                    "n_subjects": int(float(row.get("n_subjects_genotyped", 0) or 0)),
                    "in_cpic": row.get("in_cpic", "") == "True",
                })
        sources.append("AllOfUs")

    # Load UKBB data
    ukbb_file = ukbb_path / f"{gene_upper}_allele.tsv" if ukbb_path.exists() else None
    if ukbb_file and ukbb_file.exists():
        rows = _load_tsv(ukbb_file)
        for row in rows:
            pop = row.get("biogeographic_group", row.get("population", ""))
            if population.lower() == "all" or pop.lower() == population.lower():
                freq_val = row.get("frequencies", row.get("frequency", "0"))
                try:
                    freq = float(freq_val) if freq_val else 0.0
                except ValueError:
                    freq = 0.0

                frequencies.append({
                    "source": "UKBB",
                    "population": pop,
                    "allele": row.get("allele", ""),
                    "frequency": freq,
                    "n_haplotypes": int(row.get("n_haplotype", row.get("count", 0)) or 0),
                    "n_subjects": int(float(row.get("n_subjects_genotyped", row.get("total", 0)) or 0)),
                })
        sources.append("UKBB")

    # Get unique populations
    populations = list(set(f.get("population") for f in frequencies if f.get("population")))

    # Get unique alleles with aggregated data
    allele_summary = {}
    for f in frequencies:
        allele = f.get("allele", "")
        if allele not in allele_summary:
            allele_summary[allele] = {
                "allele": allele,
                "avg_frequency": 0.0,
                "frequencies_by_pop": {},
                "count": 0,
            }
        allele_summary[allele]["frequencies_by_pop"][f.get("population", "")] = f.get("frequency", 0)
        allele_summary[allele]["count"] += 1

    # Calculate average frequencies
    for allele, data in allele_summary.items():
        freqs = list(data["frequencies_by_pop"].values())
        data["avg_frequency"] = sum(freqs) / len(freqs) if freqs else 0.0

    return {
        "gene_symbol": gene_symbol,
        "sources": sources,
        "populations": sorted(populations),
        "filter_population": population,
        "frequencies": frequencies,
        "total_records": len(frequencies),
        "allele_summary": list(allele_summary.values()),
        "unique_alleles": len(allele_summary),
    }


def get_gene_drug_summary(gene_symbol: str) -> Dict[str, Any]:
    """
    Get comprehensive gene-drug summary including variants and labels.

    Args:
        gene_symbol: Gene symbol

    Returns:
        Dict with complete gene-drug information
    """
    _ensure_relationships_loaded()
    _ensure_drug_labels_loaded()

    gene_upper = gene_symbol.upper()
    gene_lower = gene_symbol.lower()

    # Get relationships where this gene is Entity1
    relationships = _relationships_cache.get(gene_lower, [])

    # Extract drug interactions
    drug_interactions = []
    for rel in relationships:
        if rel.get("Entity2_type") == "Chemical":
            drug_interactions.append({
                "drug_id": rel.get("Entity2_id"),
                "drug_name": rel.get("Entity2_name"),
                "evidence": rel.get("Evidence"),
                "association": rel.get("Association"),
                "pk": rel.get("PK") == "PK",
                "pd": rel.get("PD") == "PD",
                "pmids": rel.get("PMIDs", "").split(",") if rel.get("PMIDs") else [],
            })

    # Get drug labels for this gene
    labels_key = f"gene:{gene_upper}"
    labels = _drug_labels_cache.get(labels_key, [])

    # Get variant annotations
    variant_data = get_variant_annotations(gene_symbol)

    # Get haplotype frequencies
    haplotype_data = get_haplotype_frequencies(gene_symbol, "all")

    return {
        "gene_symbol": gene_symbol,
        "gene_info": {
            "pharmgkb_id": variant_data.get("pharmgkb_id"),
            "gene_name": variant_data.get("gene_name"),
            "ncbi_gene_id": variant_data.get("ncbi_gene_id"),
            "chromosome": variant_data.get("chromosome"),
            "is_vip": variant_data.get("is_vip"),
            "has_cpic_guideline": variant_data.get("has_cpic_guideline"),
        },
        "drug_interactions": drug_interactions,
        "total_drug_interactions": len(drug_interactions),
        "drug_labels": len(labels),
        "variant_annotations": variant_data.get("variant_annotations", [])[:10],
        "total_variants": variant_data.get("total_annotations", 0),
        "haplotype_sources": haplotype_data.get("sources", []),
        "unique_alleles": haplotype_data.get("unique_alleles", 0),
    }


def search_pharmgkb(
    query: str,
    search_type: str = "all"
) -> Dict[str, Any]:
    """
    Search PharmGKB data.

    Args:
        query: Search query
        search_type: "drug", "gene", "variant", or "all"

    Returns:
        Dict with search results
    """
    _ensure_drugs_loaded()
    _ensure_genes_loaded()
    _ensure_clinical_variants_loaded()

    query_lower = query.lower()
    query_upper = query.upper()

    results = {
        "query": query,
        "search_type": search_type,
        "drugs": [],
        "genes": [],
        "variants": [],
    }

    # Search drugs
    if search_type in ["drug", "all"]:
        for name, drug in _drugs_cache.items():
            if query_lower in name or query_lower in drug.get("Generic Names", "").lower():
                results["drugs"].append({
                    "name": drug.get("Name"),
                    "pharmgkb_id": drug.get("PharmGKB Accession Id"),
                    "type": drug.get("Type"),
                    "clinical_annotations": drug.get("Clinical Annotation Count"),
                })
                if len(results["drugs"]) >= 20:
                    break

    # Search genes
    if search_type in ["gene", "all"]:
        for symbol, gene in _genes_cache.items():
            if query_upper in symbol or query_lower in gene.get("Name", "").lower():
                results["genes"].append({
                    "symbol": gene.get("Symbol"),
                    "name": gene.get("Name"),
                    "pharmgkb_id": gene.get("PharmGKB Accession Id"),
                    "is_vip": gene.get("Is VIP") == "Yes",
                })
                if len(results["genes"]) >= 20:
                    break

    # Search variants
    if search_type in ["variant", "all"]:
        for gene, variants in _clinical_variants_cache.items():
            if query_upper in gene:
                for var in variants[:5]:
                    results["variants"].append({
                        "gene": gene,
                        "variant": var.get("variant"),
                        "type": var.get("type"),
                        "evidence_level": var.get("level of evidence"),
                    })

    results["total_drugs"] = len(results["drugs"])
    results["total_genes"] = len(results["genes"])
    results["total_variants"] = len(results["variants"])

    return results


# Convenience function for checking data availability
def check_pharmgkb_status() -> Dict[str, Any]:
    """Check PharmGKB data availability."""
    status = {
        "path": str(PHARMGKB_PATH),
        "available": PHARMGKB_PATH.exists(),
        "datasets": {},
    }

    if not status["available"]:
        return status

    # Check each dataset
    datasets = [
        ("relationships", "relationships/relationships.tsv"),
        ("drugs", "drugs/drugs.tsv"),
        ("genes", "genes/genes.tsv"),
        ("clinical_variants", "clinicalVariants/clinicalVariants.tsv"),
        ("drug_labels", "drugLabels/drugLabels.tsv"),
        ("guidelines", "guidelineAnnotations.json"),
        ("haplotypes_allofus", "pharmgkb_haplotype_frequencies_AllOfUs"),
        ("haplotypes_ukbb", "pharmgkb_haplotype_frequencies_UKBB"),
    ]

    for name, subpath in datasets:
        full_path = PHARMGKB_PATH / subpath
        status["datasets"][name] = {
            "exists": full_path.exists(),
            "path": str(full_path),
        }
        if full_path.exists() and full_path.is_file():
            status["datasets"][name]["size_bytes"] = full_path.stat().st_size

    return status
