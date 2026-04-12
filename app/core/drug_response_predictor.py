"""
Drug Response Predictor for HyperCore

Provides:
1. FDA FAERS adverse event analysis
2. ClinicalTrials.gov AACT trial data
3. Pharmacogenomic drug response prediction
4. Drug-drug interaction checking
5. Integration with genomics pipeline
6. PharmGKB pharmacogenomics integration
7. ChEMBL drug-target relationships

Data sources (configured via environment variables):
- FAERS_PATH: FDA FAERS adverse events
- AACT_PATH: ClinicalTrials.gov AACT data
- PHARMGKB_PATH: PharmGKB pharmacogenomics
- CHEMBL_PATH: ChEMBL drug-target data
"""

import os
import re
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
import time

logger = logging.getLogger(__name__)

# Import PharmGKB integration
try:
    from app.core.pharmgkb_integration import (
        get_drug_gene_interactions as pharmgkb_drug_genes,
        get_variant_annotations as pharmgkb_variants,
        get_clinical_guidelines as pharmgkb_guidelines,
        get_haplotype_frequencies as pharmgkb_haplotypes,
        get_gene_drug_summary as pharmgkb_gene_summary,
        search_pharmgkb,
        check_pharmgkb_status,
    )
    PHARMGKB_AVAILABLE = True
except ImportError:
    PHARMGKB_AVAILABLE = False
    logger.warning("PharmGKB integration not available")

# Import ChEMBL integration
try:
    from app.core.chembl_integration import (
        get_compound_info as chembl_compound,
        get_drug_targets as chembl_targets,
        get_drug_mechanisms as chembl_mechanisms,
        search_compounds_by_name as chembl_search,
        get_bioactivities as chembl_activities,
        check_chembl_status,
        CHEMBL_AVAILABLE as _chembl_db_available,
    )
    CHEMBL_AVAILABLE = _chembl_db_available
except ImportError:
    CHEMBL_AVAILABLE = False
    logger.warning("ChEMBL integration not available")

# Data paths - use environment variables with fallbacks
_BASE_DIR = Path(__file__).parent.parent
FAERS_PATH = Path(os.environ.get('FAERS_PATH', _BASE_DIR / 'data' / 'faers'))
AACT_PATH = Path(os.environ.get('AACT_PATH', _BASE_DIR / 'data' / 'aact'))

# Pharmacogenomic associations (gene -> drugs affected)
PHARMACOGENOMIC_MAP = {
    "CYP2D6": {
        "drugs": ["codeine", "tramadol", "oxycodone", "tamoxifen", "metoprolol",
                  "paroxetine", "fluoxetine", "venlafaxine", "amitriptyline"],
        "effect": "metabolism",
        "poor_metabolizer_impact": "reduced_efficacy_or_toxicity"
    },
    "CYP2C19": {
        "drugs": ["clopidogrel", "omeprazole", "pantoprazole", "escitalopram",
                  "citalopram", "voriconazole", "diazepam"],
        "effect": "metabolism",
        "poor_metabolizer_impact": "reduced_efficacy"
    },
    "CYP2C9": {
        "drugs": ["warfarin", "phenytoin", "celecoxib", "losartan", "glipizide"],
        "effect": "metabolism",
        "poor_metabolizer_impact": "increased_toxicity"
    },
    "VKORC1": {
        "drugs": ["warfarin"],
        "effect": "target_sensitivity",
        "variant_impact": "increased_sensitivity"
    },
    "SLCO1B1": {
        "drugs": ["simvastatin", "atorvastatin", "rosuvastatin", "pravastatin"],
        "effect": "transport",
        "variant_impact": "increased_myopathy_risk"
    },
    "HLA-B*57:01": {
        "drugs": ["abacavir"],
        "effect": "immune",
        "variant_impact": "hypersensitivity_contraindication"
    },
    "HLA-B*15:02": {
        "drugs": ["carbamazepine", "phenytoin", "oxcarbazepine"],
        "effect": "immune",
        "variant_impact": "severe_skin_reaction_risk"
    },
    "TPMT": {
        "drugs": ["azathioprine", "mercaptopurine", "thioguanine"],
        "effect": "metabolism",
        "poor_metabolizer_impact": "severe_myelosuppression"
    },
    "UGT1A1": {
        "drugs": ["irinotecan"],
        "effect": "metabolism",
        "variant_impact": "increased_toxicity"
    },
    "DPYD": {
        "drugs": ["fluorouracil", "capecitabine"],
        "effect": "metabolism",
        "variant_impact": "severe_toxicity"
    },
    "G6PD": {
        "drugs": ["primaquine", "dapsone", "rasburicase", "methylene blue"],
        "effect": "metabolism",
        "variant_impact": "hemolytic_anemia"
    },
    "NUDT15": {
        "drugs": ["azathioprine", "mercaptopurine"],
        "effect": "metabolism",
        "variant_impact": "myelosuppression"
    }
}

# Drug-drug interactions (simplified knowledge base)
DRUG_INTERACTIONS = {
    ("warfarin", "aspirin"): {
        "severity": "major",
        "type": "pharmacodynamic",
        "effect": "Increased bleeding risk",
        "management": "Monitor INR closely, consider alternative"
    },
    ("warfarin", "amiodarone"): {
        "severity": "major",
        "type": "pharmacokinetic",
        "effect": "Increased warfarin levels via CYP2C9 inhibition",
        "management": "Reduce warfarin dose by 30-50%"
    },
    ("metformin", "contrast"): {
        "severity": "major",
        "type": "pharmacodynamic",
        "effect": "Risk of lactic acidosis",
        "management": "Hold metformin 48h before/after contrast"
    },
    ("ssri", "maoi"): {
        "severity": "contraindicated",
        "type": "pharmacodynamic",
        "effect": "Serotonin syndrome risk",
        "management": "Contraindicated - do not use together"
    },
    ("simvastatin", "amiodarone"): {
        "severity": "major",
        "type": "pharmacokinetic",
        "effect": "Increased statin levels, myopathy risk",
        "management": "Limit simvastatin to 20mg daily"
    },
    ("clopidogrel", "omeprazole"): {
        "severity": "moderate",
        "type": "pharmacokinetic",
        "effect": "Reduced clopidogrel activation via CYP2C19",
        "management": "Consider pantoprazole or H2 blocker"
    },
    ("methotrexate", "nsaid"): {
        "severity": "major",
        "type": "pharmacokinetic",
        "effect": "Reduced methotrexate clearance",
        "management": "Avoid NSAIDs or monitor closely"
    },
    ("lithium", "nsaid"): {
        "severity": "major",
        "type": "pharmacokinetic",
        "effect": "Increased lithium levels",
        "management": "Monitor lithium levels, consider alternative"
    },
    ("digoxin", "amiodarone"): {
        "severity": "major",
        "type": "pharmacokinetic",
        "effect": "Increased digoxin levels",
        "management": "Reduce digoxin dose by 50%"
    },
    ("fluconazole", "warfarin"): {
        "severity": "major",
        "type": "pharmacokinetic",
        "effect": "Increased warfarin effect via CYP2C9",
        "management": "Monitor INR, reduce warfarin dose"
    }
}

# Drug class mappings for interaction checking
DRUG_CLASSES = {
    "ssri": ["fluoxetine", "sertraline", "paroxetine", "citalopram", "escitalopram"],
    "maoi": ["phenelzine", "tranylcypromine", "selegiline", "isocarboxazid"],
    "nsaid": ["ibuprofen", "naproxen", "diclofenac", "celecoxib", "meloxicam", "indomethacin"],
    "statin": ["simvastatin", "atorvastatin", "rosuvastatin", "pravastatin", "lovastatin"],
    "ppi": ["omeprazole", "pantoprazole", "esomeprazole", "lansoprazole", "rabeprazole"],
    "anticoagulant": ["warfarin", "apixaban", "rivaroxaban", "dabigatran", "edoxaban"],
    "antiplatelet": ["clopidogrel", "aspirin", "ticagrelor", "prasugrel"]
}


class FAERSLoader:
    """Loader for FDA FAERS adverse event data."""

    def __init__(self, faers_path: Path = FAERS_PATH):
        self.faers_path = faers_path
        self._drug_cache: Dict[str, List[Dict]] = {}
        self._reaction_cache: Dict[str, List[Dict]] = {}
        self._loaded_quarters: set = set()

    def list_available_quarters(self) -> List[str]:
        """List available FAERS data quarters."""
        if not self.faers_path.exists():
            return []

        quarters = []
        for f in self.faers_path.glob("faers_ascii_*.zip"):
            match = re.search(r'(\d{4}q\d)', f.name)
            if match:
                quarters.append(match.group(1))
        return sorted(quarters)

    def _load_quarter(self, quarter: str) -> Tuple[List[Dict], List[Dict]]:
        """Load drug and reaction data from a quarter's ZIP file."""
        zip_path = self.faers_path / f"faers_ascii_{quarter}.zip"
        if not zip_path.exists():
            return [], []

        drugs = []
        reactions = []

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Find drug and reaction files
                drug_file = None
                reac_file = None

                for name in zf.namelist():
                    if 'DRUG' in name.upper() and name.endswith('.txt'):
                        drug_file = name
                    elif 'REAC' in name.upper() and name.endswith('.txt'):
                        reac_file = name

                # Load drugs
                if drug_file:
                    with zf.open(drug_file) as f:
                        lines = f.read().decode('utf-8', errors='replace').split('\n')
                        if lines:
                            header = lines[0].strip().split('$')
                            for line in lines[1:1001]:  # Limit to 1000 per quarter for memory
                                parts = line.strip().split('$')
                                if len(parts) >= 5:
                                    drugs.append({
                                        "primaryid": parts[0] if len(parts) > 0 else "",
                                        "caseid": parts[1] if len(parts) > 1 else "",
                                        "drug_seq": parts[2] if len(parts) > 2 else "",
                                        "role_cod": parts[3] if len(parts) > 3 else "",
                                        "drugname": parts[4] if len(parts) > 4 else "",
                                        "prod_ai": parts[5] if len(parts) > 5 else "",
                                        "route": parts[7] if len(parts) > 7 else "",
                                        "dose_vbm": parts[8] if len(parts) > 8 else ""
                                    })

                # Load reactions
                if reac_file:
                    with zf.open(reac_file) as f:
                        lines = f.read().decode('utf-8', errors='replace').split('\n')
                        for line in lines[1:1001]:  # Limit
                            parts = line.strip().split('$')
                            if len(parts) >= 3:
                                reactions.append({
                                    "primaryid": parts[0],
                                    "caseid": parts[1],
                                    "pt": parts[2]  # Preferred term (reaction)
                                })

        except Exception as e:
            logger.error(f"Error loading FAERS quarter {quarter}: {e}")

        return drugs, reactions

    def search_drug(self, drug_name: str, max_quarters: int = 4) -> Dict[str, Any]:
        """Search for adverse events related to a drug."""
        drug_upper = drug_name.upper()
        quarters = self.list_available_quarters()[-max_quarters:]  # Most recent quarters

        all_events = []
        reaction_counts = defaultdict(int)
        total_reports = 0

        for quarter in quarters:
            drugs, reactions = self._load_quarter(quarter)

            # Find cases with this drug
            drug_cases = set()
            for d in drugs:
                if drug_upper in d.get("drugname", "").upper():
                    drug_cases.add(d["primaryid"])
                    total_reports += 1

            # Get reactions for these cases
            for r in reactions:
                if r["primaryid"] in drug_cases:
                    reaction = r.get("pt", "Unknown")
                    reaction_counts[reaction] += 1
                    all_events.append({
                        "primaryid": r["primaryid"],
                        "reaction": reaction,
                        "quarter": quarter
                    })

        # Sort by frequency
        top_reactions = sorted(
            reaction_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]

        return {
            "drug_name": drug_name,
            "total_reports": total_reports,
            "quarters_searched": quarters,
            "top_adverse_events": [
                {"reaction": r, "count": c} for r, c in top_reactions
            ],
            "all_events": all_events[:100]  # Limit returned events
        }


class AACTLoader:
    """Loader for ClinicalTrials.gov AACT data."""

    def __init__(self, aact_path: Path = AACT_PATH):
        self.aact_path = aact_path
        self._interventions_cache: List[Dict] = []
        self._conditions_cache: List[Dict] = []
        self._studies_cache: List[Dict] = []
        self._loaded = False

    def _find_data_zip(self) -> Optional[Path]:
        """Find the AACT data ZIP with text files."""
        for f in self.aact_path.glob("*.zip"):
            try:
                with zipfile.ZipFile(f, 'r') as zf:
                    if any('interventions.txt' in n for n in zf.namelist()):
                        return f
            except:
                continue
        return None

    def _load_data(self, max_records: int = 10000):
        """Load AACT data from ZIP file."""
        if self._loaded:
            return

        zip_path = self._find_data_zip()
        if not zip_path:
            logger.warning("No AACT data ZIP found")
            self._loaded = True
            return

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Load interventions
                if 'interventions.txt' in zf.namelist():
                    with zf.open('interventions.txt') as f:
                        lines = f.read().decode('utf-8', errors='replace').split('\n')
                        header = lines[0].strip().split('|')
                        for line in lines[1:max_records]:
                            parts = line.strip().split('|')
                            if len(parts) >= 4:
                                self._interventions_cache.append({
                                    "id": parts[0],
                                    "nct_id": parts[1],
                                    "intervention_type": parts[2],
                                    "name": parts[3],
                                    "description": parts[4] if len(parts) > 4 else ""
                                })

                # Load conditions
                if 'conditions.txt' in zf.namelist():
                    with zf.open('conditions.txt') as f:
                        lines = f.read().decode('utf-8', errors='replace').split('\n')
                        for line in lines[1:max_records]:
                            parts = line.strip().split('|')
                            if len(parts) >= 3:
                                self._conditions_cache.append({
                                    "id": parts[0],
                                    "nct_id": parts[1],
                                    "name": parts[2],
                                    "downcase_name": parts[3] if len(parts) > 3 else ""
                                })

            logger.info(f"Loaded {len(self._interventions_cache)} interventions, "
                       f"{len(self._conditions_cache)} conditions from AACT")
            self._loaded = True

        except Exception as e:
            logger.error(f"Error loading AACT data: {e}")
            self._loaded = True

    def search_trials_by_condition(self, condition: str, limit: int = 50) -> List[Dict]:
        """Search clinical trials by condition."""
        self._load_data()
        condition_lower = condition.lower()

        matching_ncts = set()
        for c in self._conditions_cache:
            if condition_lower in c.get("name", "").lower():
                matching_ncts.add(c["nct_id"])

        # Get interventions for these trials
        trials = []
        for nct_id in list(matching_ncts)[:limit]:
            interventions = [
                i["name"] for i in self._interventions_cache
                if i["nct_id"] == nct_id
            ]
            trials.append({
                "nct_id": nct_id,
                "condition": condition,
                "interventions": interventions
            })

        return trials

    def search_trials_by_drug(self, drug: str, limit: int = 50) -> List[Dict]:
        """Search clinical trials by drug/intervention."""
        self._load_data()
        drug_upper = drug.upper()

        matching_interventions = []
        for i in self._interventions_cache:
            if drug_upper in i.get("name", "").upper():
                matching_interventions.append({
                    "nct_id": i["nct_id"],
                    "intervention_type": i["intervention_type"],
                    "name": i["name"],
                    "description": i.get("description", "")[:200]
                })

        return matching_interventions[:limit]

    def get_trial_count(self) -> int:
        """Get total number of trials loaded."""
        self._load_data()
        return len(set(i["nct_id"] for i in self._interventions_cache))


class DrugResponsePredictor:
    """
    Main drug response prediction engine.

    Integrates:
    - FAERS adverse event data
    - AACT clinical trial data
    - Pharmacogenomic knowledge base
    - Drug interaction database
    """

    def __init__(self):
        self.faers = FAERSLoader()
        self.aact = AACTLoader()

    def get_drug_profile(self, drug_name: str) -> Dict[str, Any]:
        """Get comprehensive drug profile."""
        drug_lower = drug_name.lower()

        # Get pharmacogenomic genes
        pgx_genes = []
        for gene, info in PHARMACOGENOMIC_MAP.items():
            if any(drug_lower in d.lower() for d in info["drugs"]):
                pgx_genes.append({
                    "gene": gene,
                    "effect": info["effect"],
                    "impact": info.get("poor_metabolizer_impact") or info.get("variant_impact")
                })

        # Get FAERS adverse events
        faers_data = self.faers.search_drug(drug_name, max_quarters=2)

        # Get clinical trials
        trials = self.aact.search_trials_by_drug(drug_name, limit=20)

        # Find drug interactions
        interactions = self._find_interactions(drug_lower)

        # Get PharmGKB data if available
        pharmgkb_data = None
        if PHARMGKB_AVAILABLE:
            try:
                pharmgkb_data = pharmgkb_drug_genes(drug_name)
            except Exception as e:
                logger.debug(f"PharmGKB lookup failed: {e}")

        result = {
            "drug_name": drug_name,
            "pharmacogenomic_genes": pgx_genes,
            "adverse_events": faers_data.get("top_adverse_events", [])[:10],
            "total_faers_reports": faers_data.get("total_reports", 0),
            "clinical_trials": len(trials),
            "trial_details": trials[:5],
            "known_interactions": interactions,
            "pharmgkb_available": PHARMGKB_AVAILABLE,
        }

        # Add PharmGKB data if available
        if pharmgkb_data:
            result["pharmgkb"] = {
                "pharmgkb_id": pharmgkb_data.get("pharmgkb_id"),
                "gene_interactions": pharmgkb_data.get("gene_interactions", [])[:10],
                "total_interactions": pharmgkb_data.get("total_interactions", 0),
                "clinical_annotation_count": pharmgkb_data.get("clinical_annotation_count", 0),
                "has_dosing_guideline": pharmgkb_data.get("dosing_guideline", False),
                "fda_label_testing": pharmgkb_data.get("fda_label_testing", ""),
            }

        # Get ChEMBL data if available
        result["chembl_available"] = CHEMBL_AVAILABLE
        if CHEMBL_AVAILABLE:
            try:
                # Get drug targets from ChEMBL
                chembl_data = chembl_targets(drug_name)
                if chembl_data and not chembl_data.get("error"):
                    result["chembl"] = {
                        "chembl_id": chembl_data.get("chembl_id"),
                        "pref_name": chembl_data.get("pref_name"),
                        "targets": chembl_data.get("targets", [])[:10],
                        "total_targets": chembl_data.get("total_targets", 0),
                    }
                    # Also get mechanisms
                    mech_data = chembl_mechanisms(drug_name)
                    if mech_data and not mech_data.get("error"):
                        result["chembl"]["mechanisms"] = mech_data.get("mechanisms", [])[:5]
                        result["chembl"]["total_mechanisms"] = mech_data.get("total_mechanisms", 0)
            except Exception as e:
                logger.debug(f"ChEMBL lookup failed: {e}")

        return result

    def _find_interactions(self, drug: str) -> List[Dict]:
        """Find known interactions for a drug."""
        interactions = []
        drug_lower = drug.lower()

        # Check drug class membership
        drug_class = None
        for cls, members in DRUG_CLASSES.items():
            if drug_lower in members:
                drug_class = cls
                break

        # Search interactions
        for (drug_a, drug_b), info in DRUG_INTERACTIONS.items():
            match_a = drug_lower in drug_a or (drug_class and drug_class in drug_a)
            match_b = drug_lower in drug_b or (drug_class and drug_class in drug_b)

            if match_a:
                interactions.append({
                    "interacting_drug": drug_b,
                    **info
                })
            elif match_b:
                interactions.append({
                    "interacting_drug": drug_a,
                    **info
                })

        return interactions

    def predict_response(
        self,
        drug_name: str,
        metabolizer_status: Optional[Dict[str, str]] = None,
        patient_genes: Optional[Dict[str, str]] = None,
        concurrent_medications: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Predict drug response based on pharmacogenomics.

        Args:
            drug_name: Drug to predict response for
            metabolizer_status: Gene -> metabolizer status (poor/intermediate/extensive/ultra_rapid)
            patient_genes: Gene -> variant information
            concurrent_medications: Other drugs patient is taking
        """
        drug_lower = drug_name.lower()
        prediction = {
            "drug_name": drug_name,
            "efficacy_prediction": "standard",
            "efficacy_confidence": 0.5,
            "toxicity_risk": "standard",
            "toxicity_confidence": 0.5,
            "dose_adjustment": None,
            "pharmacogenomic_factors": [],
            "interaction_warnings": [],
            "recommendations": [],
            "evidence_sources": []
        }

        # Check pharmacogenomic factors
        relevant_genes = []
        for gene, info in PHARMACOGENOMIC_MAP.items():
            if any(drug_lower in d.lower() for d in info["drugs"]):
                relevant_genes.append((gene, info))

        if metabolizer_status:
            for gene, info in relevant_genes:
                if gene in metabolizer_status:
                    status = metabolizer_status[gene].lower()
                    impact = info.get("poor_metabolizer_impact") or info.get("variant_impact", "")

                    factor = {
                        "gene": gene,
                        "status": status,
                        "effect": info["effect"]
                    }

                    if status == "poor":
                        if "toxicity" in impact:
                            prediction["toxicity_risk"] = "high"
                            prediction["toxicity_confidence"] = 0.8
                            prediction["dose_adjustment"] = "Consider 50% dose reduction"
                            prediction["recommendations"].append(
                                f"Patient is {gene} poor metabolizer - increased toxicity risk"
                            )
                        elif "efficacy" in impact:
                            prediction["efficacy_prediction"] = "reduced"
                            prediction["efficacy_confidence"] = 0.75
                            prediction["recommendations"].append(
                                f"Patient is {gene} poor metabolizer - consider alternative drug"
                            )
                        factor["recommendation"] = prediction["recommendations"][-1]

                    elif status == "ultra_rapid":
                        prediction["efficacy_prediction"] = "reduced"
                        prediction["efficacy_confidence"] = 0.7
                        prediction["dose_adjustment"] = "May need higher dose"
                        prediction["recommendations"].append(
                            f"Patient is {gene} ultra-rapid metabolizer - may need dose increase"
                        )
                        factor["recommendation"] = prediction["recommendations"][-1]

                    prediction["pharmacogenomic_factors"].append(factor)
                    prediction["evidence_sources"].append(f"Pharmacogenomics: {gene}")

        # Check drug interactions
        if concurrent_medications:
            for other_drug in concurrent_medications:
                interactions = self._check_pair_interaction(drug_lower, other_drug.lower())
                if interactions:
                    for interaction in interactions:
                        warning = f"{drug_name} + {other_drug}: {interaction['effect']}"
                        prediction["interaction_warnings"].append(warning)

                        if interaction["severity"] in ["major", "contraindicated"]:
                            prediction["toxicity_risk"] = "high"
                            prediction["toxicity_confidence"] = max(
                                prediction["toxicity_confidence"], 0.85
                            )

                        prediction["recommendations"].append(interaction["management"])
                        prediction["evidence_sources"].append(
                            f"Drug interaction: {other_drug}"
                        )

        # Set confidence based on evidence
        if prediction["pharmacogenomic_factors"] or prediction["interaction_warnings"]:
            prediction["efficacy_confidence"] = min(0.9, prediction["efficacy_confidence"] + 0.2)
            prediction["toxicity_confidence"] = min(0.9, prediction["toxicity_confidence"] + 0.2)

        return prediction

    def _check_pair_interaction(self, drug_a: str, drug_b: str) -> List[Dict]:
        """Check interaction between two drugs."""
        interactions = []

        # Normalize to class if needed
        class_a = None
        class_b = None
        for cls, members in DRUG_CLASSES.items():
            if drug_a in members:
                class_a = cls
            if drug_b in members:
                class_b = cls

        # Check direct and class-based interactions
        for (d1, d2), info in DRUG_INTERACTIONS.items():
            match = False
            if (drug_a in d1 or (class_a and class_a in d1)) and \
               (drug_b in d2 or (class_b and class_b in d2)):
                match = True
            elif (drug_b in d1 or (class_b and class_b in d1)) and \
                 (drug_a in d2 or (class_a and class_a in d2)):
                match = True

            if match:
                interactions.append(info)

        return interactions

    def check_interactions(self, drugs: List[str]) -> Dict[str, Any]:
        """Check all pairwise interactions in a drug list."""
        result = {
            "drugs_checked": drugs,
            "interactions": [],
            "max_severity": "none",
            "recommendations": []
        }

        severity_order = {"none": 0, "minor": 1, "moderate": 2, "major": 3, "contraindicated": 4}

        # Check all pairs
        for i, drug_a in enumerate(drugs):
            for drug_b in drugs[i+1:]:
                interactions = self._check_pair_interaction(drug_a.lower(), drug_b.lower())
                for interaction in interactions:
                    result["interactions"].append({
                        "drug_a": drug_a,
                        "drug_b": drug_b,
                        **interaction
                    })

                    if severity_order.get(interaction["severity"], 0) > \
                       severity_order.get(result["max_severity"], 0):
                        result["max_severity"] = interaction["severity"]

                    result["recommendations"].append(
                        f"{drug_a}/{drug_b}: {interaction['management']}"
                    )

        result["interaction_count"] = len(result["interactions"])
        return result

    def get_adverse_events(self, drug_name: str, limit: int = 100) -> Dict[str, Any]:
        """Get adverse events for a drug from FAERS."""
        return self.faers.search_drug(drug_name)

    def search_clinical_trials(
        self,
        condition: Optional[str] = None,
        drug: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Search clinical trials."""
        if condition:
            trials = self.aact.search_trials_by_condition(condition, limit)
            return {
                "query_type": "condition",
                "query": condition,
                "trials": trials,
                "count": len(trials)
            }
        elif drug:
            trials = self.aact.search_trials_by_drug(drug, limit)
            return {
                "query_type": "drug",
                "query": drug,
                "trials": trials,
                "count": len(trials)
            }
        else:
            return {"error": "Either condition or drug must be specified"}


# Singleton instance
_predictor = None


def get_predictor() -> DrugResponsePredictor:
    """Get singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = DrugResponsePredictor()
    return _predictor


# API-friendly functions
def get_drug_profile(drug_name: str) -> Dict[str, Any]:
    """Get comprehensive drug profile."""
    return get_predictor().get_drug_profile(drug_name)


def predict_drug_response(
    drug_name: str,
    metabolizer_status: Optional[Dict[str, str]] = None,
    patient_genes: Optional[Dict[str, str]] = None,
    concurrent_medications: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Predict drug response."""
    return get_predictor().predict_response(
        drug_name, metabolizer_status, patient_genes, concurrent_medications
    )


def check_drug_interactions(drugs: List[str]) -> Dict[str, Any]:
    """Check drug interactions."""
    return get_predictor().check_interactions(drugs)


def get_adverse_events(drug_name: str) -> Dict[str, Any]:
    """Get adverse events for a drug."""
    return get_predictor().get_adverse_events(drug_name)


def search_trials(
    condition: Optional[str] = None,
    drug: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """Search clinical trials."""
    return get_predictor().search_clinical_trials(condition, drug, limit)


def get_data_status() -> Dict[str, Any]:
    """Get status of pharmaceutical data sources."""
    predictor = get_predictor()

    faers_quarters = predictor.faers.list_available_quarters()
    aact_trials = predictor.aact.get_trial_count()

    # Get PharmGKB status
    pharmgkb_status = None
    if PHARMGKB_AVAILABLE:
        try:
            pharmgkb_status = check_pharmgkb_status()
        except Exception as e:
            logger.debug(f"PharmGKB status check failed: {e}")

    # Get ChEMBL status
    chembl_status = None
    if CHEMBL_AVAILABLE:
        try:
            chembl_status = check_chembl_status()
        except Exception as e:
            logger.debug(f"ChEMBL status check failed: {e}")

    return {
        "faers_available": FAERS_PATH.exists(),
        "faers_quarters": len(faers_quarters),
        "faers_latest_quarter": faers_quarters[-1] if faers_quarters else None,
        "aact_available": AACT_PATH.exists(),
        "aact_trials_indexed": aact_trials,
        "pharmacogenomic_genes": len(PHARMACOGENOMIC_MAP),
        "interaction_pairs": len(DRUG_INTERACTIONS),
        "pharmgkb_available": PHARMGKB_AVAILABLE,
        "pharmgkb_status": pharmgkb_status,
        "chembl_available": CHEMBL_AVAILABLE,
        "chembl_status": chembl_status,
    }
