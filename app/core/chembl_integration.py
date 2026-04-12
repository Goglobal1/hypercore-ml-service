"""
ChEMBL Integration Module

Provides drug-target relationship data from ChEMBL database:
- Compound information and structures
- Drug-target relationships
- Mechanism of action data
- Bioactivity data

Data source: ChEMBL (https://www.ebi.ac.uk/chembl/)
Database: ChEMBL 36 SQLite
"""

import sqlite3
import logging
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ChEMBL database path - use environment variable with fallback
_BASE_DIR = Path(__file__).parent.parent
CHEMBL_PATH = Path(os.environ.get('CHEMBL_PATH', _BASE_DIR / 'data' / 'chembl'))
CHEMBL_DB_PATH = CHEMBL_PATH / "chembl_36" / "chembl_36_sqlite" / "chembl_36.db"

# Check if database exists
CHEMBL_AVAILABLE = CHEMBL_DB_PATH.exists()


@contextmanager
def get_connection():
    """Get a database connection with context management."""
    if not CHEMBL_AVAILABLE:
        raise RuntimeError("ChEMBL database not available")

    conn = sqlite3.connect(str(CHEMBL_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def get_compound_info(chembl_id: str) -> Dict[str, Any]:
    """
    Get compound/molecule information by ChEMBL ID.

    Args:
        chembl_id: ChEMBL compound ID (e.g., CHEMBL25, CHEMBL192)

    Returns:
        Dict with compound information including structure, properties, synonyms
    """
    if not CHEMBL_AVAILABLE:
        return {"error": "ChEMBL database not available", "chembl_id": chembl_id}

    chembl_id = chembl_id.upper()
    if not chembl_id.startswith("CHEMBL"):
        chembl_id = f"CHEMBL{chembl_id}"

    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            # Get molecule dictionary info
            cursor.execute("""
                SELECT
                    md.molregno,
                    md.chembl_id,
                    md.pref_name,
                    md.max_phase,
                    md.therapeutic_flag,
                    md.molecule_type,
                    md.first_approval,
                    md.oral,
                    md.parenteral,
                    md.topical,
                    md.black_box_warning,
                    md.availability_type,
                    cp.mw_freebase,
                    cp.alogp,
                    cp.hba,
                    cp.hbd,
                    cp.psa,
                    cp.rtb,
                    cp.ro3_pass,
                    cp.num_ro5_violations,
                    cp.full_mwt,
                    cs.canonical_smiles,
                    cs.standard_inchi,
                    cs.standard_inchi_key
                FROM molecule_dictionary md
                LEFT JOIN compound_properties cp ON md.molregno = cp.molregno
                LEFT JOIN compound_structures cs ON md.molregno = cs.molregno
                WHERE md.chembl_id = ?
            """, (chembl_id,))

            row = cursor.fetchone()

            if not row:
                return {"error": "Compound not found", "chembl_id": chembl_id}

            result = {
                "chembl_id": row["chembl_id"],
                "molregno": row["molregno"],
                "pref_name": row["pref_name"],
                "max_phase": row["max_phase"],
                "therapeutic_flag": bool(row["therapeutic_flag"]),
                "molecule_type": row["molecule_type"],
                "first_approval": row["first_approval"],
                "administration": {
                    "oral": bool(row["oral"]),
                    "parenteral": bool(row["parenteral"]),
                    "topical": bool(row["topical"]),
                },
                "black_box_warning": bool(row["black_box_warning"]),
                "availability_type": row["availability_type"],
                "properties": {
                    "molecular_weight": row["mw_freebase"],
                    "full_mwt": row["full_mwt"],
                    "alogp": row["alogp"],
                    "hba": row["hba"],
                    "hbd": row["hbd"],
                    "psa": row["psa"],
                    "rotatable_bonds": row["rtb"],
                    "ro3_pass": row["ro3_pass"],
                    "ro5_violations": row["num_ro5_violations"],
                },
                "structure": {
                    "canonical_smiles": row["canonical_smiles"],
                    "standard_inchi": row["standard_inchi"],
                    "standard_inchi_key": row["standard_inchi_key"],
                },
            }

            # Get synonyms
            cursor.execute("""
                SELECT syn_type, synonyms
                FROM molecule_synonyms
                WHERE molregno = ?
                ORDER BY syn_type
                LIMIT 20
            """, (row["molregno"],))

            synonyms = [{"type": r["syn_type"], "name": r["synonyms"]} for r in cursor.fetchall()]
            result["synonyms"] = synonyms

            return result

    except Exception as e:
        logger.error(f"Error getting compound info for {chembl_id}: {e}")
        return {"error": str(e), "chembl_id": chembl_id}


def get_drug_targets(drug_name: str, limit: int = 50) -> Dict[str, Any]:
    """
    Get targets for a drug by name.

    Args:
        drug_name: Drug name to search
        limit: Maximum number of targets to return

    Returns:
        Dict with drug info and list of targets
    """
    if not CHEMBL_AVAILABLE:
        return {"error": "ChEMBL database not available", "drug_name": drug_name}

    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            # Find the drug by name
            cursor.execute("""
                SELECT DISTINCT md.molregno, md.chembl_id, md.pref_name
                FROM molecule_dictionary md
                LEFT JOIN molecule_synonyms ms ON md.molregno = ms.molregno
                WHERE LOWER(md.pref_name) LIKE LOWER(?)
                   OR LOWER(ms.synonyms) LIKE LOWER(?)
                LIMIT 5
            """, (f"%{drug_name}%", f"%{drug_name}%"))

            drugs = cursor.fetchall()

            if not drugs:
                return {"error": "Drug not found", "drug_name": drug_name, "targets": []}

            # Use the first match
            drug = drugs[0]
            molregno = drug["molregno"]

            # Get targets via activities
            cursor.execute("""
                SELECT DISTINCT
                    td.chembl_id as target_chembl_id,
                    td.pref_name as target_name,
                    td.target_type,
                    td.organism,
                    cs.accession as uniprot_id,
                    a.standard_type,
                    a.standard_value,
                    a.standard_units,
                    a.pchembl_value
                FROM activities a
                JOIN assays ass ON a.assay_id = ass.assay_id
                JOIN target_dictionary td ON ass.tid = td.tid
                LEFT JOIN target_components tc ON td.tid = tc.tid
                LEFT JOIN component_sequences cs ON tc.component_id = cs.component_id
                WHERE a.molregno = ?
                  AND a.standard_type IN ('IC50', 'Ki', 'Kd', 'EC50', 'Activity')
                ORDER BY a.pchembl_value DESC NULLS LAST
                LIMIT ?
            """, (molregno, limit))

            targets = []
            for row in cursor.fetchall():
                targets.append({
                    "target_chembl_id": row["target_chembl_id"],
                    "target_name": row["target_name"],
                    "target_type": row["target_type"],
                    "organism": row["organism"],
                    "uniprot_id": row["uniprot_id"],
                    "activity": {
                        "type": row["standard_type"],
                        "value": row["standard_value"],
                        "units": row["standard_units"],
                        "pchembl": row["pchembl_value"],
                    }
                })

            return {
                "drug_name": drug_name,
                "chembl_id": drug["chembl_id"],
                "pref_name": drug["pref_name"],
                "targets": targets,
                "total_targets": len(targets),
            }

    except Exception as e:
        logger.error(f"Error getting drug targets for {drug_name}: {e}")
        return {"error": str(e), "drug_name": drug_name, "targets": []}


def get_target_compounds(uniprot_id: str, limit: int = 50) -> Dict[str, Any]:
    """
    Get compounds that target a specific protein.

    Args:
        uniprot_id: UniProt accession ID
        limit: Maximum number of compounds to return

    Returns:
        Dict with target info and list of compounds
    """
    if not CHEMBL_AVAILABLE:
        return {"error": "ChEMBL database not available", "uniprot_id": uniprot_id}

    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            # Get target info
            cursor.execute("""
                SELECT
                    td.tid,
                    td.chembl_id as target_chembl_id,
                    td.pref_name as target_name,
                    td.target_type,
                    td.organism,
                    cs.description as protein_description
                FROM target_dictionary td
                JOIN target_components tc ON td.tid = tc.tid
                JOIN component_sequences cs ON tc.component_id = cs.component_id
                WHERE cs.accession = ?
                LIMIT 1
            """, (uniprot_id,))

            target = cursor.fetchone()

            if not target:
                return {"error": "Target not found", "uniprot_id": uniprot_id, "compounds": []}

            # Get compounds
            cursor.execute("""
                SELECT DISTINCT
                    md.chembl_id,
                    md.pref_name,
                    md.max_phase,
                    md.molecule_type,
                    a.standard_type,
                    a.standard_value,
                    a.standard_units,
                    a.pchembl_value
                FROM activities a
                JOIN molecule_dictionary md ON a.molregno = md.molregno
                JOIN assays ass ON a.assay_id = ass.assay_id
                JOIN target_dictionary td ON ass.tid = td.tid
                JOIN target_components tc ON td.tid = tc.tid
                JOIN component_sequences cs ON tc.component_id = cs.component_id
                WHERE cs.accession = ?
                  AND a.standard_type IN ('IC50', 'Ki', 'Kd', 'EC50')
                ORDER BY a.pchembl_value DESC NULLS LAST
                LIMIT ?
            """, (uniprot_id, limit))

            compounds = []
            for row in cursor.fetchall():
                compounds.append({
                    "chembl_id": row["chembl_id"],
                    "pref_name": row["pref_name"],
                    "max_phase": row["max_phase"],
                    "molecule_type": row["molecule_type"],
                    "activity": {
                        "type": row["standard_type"],
                        "value": row["standard_value"],
                        "units": row["standard_units"],
                        "pchembl": row["pchembl_value"],
                    }
                })

            return {
                "uniprot_id": uniprot_id,
                "target_chembl_id": target["target_chembl_id"],
                "target_name": target["target_name"],
                "target_type": target["target_type"],
                "organism": target["organism"],
                "protein_description": target["protein_description"],
                "compounds": compounds,
                "total_compounds": len(compounds),
            }

    except Exception as e:
        logger.error(f"Error getting target compounds for {uniprot_id}: {e}")
        return {"error": str(e), "uniprot_id": uniprot_id, "compounds": []}


def get_drug_mechanisms(drug_name: str) -> Dict[str, Any]:
    """
    Get mechanism of action for a drug.

    Args:
        drug_name: Drug name to search

    Returns:
        Dict with drug info and mechanisms of action
    """
    if not CHEMBL_AVAILABLE:
        return {"error": "ChEMBL database not available", "drug_name": drug_name}

    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            # Find the drug
            cursor.execute("""
                SELECT DISTINCT md.molregno, md.chembl_id, md.pref_name
                FROM molecule_dictionary md
                LEFT JOIN molecule_synonyms ms ON md.molregno = ms.molregno
                WHERE LOWER(md.pref_name) LIKE LOWER(?)
                   OR LOWER(ms.synonyms) LIKE LOWER(?)
                LIMIT 1
            """, (f"%{drug_name}%", f"%{drug_name}%"))

            drug = cursor.fetchone()

            if not drug:
                return {"error": "Drug not found", "drug_name": drug_name, "mechanisms": []}

            # Get mechanisms
            cursor.execute("""
                SELECT
                    dm.mechanism_of_action,
                    dm.action_type,
                    dm.direct_interaction,
                    dm.disease_efficacy,
                    dm.molecular_mechanism,
                    td.chembl_id as target_chembl_id,
                    td.pref_name as target_name,
                    td.target_type,
                    td.organism
                FROM drug_mechanism dm
                JOIN target_dictionary td ON dm.tid = td.tid
                WHERE dm.molregno = ?
            """, (drug["molregno"],))

            mechanisms = []
            for row in cursor.fetchall():
                mechanisms.append({
                    "mechanism_of_action": row["mechanism_of_action"],
                    "action_type": row["action_type"],
                    "direct_interaction": bool(row["direct_interaction"]),
                    "disease_efficacy": bool(row["disease_efficacy"]),
                    "molecular_mechanism": row["molecular_mechanism"],
                    "target": {
                        "chembl_id": row["target_chembl_id"],
                        "name": row["target_name"],
                        "type": row["target_type"],
                        "organism": row["organism"],
                    }
                })

            return {
                "drug_name": drug_name,
                "chembl_id": drug["chembl_id"],
                "pref_name": drug["pref_name"],
                "mechanisms": mechanisms,
                "total_mechanisms": len(mechanisms),
            }

    except Exception as e:
        logger.error(f"Error getting drug mechanisms for {drug_name}: {e}")
        return {"error": str(e), "drug_name": drug_name, "mechanisms": []}


def search_compounds_by_name(name: str, limit: int = 20) -> Dict[str, Any]:
    """
    Search compounds by name.

    Args:
        name: Search query
        limit: Maximum results to return

    Returns:
        Dict with search results
    """
    if not CHEMBL_AVAILABLE:
        return {"error": "ChEMBL database not available", "query": name}

    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT DISTINCT
                    md.chembl_id,
                    md.pref_name,
                    md.max_phase,
                    md.molecule_type,
                    md.therapeutic_flag,
                    md.first_approval
                FROM molecule_dictionary md
                LEFT JOIN molecule_synonyms ms ON md.molregno = ms.molregno
                WHERE LOWER(md.pref_name) LIKE LOWER(?)
                   OR LOWER(ms.synonyms) LIKE LOWER(?)
                ORDER BY md.max_phase DESC NULLS LAST, md.pref_name
                LIMIT ?
            """, (f"%{name}%", f"%{name}%", limit))

            results = []
            for row in cursor.fetchall():
                results.append({
                    "chembl_id": row["chembl_id"],
                    "pref_name": row["pref_name"],
                    "max_phase": row["max_phase"],
                    "molecule_type": row["molecule_type"],
                    "therapeutic_flag": bool(row["therapeutic_flag"]),
                    "first_approval": row["first_approval"],
                })

            return {
                "query": name,
                "results": results,
                "total_results": len(results),
            }

    except Exception as e:
        logger.error(f"Error searching compounds for {name}: {e}")
        return {"error": str(e), "query": name, "results": []}


def get_bioactivities(chembl_id: str, limit: int = 100) -> Dict[str, Any]:
    """
    Get bioactivity data for a compound.

    Args:
        chembl_id: ChEMBL compound ID
        limit: Maximum activities to return

    Returns:
        Dict with compound info and bioactivities
    """
    if not CHEMBL_AVAILABLE:
        return {"error": "ChEMBL database not available", "chembl_id": chembl_id}

    chembl_id = chembl_id.upper()
    if not chembl_id.startswith("CHEMBL"):
        chembl_id = f"CHEMBL{chembl_id}"

    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            # Get molregno
            cursor.execute("""
                SELECT molregno, pref_name
                FROM molecule_dictionary
                WHERE chembl_id = ?
            """, (chembl_id,))

            mol = cursor.fetchone()

            if not mol:
                return {"error": "Compound not found", "chembl_id": chembl_id, "activities": []}

            # Get activities
            cursor.execute("""
                SELECT
                    a.activity_id,
                    a.standard_type,
                    a.standard_relation,
                    a.standard_value,
                    a.standard_units,
                    a.pchembl_value,
                    a.activity_comment,
                    ass.description as assay_description,
                    ass.assay_type,
                    td.chembl_id as target_chembl_id,
                    td.pref_name as target_name,
                    td.target_type
                FROM activities a
                JOIN assays ass ON a.assay_id = ass.assay_id
                LEFT JOIN target_dictionary td ON ass.tid = td.tid
                WHERE a.molregno = ?
                ORDER BY a.pchembl_value DESC NULLS LAST
                LIMIT ?
            """, (mol["molregno"], limit))

            activities = []
            for row in cursor.fetchall():
                activities.append({
                    "activity_id": row["activity_id"],
                    "type": row["standard_type"],
                    "relation": row["standard_relation"],
                    "value": row["standard_value"],
                    "units": row["standard_units"],
                    "pchembl": row["pchembl_value"],
                    "comment": row["activity_comment"],
                    "assay": {
                        "description": row["assay_description"],
                        "type": row["assay_type"],
                    },
                    "target": {
                        "chembl_id": row["target_chembl_id"],
                        "name": row["target_name"],
                        "type": row["target_type"],
                    }
                })

            return {
                "chembl_id": chembl_id,
                "pref_name": mol["pref_name"],
                "activities": activities,
                "total_activities": len(activities),
            }

    except Exception as e:
        logger.error(f"Error getting bioactivities for {chembl_id}: {e}")
        return {"error": str(e), "chembl_id": chembl_id, "activities": []}


def get_database_stats() -> Dict[str, Any]:
    """Get ChEMBL database statistics."""
    if not CHEMBL_AVAILABLE:
        return {
            "available": False,
            "path": str(CHEMBL_DB_PATH),
            "error": "Database not found"
        }

    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            stats = {
                "available": True,
                "path": str(CHEMBL_DB_PATH),
                "version": "ChEMBL 36",
            }

            # Get counts
            tables = [
                ("molecules", "molecule_dictionary"),
                ("activities", "activities"),
                ("assays", "assays"),
                ("targets", "target_dictionary"),
                ("mechanisms", "drug_mechanism"),
            ]

            for name, table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats[f"total_{name}"] = count

            return stats

    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {
            "available": True,
            "path": str(CHEMBL_DB_PATH),
            "error": str(e)
        }


def check_chembl_status() -> Dict[str, Any]:
    """Check ChEMBL data availability."""
    status = {
        "path": str(CHEMBL_PATH),
        "available": CHEMBL_PATH.exists(),
        "database_available": CHEMBL_AVAILABLE,
        "database_path": str(CHEMBL_DB_PATH),
        "files": {},
    }

    if not status["available"]:
        return status

    # Check files
    files_to_check = [
        ("sqlite_database", CHEMBL_DB_PATH),
        ("chemreps", CHEMBL_PATH / "chembl_36_chemreps.txt.gz"),
        ("sdf", CHEMBL_PATH / "chembl_36.sdf.gz"),
        ("uniprot_mapping", CHEMBL_PATH / "chembl_uniprot_mapping.txt"),
    ]

    for name, path in files_to_check:
        if path.exists():
            size = path.stat().st_size
            status["files"][name] = {
                "exists": True,
                "path": str(path),
                "size_bytes": size,
                "size_mb": round(size / (1024 * 1024), 2),
            }
        else:
            status["files"][name] = {"exists": False, "path": str(path)}

    # Get database stats if available
    if CHEMBL_AVAILABLE:
        try:
            stats = get_database_stats()
            status["database_stats"] = stats
        except:
            pass

    return status
