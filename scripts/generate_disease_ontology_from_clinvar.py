"""
Generate expanded disease_ontology.json from ClinVar data.
This creates disease patterns for thousands of genetic conditions.

Usage:
    python scripts/generate_disease_ontology_from_clinvar.py

Output:
    app/core/diagnostic_engine/data/disease_ontology_clinvar.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.diagnostic_engine.data_sources.clinvar_loader import ClinVarLoader


def generate_ontology(output_path: str = None, min_genes: int = 1, max_diseases: int = 5000):
    """
    Generate disease ontology from ClinVar.

    Args:
        output_path: Path to write output JSON
        min_genes: Minimum genes per disease to include
        max_diseases: Maximum diseases to include
    """
    if output_path is None:
        output_path = Path(__file__).parent.parent / 'app/core/diagnostic_engine/data/disease_ontology_clinvar.json'

    print("Loading ClinVar...")
    clinvar = ClinVarLoader()
    if not clinvar.load():
        print(f"Failed to load ClinVar: {clinvar._load_error}")
        return

    stats = clinvar.get_stats()
    print(f"ClinVar loaded: {stats['disease_count']:,} diseases, {stats['gene_count']:,} genes")

    print(f"\nGenerating disease ontology (min_genes={min_genes}, max={max_diseases})...")

    # Build disease entries directly from disease_gene_map (faster)
    disease_entries = []

    for disease_name, genes in clinvar.disease_gene_map.items():
        # Skip vague entries
        if len(disease_name) < 5:
            continue
        if disease_name.lower() in ('not specified', 'not provided', 'see cases'):
            continue

        gene_list = list(genes)
        if len(gene_list) >= min_genes:
            disease_entries.append({
                'name': disease_name,
                'genes': gene_list,
                'gene_count': len(gene_list)
            })

    print(f"Found {len(disease_entries):,} diseases with >= {min_genes} genes")

    # Sort by gene count (most evidence first)
    disease_entries.sort(key=lambda x: x['gene_count'], reverse=True)

    # Take top N
    disease_entries = disease_entries[:max_diseases]

    print(f"Processing top {len(disease_entries):,} diseases...")

    ontology = {"diseases": {}, "metadata": {
        "source": "ClinVar",
        "total_diseases": 0,
        "total_genes": 0
    }}

    all_genes = set()
    seen_ids = set()

    for i, entry in enumerate(disease_entries):
        if i % 500 == 0:
            print(f"  Processed {i:,}/{len(disease_entries):,}...")

        disease_name = entry['name']

        # Create disease ID (sanitized)
        disease_id = sanitize_id(disease_name)

        # Skip duplicates
        if disease_id in seen_ids:
            continue
        seen_ids.add(disease_id)

        genes = entry['genes'][:10]  # Top 10 genes
        all_genes.update(genes)

        ontology['diseases'][disease_id] = {
            'name': disease_name,
            'source': 'ClinVar',
            'category': 'genetic',
            'genes': genes,
            'gene_count': entry['gene_count'],
            'patterns': {
                'genetic': [
                    {'gene': g, 'condition': 'pathogenic_variant'}
                    for g in genes[:5]
                ]
            },
            'confidence_factors': {
                'base': 0.50,
                'per_pathogenic_variant': 0.15,
                'per_gene_match': 0.10,
                'max': 0.95
            },
            'recommended_followup': [
                'genetic_counseling',
                'family_history_review',
                'specialist_referral'
            ]
        }

    ontology['metadata']['total_diseases'] = len(ontology['diseases'])
    ontology['metadata']['total_genes'] = len(all_genes)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ontology, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(ontology['diseases']):,} disease patterns")
    print(f"Covering {len(all_genes):,} genes")
    print(f"Saved to: {output_path}")

    return ontology


def sanitize_id(name: str) -> str:
    """Convert disease name to valid ID."""
    # Remove special characters, lowercase, truncate
    id_str = name.lower()
    id_str = ''.join(c if c.isalnum() or c == ' ' else '_' for c in id_str)
    id_str = id_str.replace(' ', '_')
    id_str = '_'.join(filter(None, id_str.split('_')))  # Remove consecutive underscores
    return id_str[:60]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate disease ontology from ClinVar')
    parser.add_argument('--min-genes', type=int, default=1, help='Minimum genes per disease')
    parser.add_argument('--max-diseases', type=int, default=5000, help='Maximum diseases to include')
    parser.add_argument('--output', type=str, default=None, help='Output path')

    args = parser.parse_args()

    generate_ontology(
        output_path=args.output,
        min_genes=args.min_genes,
        max_diseases=args.max_diseases
    )
