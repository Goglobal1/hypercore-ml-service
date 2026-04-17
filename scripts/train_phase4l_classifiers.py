#!/usr/bin/env python3
"""
Phase 4l MIMIC-IV Classifier Training
Train 6 additional disease models to expand Layer 4c coverage.

Targets:
- R18: Ascites (Hepatic/GI)
- L89: Pressure Ulcer (Dermatologic)
- I27: Pulmonary Heart Disease (Cardiopulmonary)
- K50: Crohn's Disease (GI/Autoimmune)
- K51: Ulcerative Colitis (GI/Autoimmune)
- T40: Opioid Poisoning (Toxicology)
"""

import sys
import os

# Set MIMIC_PATH if not already set
if not os.environ.get('MIMIC_PATH'):
    os.environ['MIMIC_PATH'] = 'C:/Users/letsa/Downloads/mimic-iv-3.1/mimic-iv-3.1'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.diagnostic_engine.ml.disease_models import DiseaseModelManager

# Phase 4l disease targets
PHASE4L_DISEASES = [
    {'icd': 'R18', 'name': 'Ascites', 'max_patients': 5000},
    {'icd': 'L89', 'name': 'Pressure Ulcer', 'max_patients': 5000},
    {'icd': 'I27', 'name': 'Pulmonary Heart Disease', 'max_patients': 5000},
    {'icd': 'K50', 'name': 'Crohn Disease', 'max_patients': 5000},
    {'icd': 'K51', 'name': 'Ulcerative Colitis', 'max_patients': 5000},
    {'icd': 'T40', 'name': 'Opioid Poisoning', 'max_patients': 5000},
]


def main():
    print("=" * 70)
    print("PHASE 4l: MIMIC-IV Disease Classifier Training")
    print("=" * 70)
    print(f"\nTraining {len(PHASE4L_DISEASES)} new disease models:")
    for d in PHASE4L_DISEASES:
        print(f"  - {d['icd']}: {d['name']}")
    print()

    # Initialize model manager
    manager = DiseaseModelManager()

    results = []
    for disease in PHASE4L_DISEASES:
        print("\n" + "=" * 70)
        print(f"Training model for {disease['icd']}: {disease['name']}")
        print("=" * 70)

        try:
            disease_model = manager.train_model(
                disease_icd=disease['icd'],
                disease_name=disease['name'],
                max_patients=disease['max_patients']
            )

            if disease_model:
                metrics = disease_model.metrics
                results.append({
                    'icd': disease['icd'],
                    'name': disease['name'],
                    'status': 'SUCCESS',
                    'auc': metrics.get('roc_auc', 0),
                    'accuracy': metrics.get('accuracy', 0),
                    'samples': metrics.get('train_samples', 0)
                })
                print(f"\n[SUCCESS] {disease['icd']} trained - AUC: {metrics.get('roc_auc', 0):.4f}")
            else:
                results.append({
                    'icd': disease['icd'],
                    'name': disease['name'],
                    'status': 'FAILED',
                    'auc': 0,
                    'accuracy': 0,
                    'samples': 0
                })
                print(f"\n[FAILED] {disease['icd']} - No model returned")

        except Exception as e:
            results.append({
                'icd': disease['icd'],
                'name': disease['name'],
                'status': 'ERROR',
                'error': str(e),
                'auc': 0,
                'accuracy': 0,
                'samples': 0
            })
            print(f"\n[ERROR] {disease['icd']} - {str(e)}")

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 4l TRAINING SUMMARY")
    print("=" * 70)
    print(f"{'ICD':<8} {'Disease':<40} {'Status':<10} {'AUC':<8} {'Samples'}")
    print("-" * 70)

    successful = 0
    for r in results:
        status_str = r['status']
        auc_str = f"{r['auc']:.4f}" if r['auc'] > 0 else "N/A"
        samples_str = f"{r['samples']:,}" if r['samples'] > 0 else "N/A"
        print(f"{r['icd']:<8} {r['name']:<40} {status_str:<10} {auc_str:<8} {samples_str}")
        if r['status'] == 'SUCCESS':
            successful += 1

    print("-" * 70)
    print(f"Successfully trained: {successful}/{len(PHASE4L_DISEASES)} models")

    # Total model count
    print("\n" + "=" * 70)
    print("TOTAL MODELS AFTER PHASE 4l")
    print("=" * 70)
    model_count = manager.load_models()
    print(f"Total MIMIC-IV trained models: {model_count}")
    for icd, model in sorted(manager.models.items()):
        auc = model.metrics.get('roc_auc', 0) if hasattr(model, 'metrics') else 0
        name = model.disease_name if hasattr(model, 'disease_name') else 'Unknown'
        print(f"  {icd}: {name} (AUC: {auc:.4f})")

    return results


if __name__ == "__main__":
    main()
