#!/usr/bin/env python3
"""
Phase 4q MIMIC-IV Classifier Training
Train 6 additional disease models to expand Layer 4c coverage.

Targets:
- I44: Atrioventricular and Left Bundle-Branch Block (Cardiac)
- J20: Acute Bronchitis (Respiratory)
- G43: Migraine (Neurological)
- N40: Benign Prostatic Hyperplasia (Urological)
- I67: Other Cerebrovascular Diseases (Neurological/Vascular)
- K55: Vascular Disorders of Intestine (GI/Vascular)
"""

import sys
import os

# Set MIMIC_PATH if not already set
if not os.environ.get('MIMIC_PATH'):
    os.environ['MIMIC_PATH'] = 'C:/Users/letsa/Downloads/mimic-iv-3.1/mimic-iv-3.1'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.diagnostic_engine.ml.disease_models import DiseaseModelManager

# Phase 4q disease targets
PHASE4Q_DISEASES = [
    {'icd': 'I44', 'name': 'Atrioventricular and Left Bundle-Branch Block', 'max_patients': 5000},
    {'icd': 'J20', 'name': 'Acute Bronchitis', 'max_patients': 5000},
    {'icd': 'G43', 'name': 'Migraine', 'max_patients': 5000},
    {'icd': 'N40', 'name': 'Benign Prostatic Hyperplasia', 'max_patients': 5000},
    {'icd': 'I67', 'name': 'Other Cerebrovascular Diseases', 'max_patients': 5000},
    {'icd': 'K55', 'name': 'Vascular Disorders of Intestine', 'max_patients': 5000},
]


def main():
    print("=" * 70)
    print("PHASE 4q: MIMIC-IV Disease Classifier Training")
    print("=" * 70)
    print(f"\nTraining {len(PHASE4Q_DISEASES)} new disease models:")
    for d in PHASE4Q_DISEASES:
        print(f"  - {d['icd']}: {d['name']}")
    print()

    # Initialize model manager
    manager = DiseaseModelManager()

    results = []
    for disease in PHASE4Q_DISEASES:
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
    print("PHASE 4q TRAINING SUMMARY")
    print("=" * 70)
    print(f"{'ICD':<8} {'Disease':<45} {'Status':<10} {'AUC':<8} {'Samples'}")
    print("-" * 75)

    successful = 0
    for r in results:
        status_str = r['status']
        auc_str = f"{r['auc']:.4f}" if r['auc'] > 0 else "N/A"
        samples_str = f"{r['samples']:,}" if r['samples'] > 0 else "N/A"
        print(f"{r['icd']:<8} {r['name']:<45} {status_str:<10} {auc_str:<8} {samples_str}")
        if r['status'] == 'SUCCESS':
            successful += 1

    print("-" * 75)
    print(f"Successfully trained: {successful}/{len(PHASE4Q_DISEASES)} models")

    # Total model count
    print("\n" + "=" * 70)
    print("TOTAL MODELS AFTER PHASE 4q")
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
