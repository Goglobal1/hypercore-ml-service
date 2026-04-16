#!/usr/bin/env python3
"""
Phase 4b MIMIC-IV Classifier Training
Train 6 additional disease models to expand Layer 4c coverage.

Targets:
- I42: Cardiomyopathy (Cardiac)
- I47: Paroxysmal Tachycardia (Cardiac)
- G41: Status Epilepticus (Neurological emergency)
- G93: Encephalopathy (Neurological)
- I62: Other Nontraumatic ICH (Neurological)
- F11: Opioid Use Disorder (Psychiatric - crisis priority)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.diagnostic_engine.ml.disease_models import DiseaseModelManager

# Phase 4b disease targets
PHASE4B_DISEASES = [
    {'icd': 'I42', 'name': 'Cardiomyopathy', 'max_patients': 5000},
    {'icd': 'I47', 'name': 'Paroxysmal Tachycardia', 'max_patients': 5000},
    {'icd': 'G41', 'name': 'Status Epilepticus', 'max_patients': 5000},
    {'icd': 'G93', 'name': 'Encephalopathy', 'max_patients': 5000},
    {'icd': 'I62', 'name': 'Other Nontraumatic Intracranial Hemorrhage', 'max_patients': 5000},
    {'icd': 'F11', 'name': 'Opioid Use Disorder', 'max_patients': 5000},
]


def main():
    print("=" * 70)
    print("PHASE 4b: MIMIC-IV Disease Classifier Training")
    print("=" * 70)
    print(f"\nTraining {len(PHASE4B_DISEASES)} new disease models:")
    for d in PHASE4B_DISEASES:
        print(f"  - {d['icd']}: {d['name']}")
    print()

    # Initialize model manager
    manager = DiseaseModelManager()

    results = []
    for disease in PHASE4B_DISEASES:
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
                # disease_model is a DiseaseModel object with a metrics attribute
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
    print("PHASE 4b TRAINING SUMMARY")
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
    print(f"Successfully trained: {successful}/{len(PHASE4B_DISEASES)} models")

    # Total model count
    print("\n" + "=" * 70)
    print("TOTAL MODELS AFTER PHASE 4b")
    print("=" * 70)
    model_count = manager.load_models()  # Returns count, stores in manager.models
    print(f"Total MIMIC-IV trained models: {model_count}")
    for icd, model in sorted(manager.models.items()):
        auc = model.metrics.get('roc_auc', 0) if hasattr(model, 'metrics') else 0
        name = model.disease_name if hasattr(model, 'disease_name') else 'Unknown'
        print(f"  {icd}: {name} (AUC: {auc:.4f})")

    return results


if __name__ == "__main__":
    main()
