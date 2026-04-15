"""
Train Phase 2 Disease Classifiers
Expands Layer 4c from 10 to 15 MIMIC-IV trained ML models.

Phase 2 targets:
- G41: Status Epilepticus
- J80: ARDS (Acute Respiratory Distress Syndrome)
- D65: DIC (Disseminated Intravascular Coagulation)
- E87: Electrolyte Disorders
- I46: Cardiac Arrest
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set MIMIC path before importing
os.environ['MIMIC_PATH'] = r"C:\Users\letsa\Downloads\mimic-iv-3.1\mimic-iv-3.1"

from app.core.diagnostic_engine.ml.disease_models import DiseaseModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'scripts' / 'training_phase2.log')
    ]
)
logger = logging.getLogger(__name__)

# Phase 2 diseases to train
PHASE2_DISEASES = [
    {
        'icd': 'G41',
        'name': 'Status Epilepticus',
        'max_patients': 5000
    },
    {
        'icd': 'J80',
        'name': 'Acute Respiratory Distress Syndrome',
        'max_patients': 5000
    },
    {
        'icd': 'D65',
        'name': 'Disseminated Intravascular Coagulation',
        'max_patients': 5000
    },
    {
        'icd': 'E87',
        'name': 'Electrolyte Disorders',
        'max_patients': 5000
    },
    {
        'icd': 'I46',
        'name': 'Cardiac Arrest',
        'max_patients': 5000
    },
]


def train_phase2():
    """Train all Phase 2 classifiers."""
    logger.info("=" * 60)
    logger.info("PHASE 2 CLASSIFIER TRAINING")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Initialize model manager
    manager = DiseaseModelManager()

    results = []

    for i, disease in enumerate(PHASE2_DISEASES, 1):
        logger.info("")
        logger.info(f"[{i}/{len(PHASE2_DISEASES)}] Training: {disease['icd']} - {disease['name']}")
        logger.info("-" * 50)

        try:
            model = manager.train_model(
                disease_icd=disease['icd'],
                disease_name=disease['name'],
                max_patients=disease['max_patients'],
                model_type='gradient_boosting',
                test_size=0.2
            )

            result = {
                'icd': disease['icd'],
                'name': disease['name'],
                'status': 'SUCCESS',
                'auc': model.metrics['roc_auc'],
                'accuracy': model.metrics['accuracy'],
                'train_samples': model.metrics['train_samples'],
                'test_samples': model.metrics['test_samples']
            }

            logger.info(f"SUCCESS: {disease['icd']} - AUC: {model.metrics['roc_auc']:.3f}")

        except Exception as e:
            logger.error(f"FAILED: {disease['icd']} - {str(e)}")
            result = {
                'icd': disease['icd'],
                'name': disease['name'],
                'status': 'FAILED',
                'error': str(e)
            }

        results.append(result)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)

    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] == 'FAILED']

    logger.info(f"Successful: {len(successful)}/{len(results)}")
    logger.info(f"Failed: {len(failed)}/{len(results)}")

    if successful:
        logger.info("")
        logger.info("Trained Models:")
        for r in successful:
            logger.info(f"  {r['icd']}: {r['name']}")
            logger.info(f"    AUC: {r['auc']:.3f}, Accuracy: {r['accuracy']:.3f}")
            logger.info(f"    Samples: {r['train_samples']} train, {r['test_samples']} test")

    if failed:
        logger.info("")
        logger.info("Failed Models:")
        for r in failed:
            logger.info(f"  {r['icd']}: {r['name']} - {r['error']}")

    logger.info("")
    logger.info(f"Completed at: {datetime.now().isoformat()}")

    return results


if __name__ == '__main__':
    results = train_phase2()

    # Exit with error code if any failed
    failed = [r for r in results if r['status'] == 'FAILED']
    sys.exit(1 if failed else 0)
