"""Extract lab item IDs for biomarkers we care about."""
import pandas as pd
import gzip
from config import FILES

# Biomarkers we want to track
BIOMARKERS_OF_INTEREST = {
    # Sepsis markers
    "lactate": ["lactate", "lactic acid"],
    "crp": ["c-reactive protein", "crp"],
    "wbc": ["white blood cell", "wbc", "leukocyte"],
    "procalcitonin": ["procalcitonin"],

    # Cardiac markers
    "troponin": ["troponin"],
    "bnp": ["bnp", "b-type natriuretic", "brain natriuretic"],
    "ck_mb": ["ck-mb", "creatine kinase mb"],

    # Kidney markers
    "creatinine": ["creatinine"],
    "bun": ["bun", "blood urea nitrogen", "urea nitrogen"],
    "potassium": ["potassium"],
    "gfr": ["gfr", "glomerular filtration"],

    # Liver markers
    "alt": ["alt", "alanine aminotransferase", "sgpt"],
    "ast": ["ast", "aspartate aminotransferase", "sgot"],
    "bilirubin": ["bilirubin"],
    "albumin": ["albumin"],
    "inr": ["inr", "international normalized ratio"],

    # Hematologic
    "hemoglobin": ["hemoglobin", "hgb"],
    "platelets": ["platelet"],
    "fibrinogen": ["fibrinogen"],
}

def find_lab_items():
    """Find itemid values for each biomarker."""
    print("Loading d_labitems...")
    df = pd.read_csv(FILES["d_labitems"], compression='gzip')
    print(f"Total lab items: {len(df)}")

    results = {}
    for biomarker, keywords in BIOMARKERS_OF_INTEREST.items():
        matches = []
        for _, row in df.iterrows():
            label = str(row.get('label', '')).lower()
            if any(kw in label for kw in keywords):
                matches.append({
                    "itemid": row['itemid'],
                    "label": row.get('label', ''),
                    "fluid": row.get('fluid', ''),
                    "category": row.get('category', '')
                })
        results[biomarker] = matches
        print(f"{biomarker}: {len(matches)} items found")

    return results

if __name__ == "__main__":
    items = find_lab_items()

    # Save to file
    import json
    with open("lab_item_mapping.json", "w") as f:
        json.dump(items, f, indent=2)
    print("\nSaved to lab_item_mapping.json")
