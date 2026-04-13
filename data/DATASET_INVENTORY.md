# HyperCore Dataset Inventory
Generated: 2026-04-12

## Storage Overview

| Drive | Type | Total Size | Description |
|-------|------|------------|-------------|
| C: | Local SSD | 1TB | OS drive |
| F: | Seagate Hub | 8TB | External - Primary dataset storage |
| G: | Google Drive | 1TB | Cloud sync |
| H: | Google Drive | 200GB | Cloud sync |

---

## Dataset Summary

| Category | Location | Size | Integration Status |
|----------|----------|------|-------------------|
| MIMIC-IV 3.1 | F:/mimic-iv-3.1 | 10 GB | ENV_VAR (too large) |
| ClinVar Full | F:/DATASETS/GENETICS/ClinVar | 1.1 GB | PARTIAL (variant_summary at build) |
| PharmGKB | F:/DATASETS/PHARMACEUTICAL | 42 GB | INTEGRATED (core files in repo) |
| Human Protein Atlas | F:/DATASETS/PROTEOMICS | 5.2 GB | ENV_VAR (too large) |
| GEO Expression | F:/DATASETS/GENE_EXPRESSION | 8.3 GB | ENV_VAR (too large) |
| NHANES | F:/DATASETS/POPULATION/NHANES | 146 MB | ENV_VAR (SAS format) |
| WHO Statistics | F:/DATASETS/SURVEILLANCE/WHO | 123 MB | COPY_TO_REPO |
| FAERS | C:/Users/letsa/Downloads | 2.7 GB | ENV_VAR (too large) |
| HPO | Downloaded at build | ~5 MB | INTEGRATED |

---

## Detailed Inventory

### 1. MIMIC-IV v3.1 (ICU Clinical Data)
**Location:** `F:/mimic-iv-3.1/mimic-iv-3.1/`
**Size:** 10 GB (extracted), 10.5 GB (zip)
**Status:** TOO LARGE FOR GIT - Use environment variable
**Contains:**
- `hosp/` - Hospital data
  - labevents.csv.gz (2.5 GB) - Laboratory results
  - chartevents.csv.gz (3.3 GB) - ICU chart events
  - prescriptions.csv.gz (579 MB) - Medication prescriptions
  - emar.csv.gz (774 MB) - Medication administration records
  - pharmacy.csv.gz (502 MB) - Pharmacy data
  - diagnoses_icd.csv.gz (33 MB) - ICD diagnoses
  - patients.csv.gz (2.8 MB) - Patient demographics
- `icu/` - ICU specific data
  - chartevents.csv.gz (3.3 GB)
  - inputevents.csv.gz (383 MB)
  - outputevents.csv.gz (48 MB)
  - icustays.csv.gz (3.2 MB)

**Environment Variable:** `MIMIC_PATH=F:/mimic-iv-3.1/mimic-iv-3.1`

**MIMIC Extensions (F:/ root):**
- mimic-cxr-reports.zip (142 MB) - Chest X-ray reports
- mimic-iv-ext-cardiac-disease-1.0.0.zip (27 MB) - Cardiac disease
- mimic-iv-ext-clinical-decision-support (36 MB) - CDS referral
- mimic-iv-ext-direct-1.0.0.zip (1.2 MB) - Direct annotations
- mimic-ext-drugdetection-1.0.0.zip (396 KB) - Drug detection

---

### 2. ClinVar (Genomic Variants)
**Location:** `F:/DATASETS/GENETICS/ClinVar/`
**Size:** 1.1 GB
**Status:** PARTIAL - variant_summary downloaded at Docker build
**Contains:**
| File | Size | Description |
|------|------|-------------|
| variant_summary.txt.gz | 435 MB | Main variant database |
| submission_summary.txt.gz | 373 MB | Submission details |
| var_citations.txt | 226 MB | Literature citations |
| cross_references.txt | 112 MB | External database links |
| gene_specific_summary.txt | 3.5 MB | Gene-level summary |

**Environment Variable:** `CLINVAR_PATH=F:/DATASETS/GENETICS/ClinVar`
**Note:** Full dataset enables cross-references and citations lookup

---

### 3. PharmGKB (Pharmacogenomics)
**Location:** `F:/DATASETS/PHARMACEUTICAL/PharmGKB/`
**Size:** 42 GB (includes historical versions)
**Status:** INTEGRATED - Core files committed to repo (89 MB)
**In Repository (data/pharmgkb/):**
- relationships.tsv (15 MB) - Drug-gene relationships
- genes.tsv (11 MB) - Gene annotations
- drugs.tsv (2.2 MB) - Drug annotations
- clinicalVariants.tsv (354 KB) - Clinical variants
- drugLabels.tsv (196 KB) - FDA drug labels
- automated_annotations.tsv (9.1 MB) - ML annotations
- chemicals.tsv (2.6 MB) - Chemical compounds
- guidelineAnnotations.json/ (150+ files) - CPIC/DPWG guidelines
- pathways-biopax/ (100+ .owl files) - Drug pathways
- phenotypes.tsv - Response phenotypes
- variants.tsv - Pharmacogenomic variants
- variantAnnotations.tsv - Variant clinical evidence
- summaryAnnotations.tsv - Summary annotations
- haplotype_frequencies/ - Population frequencies (AllOfUs, UKBB)

---

### 4. Human Protein Atlas (Proteomics)
**Location:** `F:/DATASETS/PROTEOMICS/Human_Protein_Atlas/`
**Size:** 5.2 GB
**Status:** TOO LARGE FOR GIT - Use environment variable
**Contains:**
| File | Size | Description |
|------|------|-------------|
| rna_cancer_sample.tsv.gz | 1.3 GB | Cancer RNA expression |
| proteinatlas.json.gz | 11 MB | Full protein atlas |

**Additional HPA files on F:/ root:**
| File | Size | Description |
|------|------|-------------|
| transcript_rna_brain.tsv.zip | 1.2 GB | Brain transcriptomics |
| transcript_rna_tissue.tsv.zip | 184 MB | Tissue transcriptomics |
| rna_single_cell_cluster.tsv.zip | 197 MB | Single-cell clusters |
| rna_immune_cell_sample.tsv.zip | 34 MB | Immune cell expression |
| rna_tissue_consensus.tsv.zip | 5.3 MB | Tissue consensus |
| normal_ihc_data.tsv.zip | 5.7 MB | Normal IHC data |
| subcellular_location.tsv.zip | 252 KB | Subcellular locations |

**Environment Variable:** `HPA_PATH=F:/DATASETS/PROTEOMICS/Human_Protein_Atlas`

---

### 5. GEO Gene Expression
**Location:** `F:/DATASETS/GENE_EXPRESSION/GEO_Datasets/`
**Size:** 8.3 GB
**Status:** TOO LARGE FOR GIT - Use environment variable
**Contains:** 30+ GSE series matrix files (.txt.gz)

**Additional GEO in Downloads (C:/Users/letsa/Downloads/):**
- 50+ GSE series matrix files
- Various disease-specific expression datasets

**Environment Variable:** `GEO_DATA_PATH=F:/DATASETS/GENE_EXPRESSION/GEO_Datasets`

---

### 6. NHANES (Population Health)
**Location:** `F:/DATASETS/POPULATION/NHANES/`
**Size:** 146 MB
**Status:** ENV_VAR - SAS transport format (.xpt)
**Contains:** 200+ .xpt files including:
- DEMO_*.xpt - Demographics
- CBC_*.xpt - Complete blood count
- BIOPRO_*.xpt - Biochemistry profiles
- GLU_*.xpt - Glucose
- HDL_*.xpt - HDL cholesterol
- TCHOL_*.xpt - Total cholesterol
- TRIGLY_*.xpt - Triglycerides
- INS_*.xpt - Insulin
- GHB_*.xpt - Glycohemoglobin (HbA1c)
- HEPA/B/C_*.xpt - Hepatitis markers
- HIV_*.xpt - HIV antibodies
- COT_*.xpt - Cotinine (smoking marker)
- PBCD_*.xpt - Lead/cadmium/mercury
- PFAS_*.xpt - Per/polyfluoroalkyl substances

**Environment Variable:** `NHANES_PATH=F:/DATASETS/POPULATION/NHANES`

---

### 7. WHO Global Health Statistics
**Location:** `F:/DATASETS/SURVEILLANCE/WHO/`
**Size:** 123 MB
**Status:** COPY TO REPO (small enough)
**Contains:**

**Indicators/ (93 MB):**
| File | Size | Description |
|------|------|-------------|
| WHO-COVID-19-global-daily-data.csv | 24 MB | COVID daily cases/deaths |
| 2322814_ALL_LATEST.csv | 11 MB | Health workforce indicators |
| EF93DDB_ALL_LATEST.csv | 11 MB | Mortality indicators |
| FC5231F_ALL_LATEST.csv | 9.6 MB | Disease burden |
| COV_VAC_POLICY_2024.csv | 972 KB | Vaccination policies |
| COV_VAC_UPTAKE_*.csv | 1.2 MB | Vaccination uptake |
| WHO-COVID-19-global-data.csv | 3.5 MB | COVID summary |
| WHO-COVID-19-global-hosp-icu-data.csv | 2.9 MB | Hospitalization data |
| 20+ additional indicator files | ~30 MB | Various health metrics |

**World Health Statistics 2025:** 5.7 MB (F:/ root)

**Environment Variable:** `WHO_PATH=F:/DATASETS/SURVEILLANCE/WHO`

---

### 8. FAERS (FDA Adverse Event Reporting)
**Location:** `C:/Users/letsa/Downloads/`
**Size:** ~2.7 GB (53 quarterly files)
**Status:** TOO LARGE FOR GIT - Use environment variable
**Contains:** Quarterly FAERS ASCII dumps from 2012Q4 to 2025Q4
- Each quarter: 25-74 MB
- Includes: DEMO, DRUG, REAC, OUTC, RPSR, THER, INDI files
- 13 years of adverse event reports

**Environment Variable:** `FAERS_PATH=C:/Users/letsa/Downloads` (or extracted location)

---

### 9. HPO (Human Phenotype Ontology)
**Location:** Downloaded at Docker build
**Size:** ~5 MB
**Status:** INTEGRATED - Downloaded in Dockerfile
**Contains:**
- genes_to_disease.txt
- genes_to_phenotype.txt
- phenotype_to_genes.txt

---

### 10. Additional Datasets Found

**On F:/ Root:**
| File | Size | Description |
|------|------|-------------|
| cancer_prognostic_data.tsv.zip | 4.4 MB | Cancer prognosis |
| cell_line_analysis_data.tsv.zip | 461 KB | Cell line data |
| blood_immunoassay_concentration.tsv.zip | 13 KB | Blood markers |
| blood_ms_concentration.tsv.zip | 39 KB | Mass spec data |
| amr-uti-*.zip | 16 MB | AMR in UTI study |
| hospitalized-patients-with-heart-failure-*.zip | 499 KB | Heart failure |
| community-acquired-pneumonia-*.zip | 38 KB | CAP study |

**In Downloads (clinical datasets):**
| File | Size | Description |
|------|------|-------------|
| biopax.zip | 163 MB | Biological pathways |
| aipatient-kg-mimic-*.zip | 128 KB | MIMIC knowledge graph |

---

## Environment Variables Configuration

Add to `.env` or system environment:

```bash
# Large datasets - configure path to your local copy
MIMIC_PATH=F:/mimic-iv-3.1/mimic-iv-3.1
CLINVAR_PATH=F:/DATASETS/GENETICS/ClinVar
HPA_PATH=F:/DATASETS/PROTEOMICS/Human_Protein_Atlas
GEO_DATA_PATH=F:/DATASETS/GENE_EXPRESSION/GEO_Datasets
NHANES_PATH=F:/DATASETS/POPULATION/NHANES
WHO_PATH=F:/DATASETS/SURVEILLANCE/WHO
FAERS_PATH=C:/Users/letsa/Downloads

# In Docker/Railway, these will use fallback paths:
# /app/data/<dataset_name>
```

---

## Integration Priority

### Already Integrated:
1. PharmGKB (core files in repo)
2. HPO (downloaded at build)
3. ClinVar (variant_summary at build)

### High Priority (copy small files):
1. WHO Statistics (~123 MB) - Global health metrics
2. ClinVar Extended (cross_references, var_citations) - Enhanced variant lookup

### Medium Priority (env vars):
1. MIMIC-IV - ICU clinical modeling
2. NHANES - Population baselines
3. Human Protein Atlas - Protein expression

### Lower Priority (large/specialized):
1. GEO Expression - Gene expression profiles
2. FAERS - Adverse event analysis
3. MIMIC Extensions - Specialized applications

---

## Code Integration Status

| Dataset | Code Ready | Data Available | Notes |
|---------|------------|----------------|-------|
| PharmGKB | Yes | Yes (repo) | Fully integrated |
| HPO | Yes | Yes (build) | Fully integrated |
| ClinVar | Partial | Partial | Add cross_references |
| MIMIC-IV | Yes | Yes (F:) | Need env var |
| NHANES | Yes | Yes (F:) | Need env var |
| HPA | Partial | Yes (F:) | Need connector |
| GEO | Partial | Yes (F:) | Need connector |
| WHO | Partial | Yes (F:) | Copy + enhance |
| FAERS | Yes | Yes (Downloads) | Need env var |

---

## Storage Requirements

| Scenario | Minimum | Recommended |
|----------|---------|-------------|
| Repo only (committed data) | 100 MB | 100 MB |
| With WHO copied | 225 MB | 225 MB |
| Full local dev (all datasets) | 70 GB | 100 GB |
| Production (API only) | 5 GB | 10 GB |

---

## Next Steps

1. **Copy WHO data** to repo (123 MB - under 50 MB per file)
2. **Add environment variable fallbacks** in code
3. **Create data loader utilities** for each dataset
4. **Document Railway deployment** - which datasets to download at build
5. **Consider data compression** for large text files
