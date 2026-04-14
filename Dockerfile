FROM python:3.12-slim

WORKDIR /app

# Install build dependencies (gcc/g++ needed for shap, ruptures)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cache bust: 2026-04-13-kg-v2
RUN echo "Build timestamp: $(date)"

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL application code
COPY . .

# Download HPO data files
RUN mkdir -p data/hpo && \
    curl -L -o data/hpo/genes_to_disease.txt https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2026-02-16/genes_to_disease.txt && \
    curl -L -o data/hpo/genes_to_phenotype.txt https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2026-02-16/genes_to_phenotype.txt && \
    curl -L -o data/hpo/phenotype_to_genes.txt https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2026-02-16/phenotype_to_genes.txt && \
    echo "HPO files downloaded:" && ls -la data/hpo/

# Download ClinVar variant summary (genomic variants database)
RUN mkdir -p data/clinvar && \
    curl -L -o data/clinvar/variant_summary.txt.gz https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz && \
    echo "ClinVar downloaded:" && ls -la data/clinvar/

# Note: PharmGKB data requires account - gracefully handled as optional

# Download Hetionet knowledge graph (47K nodes, 2.25M edges)
RUN mkdir -p data/external/hetionet/hetnet/tsv && \
    curl -L -o data/external/hetionet/hetnet/tsv/hetionet-v1.0-nodes.tsv \
        https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-nodes.tsv && \
    curl -L -o data/external/hetionet/hetnet/tsv/hetionet-v1.0-edges.sif.gz \
        https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz && \
    echo "Hetionet downloaded:" && ls -la data/external/hetionet/hetnet/tsv/

# Download PrimeKG knowledge graph (129K nodes, 8.1M edges, ~937MB)
RUN mkdir -p data/external/PrimeKG && \
    curl -L -o data/external/PrimeKG/kg.csv \
        https://dataverse.harvard.edu/api/access/datafile/6180620 && \
    echo "PrimeKG downloaded:" && ls -lh data/external/PrimeKG/

# Verify files exist
RUN ls -la app/core/endpoints/ && ls -la app/core/pathways/

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
