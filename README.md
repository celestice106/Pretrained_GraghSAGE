# GraphSAGE Structural Pre-training for Memory R1

Pre-training pipeline for generating topology-aware structural embeddings using cognitive motif-augmented synthesis.

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic dataset
python scripts/01_generate_dataset.py

# Train GraphSAGE
python scripts/02_train_graphsage.py

# Evaluate embeddings
python scripts/03_evaluate_model.py

## Project Structure

- `config/`: Hyperparameters and settings
- `data/`: Graph generation and feature extraction
- `models/`: GraphSAGE architecture
- `training/`: Training loop and optimization
- `evaluation/`: Metrics and visualization
- `outputs/`: Generated data, checkpoints, logs

## Sprint 3 Objectives

✅ Cognitive motif library (temporal chains, causal structures, entity stars)  
✅ Synthetic graph generation (BA, WS, ER, SBM base models)  
✅ Structural feature extraction (Degree + RWPE)  
✅ GraphSAGE pre-training with link prediction  
✅ Clustering evaluation (Silhouette, Davies-Bouldin)  
✅ Zero-shot transfer validation