# Tri-CGC: Trimodal Customized Gate Control for Pesticide Toxicity Prediction

A multimodal deep learning framework for predicting acute oral, dermal, and inhalation toxicity of pesticide compounds. Tri-CGC integrates three complementary molecular representations — **molecular graph**, **SMILES sequence**, and **MACCS fingerprint** — through a Customized Gate Control (CGC) architecture with residual expert networks and multi-task learning.

## Overview

Toxicity prediction of pesticide molecules is a critical yet challenging task due to limited labeled data and complex structure-activity relationships. Tri-CGC addresses these challenges by:

- **Three complementary modalities**: molecular graphs capture topological structure, SMILES sequences capture sequential chemical grammar, and MACCS fingerprints capture domain-specific substructure patterns.
- **Customized Gate Control (CGC)**: extends the Multi-gate Mixture-of-Experts (MMoE) paradigm with shared experts per modality, task-specific experts, and dynamic gating networks.
- **Residual expert networks**: each expert uses residual connections for stable gradient flow, allowing deeper architectures even with small datasets.
- **Multi-task learning with uncertainty weighting**: jointly learns three toxicity endpoints (oral, dermal, inhalation) with learnable task weights.
- **Robustness mechanisms**: modality dropout and fingerprint soft masking for improved generalization.

## Architecture

```
┌─────────────┐   ┌──────────────┐   ┌──────────────┐
│  Molecular   │   │    SMILES    │   │    MACCS     │
│   Graph      │   │  Sequence    │   │ Fingerprint  │
│  (GATv2)    │   │  (TextCNN)   │   │   (MLP)      │
└──────┬──────┘   └──────┬───────┘   └──────┬───────┘
       │                 │                   │
       ▼                 ▼                   ▼
┌────────────────────────────────────────────────────┐
│                 CGC Expert Pool                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐ │
│  │  Graph   │ │  SMILES  │ │    FP    │ │Task-  │ │
│  │ Experts  │ │ Experts  │ │ Experts  │ │Specific│ │
│  │  (×N)    │ │  (×N)    │ │  (×N)    │ │Experts │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬───┘ │
│       └─────────────┴────────────┴──────────┘       │
│                      │                              │
│              ┌───────┴───────┐                      │
│              │ Gating Network│                      │
│              │  (×3 tasks)   │                      │
│              └───────┬───────┘                      │
└──────────────────────┼──────────────────────────────┘
                       ▼
        ┌──────────────────────────┐
        │      Task Towers         │
        │  Oral │ Dermal │ Inhal   │
        └──────────────────────────┘
```

## Project Structure

```
Tri-CGC/
├── scripts/
│   ├── Graph_encoder.py                  # GATv2 + GINE molecular graph encoder
│   ├── SMILES_encoder.py                 # SMILES tokenizer + TextCNN encoder
│   ├── FP_encoder.py                     # MACCS fingerprint MLP with residual blocks
│   ├── Multimodal_CGC_with_residual.py   # CGC expert architecture & prediction model
│   └── train_model_trimodal_with_test.py # Main training & evaluation script
├── data/
│   └── all_data.xlsx                     # Dataset (SMILES + toxicity labels)
├── model/                                # Saved model checkpoints
└── README.md
```

## Key Components

### Graph Encoder ([Graph_encoder.py](scripts/Graph_encoder.py))

- **GATv2 (dynamic attention)** with multi-head attention for atom-level message passing
- **GINE-style edge fusion**: neighbor atom features are fused with chemical bond features
- **LayerNorm + Residual connections** (Pre-Norm structure) for training stability on small datasets
- **Atom-level readout gate**: learnable soft attention pool that identifies toxicity-relevant atoms
- Enhanced atom features: Gasteiger charges, ring sizes, chirality, hybridization, atomic mass

### SMILES Encoder ([SMILES_encoder.py](scripts/SMILES_encoder.py))

- **Regex-based tokenizer** following Schwaller et al.'s approach for chemical-aware tokenization
- **TextCNN architecture**: multi-scale 1D convolutions (kernel sizes 3–9) capture local chemical n-grams
- Significantly fewer parameters than LSTM/Transformer alternatives, suited for limited data

### Fingerprint Encoder ([FP_encoder.py](scripts/FP_encoder.py))

- **MACCS keys (167-bit)** as input, providing domain-curated substructure features
- **Residual MLP blocks** with BatchNorm, ReLU, and Dropout for stable training
- Supports data augmentation via random bit masking during training

### CGC Model ([Multimodal_CGC_with_residual.py](scripts/Multimodal_CGC_with_residual.py))

- **ResidualExpert**: bottleneck MLP with residual projection for gradient flow
- **MultimodalCGC**: shared experts (per modality) + task-specific experts, all sharing a joint feature space
- **Softmax gating**: each task learns input-dependent weights over the entire expert pool
- **Dynamic modality switches**: graph, SMILES, and fingerprint modalities can be independently enabled/disabled
- **UncertaintyWeighting**: learnable log-variance parameters for multi-task loss balancing

### Training Script ([train_model_trimodal_with_test.py](scripts/train_model_trimodal_with_test.py))

- Masked BCE loss handling partial labels (valid values: 0, 1; ignored: -1)
- Class imbalance correction via per-task positive weight computation
- Modality dropout for robustness (randomly zeroes one modality's features)
- MACCS fingerprint soft masking as data augmentation
- Cosine annealing learning rate scheduling
- Early stopping on validation AUC
- Comprehensive per-task metrics: Accuracy, Precision, Recall, Specificity, MCC, F1, AUC

## Dependencies

- Python ≥ 3.8
- PyTorch ≥ 1.12
- PyTorch Geometric ≥ 2.2
- RDKit ≥ 2022.03
- pandas, numpy, scikit-learn
- openpyxl (for Excel I/O)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Tri-CGC.git
cd Tri-CGC

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install rdkit-pypi pandas numpy scikit-learn openpyxl
```

## Quick Start

### 1. Prepare Your Data

The dataset should be an Excel (`.xlsx`) or CSV file with the following columns:

| SMILES | GHS_oral | GHS_dermal | GHS_inhalation |
|--------|----------|------------|----------------|
| CC(C)Oc1ccc... | 0 | 1 | 0 |
| ... | ... | ... | ... |

- **SMILES**: canonical SMILES string of the compound
- **GHS_oral / GHS_dermal / GHS_inhalation**: binary toxicity labels (0 = non-toxic, 1 = toxic). Use -1 for unknown/missing labels.

For 10-fold cross-validation, split the data into train/validation folds and a held-out test set.

### 2. Configure & Train

Edit the `Config` class in [train_model_trimodal_with_test.py](scripts/train_model_trimodal_with_test.py#L55) to set your data paths:

```python
excel_path: str = "data/train_fold_0.xlsx"
val_path: str = "data/val_fold_0.xlsx"
test_path: str = "data/test.xlsx"
```

Then run:

```bash
cd scripts
python train_model_trimodal_with_test.py
```

### 3. Run a Single Modality

You can disable modalities via the config flags:

```python
cfg.use_graph = True
cfg.use_smiles = False
cfg.use_fp = False  # Graph-only mode
```

## Citation

If you use Tri-CGC in your research, please cite:

```bibtex
@article{tri-cgc,
  title     = {Tri-CGC: Trimodal Customized Gate Control for Pesticide Toxicity Prediction},
  author    = {},
  journal   = {},
  year      = {2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
