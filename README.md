# Tri-CGC: Trimodal Customized Gate Control for Pesticide Toxicity Prediction

A multimodal deep learning framework for predicting acute oral, dermal, and inhalation toxicity of pesticide compounds. Tri-CGC integrates three complementary molecular representations — **molecular graph**, **SMILES sequence**, and **MACCS fingerprint** — through a Customized Gate Control (CGC) architecture with residual expert networks and multi-task learning.

## Overview

Toxicity prediction of pesticide molecules is a critical yet challenging task due to limited labeled data and complex structure-activity relationships. Tri-CGC addresses these challenges by:

- **Three complementary modalities**: molecular graphs capture topological structure, SMILES sequences capture sequential chemical grammar, and MACCS fingerprints capture domain-specific substructure patterns.
- **Customized Gate Control (CGC)**: extends the Multi-gate Mixture-of-Experts (MMoE) paradigm with shared experts per modality, task-specific experts, and dynamic gating networks.
- **Residual expert networks**: each expert uses residual connections for stable gradient flow, allowing deeper architectures even with small datasets.
- **Multi-task learning with uncertainty weighting**: jointly learns three toxicity endpoints (oral, dermal, inhalation) with learnable task weights.
- **Robustness mechanisms**: modality dropout and fingerprint soft masking for improved generalization.


## Dependencies

- Python ≥ 3.8
- PyTorch ≥ 1.12
- PyTorch Geometric ≥ 2.2
- RDKit ≥ 2025
- pandas, numpy, scikit-learn
- openpyxl 

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
| CC(C)Oc1ccc... | 0 | 1 | -1 |
| ... | ... | ... | ... |

- **SMILES**: canonical SMILES string of the compound
- **GHS_oral / GHS_dermal / GHS_inhalation**: binary toxicity labels (0 = non-toxic, 1 = toxic). Use -1 for unknown/missing labels.

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
