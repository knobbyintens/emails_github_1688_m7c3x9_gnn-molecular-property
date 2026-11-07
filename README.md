# Interpretable Tabular Transformers for Clinical Risk Prediction

**Status: code released.**

This repository contains the implementation for the paper:

> **Interpretable Tabular Transformers for Clinical Risk Prediction**
> Maria Hernandez, Priya Raman, Sofia Alvarez
> MLHC 2023

The released implementation covers the tabular transformer, the attention
path explanation module and the clinical preprocessing pipeline used in
the experiments.

## Repository layout

- `src/` - model, data pipeline, training and evaluation code
- `configs/` - experiment configuration files
- `scripts/` - launch scripts

## Getting started

```bash
pip install -r requirements.txt
python -m src.train --config configs/default.yaml
```

## Citation

If you use this code, please cite the paper above.
