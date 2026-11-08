# Graph Neural Networks for Molecular Property Prediction

**This repository contains the official implementation of the paper
"Graph Neural Networks for Molecular Property Prediction" (NeurIPS 2026).**

The code provides the edge-aware message-passing model, the training loop and
the scripts used to reproduce the MoleculeNet results in the paper.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python -m gnn.train --dataset tox21 --epochs 100
```

## Evaluate

```bash
python -m gnn.evaluate --checkpoint checkpoints/tox21.pt --dataset tox21
```

## Citation

```bibtex
@inproceedings{cook2026gnn,
  title={Graph Neural Networks for Molecular Property Prediction},
  author={Cook, Ronald and Hoffmann, Lena and Chen, Wei},
  booktitle={Advances in Neural Information Processing Systems 39},
  year={2026}
}
```
