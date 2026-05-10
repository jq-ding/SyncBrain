# SyncBrain

> **SyncBrain: Exploring Brain Functional Dynamics Through Neural Oscillatory Synchronization**
> Jiaqi Ding, Tingting Dan, Zhixuan Zhou, Guorong Wu
> **AAAI 2026 (Oral)**

## Overview

`SyncBrain` is a brain-inspired machine learning model that captures functional dynamics in the brain through the lens of **neural oscillatory synchronization**. Rather than treating brain activity as static features or as outputs of a black-box neural network, SyncBrain models cross-region communication as a system of coupled oscillators, allowing it to learn interpretable, biologically grounded representations of brain dynamics from fMRI.

This framework operates on functional connectivity and BOLD time series, and supports a wide range of downstream tasks — disease classification, cognitive state decoding, and biomarker discovery — across multiple large-scale neuroimaging cohorts.

## Repository Structure

```
SyncBrain/
├── src/
│   ├── data/
│   │   ├── create_dataset.py
│   │   └── dataset.py            # Loaders for HCP-A, HCP-YA, HCP-WM,
│   │                             #   ADNI, OASIS, PPMI, NIFD
│   ├── modules/
│   │   ├── GST.py                # Graph Scattering Transform
│   │   └── SyncBrain_solver.py   # Oscillatory synchronization solver
│   ├── SyncBrain.py              # Main SyncBrain model
│   └── utils.py
└── train_and_eval.py             # Training and evaluation entry point
```

## Datasets

SyncBrain has been validated on the following large-scale neuroimaging cohorts:

- **HCP-Aging (HCP-A)** — lifespan brain organization
- **HCP-Young Adult (HCP-YA)** — resting-state and task fMRI
- **HCP Working Memory (HCP-WM)** — cognitive state decoding
- **ADNI** — Alzheimer's disease progression
- **OASIS** — aging and dementia
- **PPMI** — Parkinson's disease
- **NIFD** — frontotemporal dementia

Please follow each dataset's official access process to obtain the raw data, then use `src/data/create_dataset.py` to preprocess into the format expected by the model.


## Usage

```bash
python train_and_eval.py \
    --dataset HCP-WM \
    --batch_size 32 \
    --epochs 200 \
    --gpu 0
```

See `train_and_eval.py` for the full list of arguments.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{ding2026syncbrain,
  title     = {SyncBrain: Exploring Brain Functional Dynamics Through Neural Oscillatory Synchronization},
  author    = {Ding, Jiaqi and Dan, Tingting and Zhou, Zhixuan and Wu, Guorong},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {3},
  pages     = {1774--1782},
  year      = {2026}
}
```


## Contact

Jiaqi Ding — `jiaqid@cs.unc.edu`
[Personal website](https://jq-ding.github.io/) · [Google Scholar](https://scholar.google.com/citations?hl=en&user=5h5qru8AAAAJ)
