# Reproducibility Guide

> This document is filled in progressively. Currently covers environment setup
> and dataset acquisition only (Phase 1).

## 1. Environment Setup (Windows / cmd)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

Python version used in development: **3.13** (confirm exact patch version via `python --version`)

## 2. Dataset Acquisition

1. Register at https://xview2.org
2. Download the **Tier1 training set** (do not download Tier3/holdout yet)
3. Extract into `data/raw/xbd/` such that the structure is:
data/raw/xbd/train/images/
data/raw/xbd/train/labels/

4. Do **not** commit this data to git — excluded via `.gitignore`, and its
   CC BY-NC-SA 3.0 license does not permit redistribution here.

*(Preprocessing, training, evaluation, and inference sections added in their
respective phases.)*