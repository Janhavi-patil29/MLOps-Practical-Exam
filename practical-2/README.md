# MLOps Breast Cancer Classifier

Classifies tumours as benign or malignant from 9 cell measurements. A sample ML project used to demonstrate Git-based version control
workflows in an MLOps context (MLOps Lab Experiment 2).

**Dataset:** Wisconsin breast cancer CSV (699 samples, 9 features, benign vs malignant)
Source: <https://github.com/jeffheaton/aifh/blob/master/vol1/python-examples/datasets/breast-cancer-wisconsin.csv>

The CSV is downloaded automatically into `data/` on the first run (needs internet). `data/*.csv` is git-ignored, so the file itself is never committed. If the download fails, save the file manually as `data/breast-cancer-wisconsin.csv` (raw link is in `src/train.py`). Missing `?` values in `bare_nucleoli` are imputed with the median.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate      # Git Bash on Windows (PowerShell: .\.venv\Scripts\Activate.ps1)
pip install -r requirements.txt
python src/train.py
```

## Project structure

```
mlops-breast-cancer-classifier/
├── src/
│   ├── __init__.py
│   └── train.py
├── data/
├── models/
├── notebooks/
├── docs/
│   └── VERSION_CONTROL_WORKFLOW.md
├── README.md
├── requirements.txt
└── .gitignore
```
