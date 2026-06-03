# Wafer Map and DRAM Fail Bit Map Defect Classification

This repository is a prototype for defect pattern classification using
synthetic data only.

It separates two related inspection views:

- Wafer map: die-level spatial defect distribution across a wafer.
- DRAM fail bit map: bit/address-level defect distribution inside one memory
  array.

The base project is designed to run without PyTorch. Classical image features
and scikit-learn models will be used first. Optional CNN support may be added
later through `requirements-cnn.txt`, but PyTorch is not required for the base
workflow and should not be imported by base modules.

## Synthetic Data Disclosure

All planned datasets, examples, figures, models, metrics, and reports in this
project are synthetic. They are intended for software prototyping and portfolio
review only. Results must not be interpreted as real semiconductor
manufacturing performance.

## Planned Defect Classes

- random defect
- center defect
- edge defect
- ring defect
- scratch defect
- cluster defect
- row failure
- column failure
- block failure

## Reproducible PowerShell Commands

Use the virtual environment Python for project commands:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m wafer_defect_classification.cli generate --map-type wafer
.\.venv\Scripts\python.exe -m wafer_defect_classification.cli generate --map-type dram
.\.venv\Scripts\python.exe -m wafer_defect_classification.cli generate --map-type both
.\.venv\Scripts\python.exe -m wafer_defect_classification.cli train --map-type wafer
.\.venv\Scripts\python.exe -m wafer_defect_classification.cli train --map-type dram
.\.venv\Scripts\python.exe -m wafer_defect_classification.cli evaluate --map-type wafer
.\.venv\Scripts\python.exe -m wafer_defect_classification.cli evaluate --map-type dram
.\.venv\Scripts\python.exe -m pytest
```
