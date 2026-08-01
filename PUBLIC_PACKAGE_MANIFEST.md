# Public Package Manifest

## Release

- Project: `wafer-defect-classification`
- Release version: `0.1.1`
- Release date: `2026-07-31`
- Package purpose: public portfolio and reproducible prototype review
- Data scope: synthetic wafer maps and synthetic DRAM fail bit maps only

## Completed Release Work

1. Removed local and private development artifacts from the public package:
   `.venv/`, `.tmp/`, `.git/`, `_private_notes/`, and `__pycache__/`.
2. Implemented the `predict` inference path:
   model loading, `.npy`/`.npz`/`.csv` input validation, feature extraction,
   batch classification, confidence/probability output, and JSON/CSV export.
3. Locked the archived Python 3.12.8 environment:
   direct pins in `requirements.txt`, full environment freeze in
   `requirements-lock.txt`, and model-loading guidance in
   `MODEL_COMPATIBILITY.md`.

No new model family, dataset, noise experiment, or defect-pattern test scenario
was added in this release.

## Verification

- Existing automated test suite: `27 passed`
- New automated test cases added: `0`
- CLI commands available: `generate`, `train`, `predict`
- Manual prediction smoke checks: single-map JSON and CSV output completed
- Source syntax check: passed

The smoke check in the packaging environment does not replace the locked model
environment. Use Python 3.12 and `requirements-lock.txt` for the committed
`.joblib` files.

## Included Scope

- Source code and CLI
- Configuration files
- Synthetic datasets and metadata
- Train/validation/test split records
- Four trained model artifacts
- Metrics, figures, and technical reports
- Existing test suite
- Dependency locks and model compatibility documentation
- File hash list

## Excluded Scope

- Virtual environments
- Git internals and local history
- Temporary test directories
- Private notes and Word exports
- Python bytecode and caches
- Optional CNN environment

## Package Statistics Before Release Metadata

- Included files: `61`
- Included size: `4.00 MB`
