# Model Compatibility

The committed `.joblib` model artifacts were produced in the archived project
environment below:

- Python `3.12.8`
- NumPy `2.5.1`
- Pandas `3.0.5`
- SciPy `1.18.0`
- scikit-learn `1.9.0`
- scikit-image `0.26.0`
- Pillow `12.3.0`
- Matplotlib `3.11.1`
- tqdm `4.70.0`
- PyYAML `6.0.3`
- Joblib `1.5.3`
- Rich `15.0.0`
- Pytest `9.1.1`

Use Python 3.12 and install `requirements-lock.txt` before loading the
committed models. `requirements.txt` is a shorter list of direct dependencies. Scikit-learn and Joblib do not guarantee reliable model
artifact compatibility across arbitrary library versions.

If a different environment must be used, regenerate the synthetic datasets and
retrain the models in that environment instead of assuming the committed
`.joblib` files are portable.
