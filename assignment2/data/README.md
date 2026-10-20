# Preprocessed Data for Teammates

This folder contains the **357 fixed image features** used by the final Random Forest notebook. These arrays contain extracted features rather than raw or simply normalized images. Feature extraction is already complete: do not divide these arrays by 255 again. Row order matches the original image arrays, and label encoding is unchanged.

| File | Shape / purpose |
| --- | --- |
| `X_train_features.npy` | `(32000, 357)`, float32; features for all training samples |
| `y_train.npy` | `(32000,)`; corresponding training labels |
| `X_test_features.npy` | `(8000, 357)`, float32; features for all test samples |
| `y_test.npy` | `(8000,)`; corresponding test labels |
| `split_indices.npz` | `fit_idx`: 25,600 training indices; `val_idx`: 6,400 validation indices, matching the original experiment |
| `feature_names.json` | Names of all 357 feature columns, in matrix column order |
| `manifest.json` | Data provenance and SHA-256 file checksums |

Feature extraction first converts pixels to float32 and divides them by 255. It then computes RGB means (147 columns) and standard deviations (147 columns) over 49 spatial blocks of size 4×4, normalized 16-bin histograms for each color channel (48 columns), and the 0.1, 0.25, 0.5, 0.75, and 0.9 quantiles for each channel (15 columns). Feature groups appear in that order; see the JSON file for individual column names. All transformations operate independently on each image and require no parameters fitted across the dataset.

## Loading the Data

Run the following code from the `assignment2` directory. Loading the arrays requires only NumPy:

```python
from pathlib import Path
import numpy as np

data = Path("data")
X_train = np.load(data / "X_train_features.npy", allow_pickle=False)
y_train = np.load(data / "y_train.npy", allow_pickle=False)
with np.load(data / "split_indices.npz", allow_pickle=False) as split:
    fit_idx = split["fit_idx"]
    val_idx = split["val_idx"]

X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
X_val, y_val = X_train[val_idx], y_train[val_idx]
# model.fit(X_fit, y_fit)
# Compare hyperparameters using X_val and y_val.

# After selecting the configuration, refit on all X_train and y_train,
# then load the test set for final evaluation:
X_test = np.load(data / "X_test_features.npy", allow_pickle=False)
y_test = np.load(data / "y_test.npy", allow_pickle=False)
```

Random Forest can use these features directly. For models that need standardization, such as SVM, kNN, logistic regression, or MLP, fit the scaler only on `X_fit` and use it to transform the validation set. Once the configuration is selected, refit the complete pipeline, including the scaler, on all training samples before transforming the test set. Do not select hyperparameters using the test set.

**For MLP:** these arrays can be used directly as 357 input features, with any required scaling fitted as described above. Reuse the provided split to compare models on the same samples.

**For CNN:** use the original `(N, 28, 28, 3)` images in `../Assignment2Data/` relative to this folder, and preprocess them for the CNN's input requirements. The 357-dimensional features do not preserve the original image layout and cannot reconstruct the original images.

## Regenerating the Files

Run `python export_preprocessed_data.py` from `assignment2`. This requires the original `Assignment2Data/` folder, the final notebook, and the `manifest.json` and `split_indices.npz` files in `optimization_20261018/`. The script extracts the notebook's `features()` function to use the same implementation. It does not train a model or modify the original data. Use the environment specified in the repository's `requirements.txt` to match the NumPy version.

This folder can be shared with teammates independently. It includes every original test sample, without filtering. The test set was already evaluated in earlier experiments; the limitations documented in the original report still apply.
