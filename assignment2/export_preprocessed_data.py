"""Export the final notebook's exact fixed features for teammates."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    notebook = ROOT / "random_forest_optimization.ipynb"
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    definitions = [
        node
        for cell in cells if cell["cell_type"] == "code"
        for node in ast.parse("".join(cell["source"])).body
        if isinstance(node, ast.FunctionDef) and node.name == "features"
    ]
    assert len(definitions) == 1, "Expected one features function in final notebook"
    namespace = {"np": np}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(notebook), "exec"), namespace)
    features = namespace["features"]
    OUT.mkdir(exist_ok=True)
    manifest = {"representation": "colour_texture", "numpy": np.__version__,
                "source_notebook_sha256": sha256(notebook),
                "row_order": "Unchanged from original arrays; labels unchanged.",
                "preprocessing": "Fixed per-image descriptors; no learned preprocessing parameters.",
                "files": {}, "source_data_sha256": {}}
    original_manifest = json.loads((ROOT/"optimization_20261018/manifest.json").read_text())
    for split, count in [("train", 32000), ("test", 8000)]:
        image_path = ROOT / "Assignment2Data" / f"X_{split}.npy"
        label_path = ROOT / "Assignment2Data" / f"y_{split}.npy"
        for path in [image_path, label_path]:
            digest = sha256(path)
            assert digest == original_manifest["data_sha256"][path.name]
            manifest["source_data_sha256"][path.name] = digest
        images = np.load(image_path, allow_pickle=False)
        labels = np.load(label_path, allow_pickle=False).ravel()
        result = features(images, "colour_texture")
        assert result.shape == (count, 357) and labels.shape == (count,)
        assert result.dtype == np.float32 and np.isfinite(result).all()
        np.save(OUT / f"X_{split}_features.npy", result, allow_pickle=False)
        np.save(OUT / f"y_{split}.npy", labels, allow_pickle=False)
        assert np.array_equal(np.load(OUT/f"X_{split}_features.npy"), result)
        assert np.array_equal(np.load(OUT/f"y_{split}.npy"), labels)
        print(f"{split}: {result.shape}, {result.dtype}; labels aligned", flush=True)
    source_split = ROOT / "optimization_20261018/split_indices.npz"
    shutil.copy2(source_split, OUT / "split_indices.npz")
    with np.load(OUT/"split_indices.npz") as split:
        fit, val = split["fit_idx"], split["val_idx"]
        assert len(fit) == 25600 and len(val) == 6400
        assert np.array_equal(np.sort(np.concatenate([fit, val])), np.arange(32000))
    names = [
        f"block_{row}_{col}_{channel}_{stat}"
        for stat in ["mean", "std"]
        for row in range(7) for col in range(7) for channel in ["R", "G", "B"]
    ]
    names += [f"{channel}_hist_bin_{b}" for channel in ["R", "G", "B"] for b in range(16)]
    names += [f"{channel}_quantile_{q}" for q in [0.1, 0.25, 0.5, 0.75, 0.9] for channel in ["R", "G", "B"]]
    assert len(names) == 357
    (OUT/"feature_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    for name in ["X_train_features.npy", "X_test_features.npy", "y_train.npy", "y_test.npy",
                 "split_indices.npz", "feature_names.json"]:
        path = OUT/name
        manifest["files"][name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Verified exact notebook feature function, original data hashes, saved arrays and split coverage.")


if __name__ == "__main__":
    main()
