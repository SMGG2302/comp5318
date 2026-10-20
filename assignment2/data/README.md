# 给组员使用的预处理数据

这里保存最终随机森林 Notebook 使用的 **357 维固定图像特征**，不是原始图像，也不是仅归一化的图像。无需再执行特征提取或再除以 255。每一行仍对应原始数据中同一位置的图像，标签编码没有改变。

| 文件 | 形状 / 作用 |
| --- | --- |
| `X_train_features.npy` | `(32000, 357)`，float32，全部训练样本的特征 |
| `y_train.npy` | `(32000,)`，对应训练标签 |
| `X_test_features.npy` | `(8000, 357)`，float32，测试样本的特征 |
| `y_test.npy` | `(8000,)`，对应测试标签 |
| `split_indices.npz` | `fit_idx`：25,600 个训练索引；`val_idx`：6,400 个验证索引，与原实验一致 |
| `feature_names.json` | 357 列的名称，顺序与矩阵一致 |
| `manifest.json` | 来源及文件 SHA-256 校验值 |

特征计算：像素转 float32 后除以 255；49 个 4×4 图像块的 RGB 均值（147 列）、标准差（147 列）；每通道 16 档归一化直方图（48 列）；0.1、0.25、0.5、0.75、0.9 分位数（15 列）。特征按上述顺序排列，具体列名见 JSON。这些转换按单张图片计算，不需要在整个数据集上拟合参数。

## 直接加载

从 `assignment2` 目录运行以下代码，只需安装 NumPy：

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
# 用 X_val 和 y_val 比较参数。

# 配置确定后，在全部 X_train、y_train 上重新训练，再加载测试集评估：
X_test = np.load(data / "X_test_features.npy", allow_pickle=False)
y_test = np.load(data / "y_test.npy", allow_pickle=False)
```

随机森林可以直接使用这些特征。SVM、kNN、逻辑回归等模型若需要标准化，应仅在 `X_fit` 上拟合 scaler，再转换验证集；确定配置后，在全部训练集上重新拟合包含 scaler 的完整 pipeline，再转换测试集。不要用测试集选参数。

**如果组员负责 CNN**：应使用 `../Assignment2Data/` 中保留空间结构的原始 `(N, 28, 28, 3)` 图像，按 CNN 的输入要求处理。这里的 357 维特征不能代替 CNN 图像输入，也不能还原原图。

## 重新生成

在 `assignment2` 下运行 `python export_preprocessed_data.py`。需要原始 `Assignment2Data/`、最终 Notebook，以及 `optimization_20261018/` 下的 `manifest.json` 和 `split_indices.npz`。脚本直接提取 Notebook 中的 `features()` 函数，确保使用同一实现；不会训练模型或修改原始数据。使用仓库 `requirements.txt` 中的环境可保持 NumPy 版本一致。

本文件夹可单独分享给组员。它包含原来的全部测试样本，没有筛选样本。原测试集在早期实验中已有评估，实验限制沿用原报告。
