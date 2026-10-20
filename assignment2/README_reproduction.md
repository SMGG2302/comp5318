# 最终 Random Forest Notebook 的运行与复现

最终文件：`random_forest_optimization.ipynb`。推荐 Python 3.12。

## 最少需要什么

**从头训练最终模型**：Notebook、下列四个 `.npy` 数据文件，以及 `requirements.txt` 定义的软件环境。Notebook 已包含特征提取和训练代码，不需要额外的 `.py` 脚本或预训练模型。

| 文件 | 作用 |
| --- | --- |
| `Assignment2Data/X_train.npy` | 32,000 张训练图像，用于最终模型训练 |
| `Assignment2Data/y_train.npy` | 训练图像对应的类别标签 |
| `Assignment2Data/X_test.npy` | 8,000 张测试图像，用于训练完成后的评估 |
| `Assignment2Data/y_test.npy` | 测试集真实标签，用于计算指标和混淆矩阵 |
| `requirements.txt` | 固定主要 Python 库版本，安装训练与 Notebook 执行依赖 |

**直接 Run All，并展示原来的调参过程和最终结果**：还需要以下轻量文件。本仓库同时提供它们，数据与这些文件合计约 90 MiB。

| `optimization_20261018/` 下的文件 | 作用 |
| --- | --- |
| `manifest.json` | 原始环境、随机种子、数据 SHA-256 和实验限制；也是已有实验的初始化标记 |
| `plan.json` | 分阶段调参计划，供 Notebook 展示 |
| `split_indices.npz` | 保留原始训练/验证划分；缺失时虽然可重新生成，但保留可明确复用同一划分 |
| `selection_frozen.json` | 已冻结的最终选择；默认运行时跳过重新搜索 |
| `experiment_history.csv` | 优化实验各次参数和指标的汇总 |
| `optimization_history.png` | 优化过程图 |
| `seed_comparison.json` | 候选配置在不同随机种子上的比较 |
| `final/config.json` | 最终配置；检查缓存是否与 Notebook 参数一致 |
| `final/metrics.json` | 原始最终测试指标 |
| `final/classification_report.csv` | 各类别 precision、recall、F1 等指标 |
| `final/confusion_matrix.npy` | 绘制原始最终混淆矩阵 |

## 安装与运行

```powershell
git clone --branch dev https://github.com/SMGG2302/comp5318.git
cd comp5318/assignment2
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

用 VS Code 的 Jupyter 扩展打开 Notebook，选择 `.venv` 环境作为内核。内核工作目录必须是 `assignment2`，其中能直接看到 `Assignment2Data` 文件夹。也可使用自己的 Jupyter 前端。

### 1. 查看原始结果

保留最后一个代码单元的 `RETRAIN_FINAL = False`，点击 Run All。此模式读取已保存的模型选择和测试结果，**不会重新训练**。运行生成的 `events.jsonl` 和 `training_labels.npy` 无需事先提供。

### 2. 真正重新训练最终模型

先运行包含导入、路径与函数定义的 setup 代码单元，再把最后一个代码单元改为：

```python
RETRAIN_FINAL = True
```

运行最后一个代码单元即可，无需运行调参单元。它使用 400 棵树、`max_features=0.3`、`max_depth=None`、`min_samples_leaf=1`、随机种子 42 和 4 个并行任务，在全部训练数据上拟合，再评估测试集。耗时取决于电脑，原实验训练约 5 分钟。

新输出写入独立的 `reproduction_final_日期_时间/` 文件夹，包含配置、模型、每个测试样本的预测和概率、指标、分类报告及混淆矩阵，不覆盖原始实验。原实验 accuracy 为 0.860625，macro-F1 为 0.8562440574；运行耗时不应期待相同。

## 包含范围

此最小包支持重放原 Notebook 展示和独立重训最终配置。原来的全部搜索模型、逐次预测、脚本、报告 DOCX/PDF 都不是上述运行的依赖，因此没有作为依赖重复加入。原有完整实验档案仍保留在本地。

本包不等于完整的逐次搜索档案，也不用于在空目录中重建全部历史调参流程；不要删除 `selection_frozen.json` 后把 Run All 当作完整搜索复现入口。原测试集在早期实验中已有评估，相关限制记录在 `manifest.json` 中。调参依据是验证集，不应根据重新运行后的测试指标继续挑选参数。
