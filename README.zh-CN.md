[English](./README.md) | [Español](./README.es.md) | 中文

[![PyPI](https://img.shields.io/pypi/v/HSPiPy.svg)](https://pypi.org/project/HSPiPy/) ![version](https://img.shields.io/badge/version-1.1.8-orange.svg)

> [!TIP]
> 寻找无需编程的版本？试试 [**HSP Desktop**](https://g-npd.com/zh/hsp)

# HSPiPy

#### Python 中的汉森溶解度参数（Hansen Solubility Parameters）

### 简介
---------------

HSPiPy 是一个用于计算和可视化汉森溶解度参数（HSP）的 Python 库。它提供了适用于机器学习的估计器、便捷的数据导入工具，以及用于分析材料科学、聚合物和涂料领域中溶剂相容性的绘图工具。

### 功能特性
---------------

* 支持从多种格式读取溶剂数据（CSV、HSD、HSDX）
* 使用鲁棒优化方法计算汉森溶解度参数
* 支持单球或多球（最多 2 个）溶解度球模型
* 生成溶解度球的 2D 和 3D 可视化图
* 与 scikit-learn 完全兼容，可无缝集成到机器学习工作流中


### 文档
---------------

- 类参考：
    - [HSP（高级辅助类）](https://github.com/Gnpd/HSPiPy/blob/main/docs/HSP.md)
    - [HSPEstimator（scikit-learn 兼容估计器）](https://github.com/Gnpd/HSPiPy/blob/main/docs/HSPEstimator.md)


### 安装
---------------

使用 pip 安装 **HSPiPy**：

```
pip install HSPiPy
```

#### 依赖项

* numpy
* pandas
* matplotlib
* scipy
* scikit-learn

### 使用方法
--------

#### 读取 HSP 数据

从文件（CSV、HSD 或 HSDX）读取 HSP 数据，创建 `HSP` 类的实例并使用 `read` 方法：
```python
from hspipy import HSP

hsp = HSP()
hsp.read('your_hsp_file.csv')
```


#### 计算 HSP

使用 `get` 方法从数据中计算汉森溶解度参数：
```python
# 单球模型（默认）
result = hsp.get(inside_limit=1)

# 双球模型
result = hsp.get(inside_limit=1, n_spheres=2)
```

`get()` 返回一个 `HSPResult` 对象。在 Jupyter Notebook 中会自动渲染为格式化表格。在脚本中使用 `print(result)` 或直接访问属性：

```python
print(result.hsp)             # 拟合的 D、P、H 球心坐标
print(result.radius)          # 球体半径
print(result.accuracy)        # 分类精度
print(result.datafit)         # DATAFIT 值
print(result.n_solvents_in)   # 良溶剂数量
print(result.n_solvents_out)  # 劣溶剂数量
print(result.n_total)         # 溶剂总数
print(result.n_wrong_in)      # 误判为球内的劣溶剂（假阳性）
print(result.n_wrong_out)     # 误判为球外的良溶剂（假阴性）
```

`inside_limit` 参数定义将溶剂判定为"球内"的分数阈值（默认：`inside_limit=1`）。

#### 可视化 HSP

使用 `plot_3d` 和 `plot_2d` 方法分别生成 3D 和 2D 可视化图：
```python
# 分别生成图表
hsp.plot_3d()
hsp.plot_2d()

# 一次生成两张图
hsp.plots()
```
![3dHSP](https://github.com/Gnpd/HSPiPy/blob/main/3dPlot.png)
![2dHSP](https://github.com/Gnpd/HSPiPy/blob/main/2dPlot.png)

### `HSP` 类方法
| 方法 | 描述 |
|------|:----:|
| read(path) | 从 CSV、HSD 或 HSDX 文件读取溶剂数据 |
| get(inside_limit=1, n_spheres=1) | 拟合 HSP 球体并返回 `HSPResult` 对象 |
| plot_3d() | 绘制 3D HSP 图，返回图形对象 |
| plot_2d() | 绘制 2D HSP 图，返回图形对象 |
| plots() | 同时生成 2D 和 3D 图，返回 `(fig_3d, fig_2d)` |

`get()` 返回 `HSPResult` 对象，其属性也会同步设置在 `HSP` 实例上：

| 属性 | 描述 |
|------|------|
| `result.hsp` | Numpy 数组 — 拟合的 HSP 坐标。单球时形状为 `(3,)`（D、P、H），多球时为 `(n_spheres, 3)` |
| `result.radius` | 浮点数或数组 — 溶解度球的半径 |
| `result.error` | 浮点数 — 优化目标函数值（越低越好） |
| `result.accuracy` | 浮点数 — 拟合模型在数据集上的分类精度 |
| `result.datafit` | 浮点数 — DATAFIT 值（几何平均适配度；1.0 = 完美分类） |
| `result.n_solvents_in` | 整数 — 良溶剂数量（`0 < score <= inside_limit`） |
| `result.n_solvents_out` | 整数 — 劣溶剂数量（`score == 0` 或 `score > inside_limit`） |
| `result.n_total` | 整数 — 溶剂总数 |
| `result.n_wrong_in` | 整数 — 被预测为球内的劣溶剂（假阳性） |
| `result.n_wrong_out` | 整数 — 被预测为球外的良溶剂（假阴性） |
| `hsp.d`, `hsp.p`, `hsp.h` | 浮点数 — 各 HSP 分量（δD、δP、δH），仅适用于单球；多球时为 `None` |
| `hsp.inside` | 列表 — 被分类为溶解度球内的溶剂及其 HSP 值和分数 |
| `hsp.outside` | 列表 — 被分类为溶解度球外的溶剂及其 HSP 值和分数 |
| `hsp.grid` | Pandas DataFrame — 完整输入数据集，标准化列名为：`Solvent`、`D`、`P`、`H`、`Score` |


### 使用 scikit-learn 风格估计器
```python
import numpy as np
from hspipy import HSPEstimator

# 示例数据集（D、P、H 值及评分）
X = np.array([
    [16.0, 8.0, 5.0],
    [18.0, 7.5, 9.0],
    [20.0, 10.0, 12.0],
])
y = np.array([1, 1, 0])  # 球内/球外标签或分数

est = HSPEstimator(n_spheres=1)
est.fit(X, y)

print("拟合 HSP:", est.hsp_)
print("精度:", est.score(X, y))
```

### 贡献
----------------

欢迎贡献！如有任何建议、功能请求或错误报告，请在 [GitHub 仓库](https://github.com/Gnpd/HSPiPy/issues) 提交 Issue。

### 支持本项目
------------------------
如果 HSPiPy 对您的工作有所帮助，以下是几种支持其持续开发的方式。

#### 🖥️ HSP Desktop（无代码版本）

HSP Desktop 是基于 HSPiPy 引擎构建的桌面应用程序，提供快速直观的图形界面，用于汉森溶解度参数分析。

适合以下用户：

- 偏好图形界面而非 Python 脚本
- 需要快速、一键式 HSP 计算工作流
- 希望使用高级可视化和溶剂探索工具

购买 HSP Desktop 有助于资助 HSPiPy 的持续开发，包括新功能、数据库、文档和维护工作。

👉 https://g-npd.com/zh/hsp

#### ❤️ GitHub Sponsors

您也可以通过 GitHub Sponsors 直接支持本项目：

👉 https://github.com/sponsors/Gnpd

每一份贡献都有助于让 HSPiPy 保持免费、开源并持续维护。

### 许可证
-----------

本库使用 MIT 许可证。详情请参阅 [LICENSE](https://github.com/Gnpd/HSPiPy/blob/main/LICENSE) 文件。

### 致谢
----------------

HSPiPy 的灵感来源于广为人知的 HSP 软件套件 [Hansen Solubility Parameters in Practice (HSPiP)](https://www.hansen-solubility.com/HSPiP/) 以及 HSP 社区。
