# KNN 训练前端

本项目是北京信息科技大学作业《人工智能实验—汉字手写字体识别》的个人实现，并增加了一定注释。

## 使用方法

1. 克隆项目：`https://github.com/ProjektMing/KNN-Trains.git`
2. 进入项目目录：`cd KNN-Trains`
3. 创建虚拟环境：建议使用 conda（可选）
4. 安装依赖库：`pip install -r requirements.txt`
5. 运行程序：`python -u main.py`

或者直接运行 `run.ps1` 文件。

## 运行说明

程序会将图片转换为灰度图像，并进行二值化处理。然后，程序会使用 KNN 算法对图像进行分类，并输出识别结果。

## 数据集说明

数据集不提供，使用的是老师提供的 中国二字 数据集。

## 数据集结构

数据集的结构如下：

```bash
└─ data_set
   └─ label...
      └─ *.bmp
```

## 软件包说明

- `numpy`：用于数值计算和数组操作。
- `matplotlib`：用于绘图和可视化。
- `sklearn`：用于机器学习和数据挖掘。
- `seaborn`：用于数据可视化。
