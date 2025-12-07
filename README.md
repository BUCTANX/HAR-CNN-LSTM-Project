# 基于 CNN-LSTM 的智能手机人体活动识别 (HAR)

本项目实现了一个基于深度学习的人体活动识别系统。通过融合智能手机加速度计和陀螺仪的多模态传感器数据，利用 CNN 提取空间特征，结合 LSTM 处理时序依赖，实现了对 6 种日常活动的准确分类。

## 📁 项目简介
* **任务目标**：识别 6 种人体活动（走路、上楼、下楼、坐、站、躺）。
* **核心模型**：1D-CNN + LSTM 混合神经网络。
* **数据集**：UCI HAR Dataset (Human Activity Recognition Using Smartphones)。

## 🛠️ 环境依赖
* Python 3.x
* PyTorch
* NumPy
* Pandas

## 🚀 如何运行
1. **下载数据集**：
   从 [UCI 官网](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) 下载数据集并解压。
2. **配置路径**：
   打开 `data_loader.py`，将 `DATA_PATH` 修改为你本地的数据集路径。
3. **开始训练**：
   ```bash
   python train.py

  📊 实验结果
在测试集上经过 20 轮训练后：

测试集准确率 (Test Accuracy): 约 91.5%
模型架构: 输入层(9通道) -> CNN特征提取 -> LSTM时序建模 -> 全连接分类
📝 文件说明
data_loader.py: 数据预处理与加载模块。
model.py: 定义 CNN-LSTM 网络架构。
train.py: 模型训练与评估脚本。
