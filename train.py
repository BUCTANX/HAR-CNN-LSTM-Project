import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# 导入我们之前写的两个文件
from data_loader import load_data
from model import HARModel

# ---------------------------------------------------------
# 1. 超参数设置 (Hyperparameters)
# 这些数字就像是机器的控制旋钮，可以调整
# ---------------------------------------------------------
BATCH_SIZE = 64  # 一次训练抓取多少样本
LEARNING_RATE = 0.001  # 学习率：每次参数调整的幅度（太大容易乱，太小这就学得慢）
NUM_EPOCHS = 20  # 训练轮数：把所有数据看几遍？
HIDDEN_DIM = 128  # LSTM 记忆容量
LAYER_DIM = 1  # LSTM 层数
OUTPUT_DIM = 6  # 输出类别数 (固定为6)

# 检查是否有 GPU，如果有就用 GPU 跑，没有就用 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {DEVICE}")


def train():
    # ---------------------------------------------------------
    # 2. 准备数据
    # ---------------------------------------------------------
    print("\n[Step 1] 正在加载数据...")
    # 调用 data_loader.py 里的函数
    X_train, y_train = load_data('train')
    X_test, y_test = load_data('test')

    # 将 numpy 数组转换为 PyTorch 的 Tensor (张量)
    # 这一步是必须的，因为 PyTorch 只能处理 Tensor
    train_data = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    test_data = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )

    # 创建 DataLoader
    # 它帮我们自动把数据切成一个个 Batch，还能打乱顺序 (shuffle=True)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)

    # ---------------------------------------------------------
    # 3. 初始化模型
    # ---------------------------------------------------------
    print("\n[Step 2] 正在初始化模型...")
    # 这里的 input_dim=9 对应数据的 9 个通道
    model = HARModel(input_dim=9, hidden_dim=HIDDEN_DIM, layer_dim=LAYER_DIM, output_dim=OUTPUT_DIM)

    # 把模型搬到 GPU (如果可用)
    model.to(DEVICE)

    # 定义损失函数：交叉熵损失 (CrossEntropyLoss) 是分类任务的标准配置
    criterion = nn.CrossEntropyLoss()

    # 定义优化器：Adam 是目前最流行的优化算法，它能自动调整学习速度
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---------------------------------------------------------
    # 4. 开始训练循环
    # ---------------------------------------------------------
    print(f"\n[Step 3] 开始训练 (共 {NUM_EPOCHS} 轮)...")

    for epoch in range(NUM_EPOCHS):
        # --- 训练阶段 ---
        model.train()  # 告诉模型：现在是学习模式
        train_loss = 0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            # 把数据搬到设备上
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # A. 梯度清零 (把上一次算的梯度擦掉，避免累积)
            optimizer.zero_grad()

            # B. 前向传播 (模型猜结果)
            outputs = model(inputs)

            # C. 计算损失 (猜得有多离谱)
            loss = criterion(outputs, labels)

            # D. 反向传播 (算出参数该怎么改)
            loss.backward()

            # E. 更新参数 (真正的改参数)
            optimizer.step()

            # --- 记录一下数据用于打印 ---
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # 找出概率最大的那个类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算这一轮的平均准确率和损失
        train_acc = 100 * correct / total
        avg_loss = train_loss / len(train_loader)

        # --- 测试阶段 (Validation) ---
        # 每训练完一轮，我们用测试集考考它，看它是不是真的学会了
        model.eval()  # 告诉模型：现在是考试模式，别改参数了
        test_correct = 0
        test_total = 0

        with torch.no_grad():  # 考试时不需要算梯度，节省内存
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100 * test_correct / test_total

        # 打印这一轮的成绩单
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")

    print("\n✅ 训练结束！")

    # ---------------------------------------------------------
    # 5. 保存模型
    # ---------------------------------------------------------
    torch.save(model.state_dict(), 'har_model_weights.pth')
    print("模型参数已保存至 'har_model_weights.pth'")


if __name__ == '__main__':
    train()