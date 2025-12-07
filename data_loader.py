import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# 配置区域
# ---------------------------------------------------------
# 注意：在字符串前加 'r' 是为了告诉 Python 这是一个"原始字符串"，
# 不要把反斜杠 '\' 当作转义字符处理（Windows路径必备技巧）
DATA_PATH = r"F:\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset"


# ---------------------------------------------------------
# 数据读取函数
# ---------------------------------------------------------

def load_X(X_signal_paths):
    X_signals = []

    print("正在读取传感器信号数据，可能需要几秒钟...")
    for signal_type_path in X_signal_paths:
        # 读取txt文件
        file = open(signal_type_path, 'r')

        # 下面这行代码做了三件事：
        # 1. replace: 去掉多余的空格
        # 2. split: 按空格切割成数字字符串
        # 3. float: 转成浮点数
        X_signals.append(
            [np.array(series, dtype=np.float32) for series in
             [row.replace('  ', ' ').strip().split(' ') for row in file]]
        )
        file.close()

    # 此时 X_signals 是一个 list，包含9个 (7352, 128) 的矩阵
    # 我们将其转换为 numpy 数组并转置
    # 转置前形状: (9, 7352, 128)
    # 转置后形状: (7352, 128, 9) -> (样本数, 时间步, 通道数)
    return np.transpose(np.array(X_signals), (1, 2, 0))


def load_y(y_path):
    print(f"正在读取标签文件: {y_path}")
    file = open(y_path, 'r')
    # 读取每一行，转为整数
    # 原始标签是 1-6，为了深度学习(从0开始计数)，我们减去 1，变成 0-5
    y_ = np.array(
        [int(row.replace('  ', ' ').strip().split(' ')[0]) for row in file],
        dtype=np.int32
    )
    file.close()
    return y_ - 1


def load_data(mode='train'):
    """
    mode: 'train' 或 'test'
    """
    # 9种信号的文件名部分
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_", "body_acc_y_", "body_acc_z_",
        "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
        "total_acc_x_", "total_acc_y_", "total_acc_z_"
    ]

    # 拼接完整路径
    X_signal_paths = [
        os.path.join(DATA_PATH, mode, "Inertial Signals", signal + mode + '.txt')
        for signal in INPUT_SIGNAL_TYPES
    ]
    y_path = os.path.join(DATA_PATH, mode, "y_" + mode + '.txt')

    # 加载特征和标签
    X = load_X(X_signal_paths)
    y = load_y(y_path)

    return X, y


# ---------------------------------------------------------
# 主程序入口 (测试代码)
# ---------------------------------------------------------
if __name__ == '__main__':
    # 检查路径是否存在，避免报错一脸懵
    if not os.path.exists(DATA_PATH):
        print(f"错误：路径不存在！请检查你的硬盘: {DATA_PATH}")
    else:
        print("路径检查通过，开始加载数据...")

        try:
            # 1. 加载训练集
            X_train, y_train = load_data('train')
            print(f"\n=== 训练集加载完成 ===")
            print(f"X_train 形状: {X_train.shape}")
            print(f"y_train 形状: {y_train.shape}")

            # 2. 加载测试集
            X_test, y_test = load_data('test')
            print(f"\n=== 测试集加载完成 ===")
            print(f"X_test 形状: {X_test.shape}")
            print(f"y_test 形状: {y_test.shape}")

            # 3. 验证一下数据对不对
            # 期望结果：(7352, 128, 9)
            if X_train.shape == (7352, 128, 9):
                print("\n✅ 成功！数据形状非常完美，可以进行下一步模型设计了。")
            else:
                print("\n⚠️ 警告：数据形状看起来不太对，请检查文件。")

        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            print("提示：请检查是否安装了 numpy (pip install numpy)")