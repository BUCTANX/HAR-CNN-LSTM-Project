import torch
import torch.nn as nn
import torch.nn.functional as F


class HARModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        """
        参数说明:
        input_dim: 输入通道数 (这里是 9，因为有 9 个传感器通道)
        hidden_dim: LSTM 的隐藏层神经元数量
        layer_dim: LSTM 的层数
        output_dim: 输出类别数 (这里是 6，对应走路、站立等 6 类)
        """
        super(HARModel, self).__init__()

        # -----------------------------------
        # 1. 1D CNN 层 (特征提取)
        # -----------------------------------
        # Conv1d 参数: (输入通道, 输出通道, 卷积核大小)
        # 它可以把 9 个原始通道的信息混合，提取出 64 种更高级的波形特征
        self.cnn_layer = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3)

        # -----------------------------------
        # 2. LSTM 层 (时序分析)
        # -----------------------------------
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM 输入: (Batch, Seq_Len, Features)
        # 注意: CNN 处理后，Feature 维度变成了 64
        self.lstm = nn.LSTM(
            input_size=64,  # 输入特征数 (CNN的输出通道)
            hidden_size=hidden_dim,  # LSTM 记忆单元的大小
            num_layers=layer_dim,  # 堆叠几层 LSTM
            batch_first=True  # 让输入数据的第一个维度是 Batch Size
        )

        # -----------------------------------
        # 3. 全连接层 (分类输出)
        # -----------------------------------
        # 这里的 126 是怎么来的？
        # 原始时间步 128 -> 经过 kernel_size=3 的卷积 -> 变成了 128 - 3 + 1 = 126
        # 如果你改了卷积核大小，这里也要相应修改
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        前向传播逻辑
        x 的形状: (Batch_Size, 128, 9)
        """

        # --- 步骤 1: CNN 处理 ---
        # PyTorch 的 Conv1d 要求输入形状为 (Batch, Channel, Length)
        # 而我们的 x 是 (Batch, Length, Channel)，所以要交换一下维度
        x = x.permute(0, 2, 1)  # 变成 (Batch, 9, 128)

        out = self.cnn_layer(x)  # 经过卷积
        out = F.relu(out)  # 激活函数

        # 现在的 out 形状: (Batch, 64, 126)

        # --- 步骤 2: LSTM 处理 ---
        # LSTM 要求输入形状为 (Batch, Length, Features)
        # 所以我们要把维度再换回来
        out = out.permute(0, 2, 1)  # 变成 (Batch, 126, 64)

        # LSTM 输出包含: out (所有时间步的输出), (hn, cn) (最后的状态)
        # 我们只需要 out
        out, (hn, cn) = self.lstm(out)

        # --- 步骤 3: 分类 ---
        # 我们通常只需要 LSTM 读完整个序列后的"最后一眼"看到的特征
        # 取最后一个时间步的数据
        out = out[:, -1, :]

        # 进入全连接层
        out = self.fc(out)

        return out


# --- 简单测试模型结构是否正确 ---
if __name__ == "__main__":
    # 模拟一个 Batch 的数据: 32个样本, 128时间步, 9通道
    dummy_input = torch.randn(32, 128, 9)

    # 实例化模型
    model = HARModel(input_dim=9, hidden_dim=100, layer_dim=2, output_dim=6)

    # 试着跑一次前向传播
    output = model(dummy_input)

    print("模型构建成功！")
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")  # 应该输出 (32, 6)

    if output.shape == (32, 6):
        print("✅ 测试通过：输出维度正确，模型逻辑无误。")