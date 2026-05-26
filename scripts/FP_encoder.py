import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    带有残差连接的全连接块。
    结构: x -> [Linear -> BN -> ReLU -> Dropout] + [Shortcut] -> Output
    """

    def __init__(self, in_dim, out_dim, dropout=0.4):
        super(ResidualBlock, self).__init__()

        # 主路径 (Main Path)
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 捷径路径 (Shortcut Path)
        # 如果维度发生变化 (如 512 -> 256)，需要通过一个线性层来对齐维度
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)  # Shortcut路径通常也加BN以保持分布一致
            )
        else:
            self.shortcut = nn.Identity()  # 维度相同时直接相加

    def forward(self, x):
        # 核心残差公式: F(x) + x
        return self.block(x) + self.shortcut(x)


class FingerprintMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512, 256], output_dim=128, dropout=0.4):
        """
        引入残差连接的 MLP 编码器
        :param output_dim: 建议设为 embedding 维度 (如 128)，以便与其他模态融合
        """
        super(FingerprintMLP, self).__init__()

        layers = []
        curr_dim = input_dim

        # 1. 构建残差隐藏层
        # 这里使用 list 循环构建多个 ResBlock
        for h_dim in hidden_dims:
            layers.append(ResidualBlock(curr_dim, h_dim, dropout))
            curr_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)

        # 2. 最终输出层 (Projection Head)
        # 将特征映射到指定的 output_dim (融合维度)
        self.output_layer = nn.Linear(curr_dim, output_dim)

    def forward(self, x):

        # --- 前向传播 ---
        # 提取深层特征
        features = self.feature_extractor(x)

        # 映射到输出维度
        out = self.output_layer(features)

        return out


# --- 测试代码 ---
