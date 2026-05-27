import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import pandas as pd
from collections import OrderedDict


class SmilesTokenizer:
    """
    基于正则表达式的 SMILES 分词器。
    支持从数据中构建词表，并处理未登录词。
    """

    def __init__(self, vocab=None, max_len=200):
        # 初始化基础词表 (包含特殊 token)
        # 0: Pad, 1: Unk, 2: Mask (可选), 3: Start (可选), 4: End (可选)
        if vocab is None:
            self.vocab = OrderedDict({'<pad>': 0, '<unk>': 1})
        else:
            self.vocab = vocab

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.max_len = max_len

        # 业界通用的 SMILES 分词正则 (Schwaller et al.)
        # 它可以正确识别:
        # 1. 方括号内的整体 (如 [O-], [nH], [Na+])
        # 2. 双字符元素 (Br, Cl)
        # 3. 单字符元素
        # 4. 各种键和符号
        self.pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.pattern)

    def tokenize_smiles(self, smiles):
        """将单个 SMILES 字符串分解为 token 列表"""
        return [token for token in self.regex.findall(smiles)]

    def build_vocab(self, smiles_list):
        """
        扫描整个数据集，构建完整的词表。
        """
        print("正在构建 SMILES 词表...")
        unique_tokens = set()
        for smi in smiles_list:
            tokens = self.tokenize_smiles(smi)
            unique_tokens.update(tokens)

        # 将新发现的 token 加入词表
        start_idx = len(self.vocab)
        for i, token in enumerate(sorted(list(unique_tokens))):
            if token not in self.vocab:
                self.vocab[token] = start_idx + i

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"词表构建完成，大小: {len(self.vocab)}")
        print(f"部分 Token 示例: {list(self.vocab.keys())[:15]}")
        return self.vocab

    def encode(self, smiles_list):
        """
        将 SMILES 列表转换为 LongTensor
        """
        token_matrix = []
        for smi in smiles_list:
            tokens = self.tokenize_smiles(smi)
            ids = []
            for token in tokens:
                if token in self.vocab:
                    ids.append(self.vocab[token])
                else:
                    ids.append(self.vocab['<unk>'])

            # 截断或填充
            if len(ids) > self.max_len:
                ids = ids[:self.max_len]
            else:
                ids += [self.vocab['<pad>']] * (self.max_len - len(ids))

            token_matrix.append(ids)

        return torch.tensor(token_matrix, dtype=torch.long)



class SmilesTextCNN(nn.Module):
    """
    使用 1D-CNN 处理 SMILES 序列。
    相比 LSTM，它更能捕捉局部化学子结构 (如羰基、甲基等)，且训练更稳定。
    """

    def __init__(self, vocab_size, embedding_dim=128, output_dim=128,
                 kernel_sizes=[3, 4, 5], num_filters=64, dropout=0.3):
        super(SmilesTextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 定义多个不同尺寸的卷积核 (相当于捕捉 3-gram, 4-gram, 5-gram)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=num_filters,
                      kernel_size=k)
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)

        # 最终映射层
        # 输入维度是 num_filters * len(kernel_sizes)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        # x: [batch, seq_len]
        embed = self.embedding(x)  # [batch, seq_len, emb_dim]

        # Conv1d 要求输入为 [batch, channels, seq_len]
        embed = embed.permute(0, 2, 1)

        # 卷积 + ReLU + MaxPool
        # 1. Conv: [batch, num_filters, seq_len_out]
        # 2. ReLU
        # 3. MaxPool: [batch, num_filters, 1] -> squeeze -> [batch, num_filters]
        conved = [F.max_pool1d(F.relu(conv(embed)), conv(embed).shape[2]).squeeze(2)
                  for conv in self.convs]

        # 拼接所有卷积核的结果
        cat = torch.cat(conved, dim=1)  # [batch, num_filters * len(kernel_sizes)]

        cat = self.dropout(cat)
        return self.fc(cat)








