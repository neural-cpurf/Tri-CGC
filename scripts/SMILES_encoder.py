import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import pandas as pd
from collections import OrderedDict


class SmilesTokenizer:

    def __init__(self, vocab=None, max_len=200):
        if vocab is None:
            self.vocab = OrderedDict({'<pad>': 0, '<unk>': 1})
        else:
            self.vocab = vocab

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.max_len = max_len

        self.pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.pattern)

    def tokenize_smiles(self, smiles):
        return [token for token in self.regex.findall(smiles)]

    def build_vocab(self, smiles_list):
        print("正在构建 SMILES 词表...")
        unique_tokens = set()
        for smi in smiles_list:
            tokens = self.tokenize_smiles(smi)
            unique_tokens.update(tokens)

        start_idx = len(self.vocab)
        for i, token in enumerate(sorted(list(unique_tokens))):
            if token not in self.vocab:
                self.vocab[token] = start_idx + i

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"词表构建完成，大小: {len(self.vocab)}")
        print(f"部分 Token 示例: {list(self.vocab.keys())[:15]}")
        return self.vocab

    def encode(self, smiles_list):
        token_matrix = []
        for smi in smiles_list:
            tokens = self.tokenize_smiles(smi)
            ids = []
            for token in tokens:
                if token in self.vocab:
                    ids.append(self.vocab[token])
                else:
                    ids.append(self.vocab['<unk>'])

            if len(ids) > self.max_len:
                ids = ids[:self.max_len]
            else:
                ids += [self.vocab['<pad>']] * (self.max_len - len(ids))

            token_matrix.append(ids)

        return torch.tensor(token_matrix, dtype=torch.long)



class SmilesTextCNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim=128, output_dim=128,
                 kernel_sizes=[3, 4, 5], num_filters=64, dropout=0.3):
        super(SmilesTextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=num_filters,
                      kernel_size=k)
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        embed = self.embedding(x)  # [batch, seq_len, emb_dim]
        embed = embed.permute(0, 2, 1)
        conved = [F.max_pool1d(F.relu(conv(embed)), conv(embed).shape[2]).squeeze(2)
                  for conv in self.convs]

        cat = torch.cat(conved, dim=1)  # [batch, num_filters * len(kernel_sizes)]

        cat = self.dropout(cat)
        return self.fc(cat)








