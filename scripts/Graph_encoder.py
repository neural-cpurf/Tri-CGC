from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import AllChem

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import softmax

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# =============================================================================
# 1. 特征提取部分 (保持标准)
# =============================================================================

def get_atom_features(atom: rdchem.Atom) -> np.ndarray:
    """
    针对农药毒性预测增强版原子特征
    新增：Gasteiger电荷、环大小、氢键性质
    """
    # 1. 原子符号 (保持不变)
    atom_symbols = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'Metal']
    symbol_feature = [1 if atom.GetSymbol() == s else 0 for s in atom_symbols]
    if sum(symbol_feature) == 0: symbol_feature[-1] = 1

    # 2. 度 (保持不变)
    degree_feature = [0] * 6
    deg = min(atom.GetDegree(), 5)
    degree_feature[deg] = 1

    # 3. 电子与化合价 (保持不变)
    formal_charge = [atom.GetFormalCharge()]
    radical_electrons = [atom.GetNumRadicalElectrons()]

    # 4. 杂化 (保持不变)
    hybridizations = [
        Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2, 'other'
    ]
    hyb_feature = [0] * 6
    hyb = atom.GetHybridization()
    if hyb in hybridizations[:5]:
        hyb_feature[hybridizations.index(hyb)] = 1
    else:
        hyb_feature[5] = 1

    # 5. 芳香性 (保持不变)
    aromatic_feature = [1 if atom.GetIsAromatic() else 0]

    # 6. 连接氢数 (保持不变)
    num_h_feature = [0] * 5
    num_h = min(atom.GetTotalNumHs(), 4)
    num_h_feature[num_h] = 1

    # 7. 手性 (保持不变 - 农药中手性非常重要)
    chirality_feature = [1 if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED else 0]
    chiral_type_feature = [0, 0]
    if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        chiral_type_feature[0] = 1
    elif atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        chiral_type_feature[1] = 1

    # --- 【新增 1】 Gasteiger 偏电荷 ---
    # 电荷是浮点数，最好归一化一下，或者直接作为一个数值特征
    # RDKit 计算后会存储在 Prop 中
    try:
        charge = float(atom.GetProp('_GasteigerCharge'))
        if math.isnan(charge) or math.isinf(charge):
            charge = 0.0
    except:
        charge = 0.0
    charge_feature = [charge]

    # --- 【新增 2】 环的大小 (Ring Size) ---
    # 检查原子是否在 3, 4, 5, 6, 7+ 元环中
    ring_size_feature = [0] * 5
    if atom.IsInRing():
        if atom.IsInRingSize(3): ring_size_feature[0] = 1
        elif atom.IsInRingSize(4): ring_size_feature[1] = 1
        elif atom.IsInRingSize(5): ring_size_feature[2] = 1
        elif atom.IsInRingSize(6): ring_size_feature[3] = 1
        else: ring_size_feature[4] = 1 # >6

    # --- 【新增 3】 质量 (Mass) ---
    # 归一化质量 (除以 100 简单缩放)
    mass_feature = [atom.GetMass() * 0.01]

    # 合并
    return np.concatenate([
        symbol_feature, degree_feature, formal_charge, radical_electrons,
        hyb_feature, aromatic_feature, num_h_feature,
        chirality_feature, chiral_type_feature,
        charge_feature, ring_size_feature, mass_feature
    ])


def get_bond_features(bond: rdchem.Bond) -> np.ndarray:
    """计算键的10维特征向量"""
    bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    bond_type_feature = [1 if bond.GetBondType() == bt else 0 for bt in bond_types]

    conjugated_feature = [1 if bond.GetIsConjugated() else 0]
    ring_feature = [1 if bond.IsInRing() else 0]

    stereo_types = [Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY, Chem.BondStereo.STEREOZ,
                    Chem.BondStereo.STEREOE]
    stereo_feature = [0] * 4
    stereo = bond.GetStereo()
    if stereo in stereo_types:
        stereo_feature[stereo_types.index(stereo)] = 1

    return np.concatenate([bond_type_feature, conjugated_feature, ring_feature, stereo_feature])


def mol_to_graph(mol):
    AllChem.ComputeGasteigerCharges(mol)
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.extend([(i, j), (j, i)])
        bond_features = get_bond_features(bond)
        edge_attr.extend([bond_features, bond_features])

    return Data(
        x=torch.tensor(np.array(atom_features), dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(np.array(edge_attr), dtype=torch.float)
    )


# =============================================================================
# 2. 现代原子更新层 (GATv2 + GINE + LayerNorm + Residual)
# =============================================================================

class ModernAtomUpdateLayer(MessagePassing):
    """
    针对小样本数据优化的原子更新层
    特点：
    1. 使用 LayerNorm 和 Residual (残差连接)，训练极度稳定。
    2. 使用 GATv2 (动态注意力)，表达能力强。
    3. 使用 GINE (边特征融合)，更好地利用化学键信息。
    4. 移除 GRU，改用 MLP，计算更快且更容易正则化。
    """

    def __init__(self, atom_dim, bond_dim, heads=4, dropout=0.4, **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.atom_dim = atom_dim
        self.heads = heads
        self.dropout = dropout
        self.head_dim = atom_dim // heads

        assert atom_dim % heads == 0, f"Atom dim {atom_dim} must be divisible by heads {heads}"

        # 1. 边特征对齐层 (将 bond_dim 映射到 atom_dim 以便相加)
        self.edge_encoder = nn.Linear(bond_dim, atom_dim)

        # 2. GATv2 注意力机制参数
        self.lin_l = nn.Linear(atom_dim, atom_dim)  # Source 投影
        self.lin_r = nn.Linear(atom_dim, atom_dim)  # Target 投影
        self.att = nn.Parameter(torch.Tensor(1, heads, self.head_dim))

        # 3. 前馈神经网络 (FFN) - 替代 GRU
        self.ffn = nn.Sequential(
            nn.Linear(atom_dim, atom_dim * 2),  # 升维
            nn.ReLU(),
            nn.Dropout(dropout),  # 强Dropout
            nn.Linear(atom_dim * 2, atom_dim)  # 降维
        )

        # 4. 归一化层 (Pre-Norm 结构)
        self.norm1 = nn.LayerNorm(atom_dim)
        self.norm2 = nn.LayerNorm(atom_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.edge_encoder.weight)

    def forward(self, x, edge_index, edge_attr):
        # 预处理边特征
        edge_emb = self.edge_encoder(edge_attr)

        # --- Block 1: Graph Attention ---
        residual = x
        x = self.norm1(x)  # Pre-Norm

        # 消息传递
        x_out = self.propagate(edge_index, x=x, edge_attr=edge_emb)

        # 残差连接 1
        x = x_out + residual

        # --- Block 2: Feed Forward (MLP) ---
        residual = x
        x = self.norm2(x)  # Pre-Norm
        x = self.ffn(x)

        # 残差连接 2
        x = x + residual

        return x

    def message(self, x_i, x_j, edge_attr, index):
        # x_i: target (中心原子), x_j: source (邻居原子)

        # [GINE] 边特征融合: 邻居特征 + 边特征
        # 这比简单的拼接更能体现 "化学键定义了邻居性质" 的物理意义
        x_j_fused = x_j + edge_attr

        # [GATv2] 动态注意力计算
        # Reshape 为多头 [E, heads, head_dim]
        g_l = self.lin_l(x_i).view(-1, self.heads, self.head_dim)
        g_r = self.lin_r(x_j_fused).view(-1, self.heads, self.head_dim)

        # LeakyReLU 在求和之后应用 (GATv2 关键特征)
        g = F.leaky_relu(g_l + g_r)

        # 计算分数: (E, heads, dim) * (1, heads, dim) -> Sum -> (E, heads)
        alpha = (g * self.att).sum(dim=-1)

        # Softmax 归一化
        alpha = softmax(alpha, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 消息加权
        # x_j_fused: [E, dim] -> [E, heads, head_dim]
        out = x_j_fused.view(-1, self.heads, self.head_dim) * alpha.unsqueeze(-1)

        # 展平多头回归 [E, atom_dim]
        return out.view(-1, self.atom_dim)


# =============================================================================
# 3. 模型主体 (Modern AttentiveFP Style)
# =============================================================================

class GATv2(nn.Module):
    def __init__(self, atom_dim=39, bond_dim=10, hidden_dim=64,
                 atom_layers=2, mol_layers=1, output_dim=1, heads=4, dropout=0.4):
        super().__init__()

        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.heads = heads
        self.dropout = dropout

        # 1. 初始原子嵌入
        self.atom_embedding = nn.Linear(atom_dim, hidden_dim)

        # 2. 堆叠现代化的原子更新层
        self.atom_layers = nn.ModuleList()
        for _ in range(atom_layers):
            self.atom_layers.append(
                ModernAtomUpdateLayer(
                    atom_dim=hidden_dim,  # 输入输出同维度
                    bond_dim=bond_dim,  # 原始键维度 (层内会投影)
                    heads=heads,  # 4头注意力
                    dropout=dropout  # 高 Dropout
                )
            )

        # 3. Global Readout Gate (计算分子级注意力)
        self.readout_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 4. 预测 MLP
        self.mol_layers = nn.ModuleList()
        for _ in range(mol_layers):
            self.mol_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def encode(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 初始嵌入
        x = self.atom_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 迭代更新原子特征
        for layer in self.atom_layers:
            x = layer(x, edge_index, edge_attr)
            # 注意: LayerNorm 和 Residual 已经在 layer 内部处理了

        # --- 全局池化 (Readout) ---
        # 计算注意力权重
        gate_logits = self.readout_gate(x)
        atom_weights = softmax(gate_logits, batch)  # 真实权重

        # 加权求和得到分子向量
        mol_rep = global_add_pool(atom_weights * x, batch)

        # 提取可视化权重 (仅在 CPU 上操作，避免阻塞 GPU)
        atom_importance_list = []
        cpu_weights = atom_weights.detach().cpu().numpy()
        cpu_batch = batch.detach().cpu().numpy()
        num_graphs = int(cpu_batch.max()) + 1

        for g in range(num_graphs):
            mask = (cpu_batch == g)
            atom_importance_list.append(cpu_weights[mask])

        # 分子 MLP 处理
        for mlp in self.mol_layers:
            mol_rep = mlp(mol_rep)

        return mol_rep, atom_importance_list

    def forward(self, data):
        mol_rep, atom_importance = self.encode(data)
        output = self.output_layer(mol_rep)
        return output, atom_importance


# =============================================================================
# 4. 运行测试
# =============================================================================

if __name__ == '__main__':
    # 1. 准备数据
    smiles = 'O=C(C)Oc1ccccc1C(=O)O'  # 阿司匹林
    mol = Chem.MolFromSmiles(smiles)
    graph = mol_to_graph(mol)

    # 【重要】封装为 Batch 对象
    batch_graph = Batch.from_data_list([graph])

    # 2. 动态获取输入维度
    input_atom_dim = batch_graph.x.shape[1]
    input_bond_dim = batch_graph.edge_attr.shape[1]

    # 3. 初始化模型 (针对 6000 样本的小数据配置)
    model = GATv2(
        atom_dim=input_atom_dim,
        bond_dim=input_bond_dim,
        hidden_dim=64,  # 窄网络，参数少
        atom_layers=2,  # 浅层，防过平滑
        mol_layers=1,  # 简单的读出层
        dropout=0.4  # 强正则化
    )

    print(f"模型配置: Hidden=64, Layers=2, Heads=4, Dropout=0.4")
    print(f"参数量: {sum(p.numel() for p in model.parameters())}")

    # 4. 运行预测
    model.eval()
    with torch.no_grad():
        pred, importance = model(batch_graph)

    print(f"\n输入分子: {smiles}")
    print(f"预测输出: {pred.item():.4f}")

    # 可视化最重要的原子
    weights = importance[0]
    top_idx = weights.flatten().argmax()
    atom_sym = mol.GetAtomWithIdx(int(top_idx)).GetSymbol()
    print(f"模型认为最重要的原子: Index {top_idx} ({atom_sym}), Weight {weights[top_idx][0]:.4f}")