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

def get_atom_features(atom: rdchem.Atom) -> np.ndarray:

    atom_symbols = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'Metal']
    symbol_feature = [1 if atom.GetSymbol() == s else 0 for s in atom_symbols]
    if sum(symbol_feature) == 0: symbol_feature[-1] = 1

    degree_feature = [0] * 6
    deg = min(atom.GetDegree(), 5)
    degree_feature[deg] = 1

    formal_charge = [atom.GetFormalCharge()]
    radical_electrons = [atom.GetNumRadicalElectrons()]

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

    aromatic_feature = [1 if atom.GetIsAromatic() else 0]

    num_h_feature = [0] * 5
    num_h = min(atom.GetTotalNumHs(), 4)
    num_h_feature[num_h] = 1

    chirality_feature = [1 if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED else 0]
    chiral_type_feature = [0, 0]
    if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        chiral_type_feature[0] = 1
    elif atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        chiral_type_feature[1] = 1
    try:
        charge = float(atom.GetProp('_GasteigerCharge'))
        if math.isnan(charge) or math.isinf(charge):
            charge = 0.0
    except:
        charge = 0.0
    charge_feature = [charge]

    ring_size_feature = [0] * 5
    if atom.IsInRing():
        if atom.IsInRingSize(3): ring_size_feature[0] = 1
        elif atom.IsInRingSize(4): ring_size_feature[1] = 1
        elif atom.IsInRingSize(5): ring_size_feature[2] = 1
        elif atom.IsInRingSize(6): ring_size_feature[3] = 1
        else: ring_size_feature[4] = 1 # >6

    mass_feature = [atom.GetMass() * 0.01]

    return np.concatenate([
        symbol_feature, degree_feature, formal_charge, radical_electrons,
        hyb_feature, aromatic_feature, num_h_feature,
        chirality_feature, chiral_type_feature,
        charge_feature, ring_size_feature, mass_feature
    ])


def get_bond_features(bond: rdchem.Bond) -> np.ndarray:
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

class ModernAtomUpdateLayer(MessagePassing):
    def __init__(self, atom_dim, bond_dim, heads=4, dropout=0.4, **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.atom_dim = atom_dim
        self.heads = heads
        self.dropout = dropout
        self.head_dim = atom_dim // heads

        assert atom_dim % heads == 0, f"Atom dim {atom_dim} must be divisible by heads {heads}"
        self.edge_encoder = nn.Linear(bond_dim, atom_dim)

        self.lin_l = nn.Linear(atom_dim, atom_dim)  
        self.lin_r = nn.Linear(atom_dim, atom_dim)  
        self.att = nn.Parameter(torch.Tensor(1, heads, self.head_dim))

        self.ffn = nn.Sequential(
            nn.Linear(atom_dim, atom_dim * 2),  
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(atom_dim * 2, atom_dim)  
        )

        self.norm1 = nn.LayerNorm(atom_dim)
        self.norm2 = nn.LayerNorm(atom_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.edge_encoder.weight)

    def forward(self, x, edge_index, edge_attr):
        edge_emb = self.edge_encoder(edge_attr)
        residual = x
        x = self.norm1(x)  # Pre-Norm
        x_out = self.propagate(edge_index, x=x, edge_attr=edge_emb)
        x = x_out + residual

        residual = x
        x = self.norm2(x)  # Pre-Norm
        x = self.ffn(x)
        x = x + residual

        return x

    def message(self, x_i, x_j, edge_attr, index):
        x_j_fused = x_j + edge_attr
        g_l = self.lin_l(x_i).view(-1, self.heads, self.head_dim)
        g_r = self.lin_r(x_j_fused).view(-1, self.heads, self.head_dim)

        g = F.leaky_relu(g_l + g_r)
        alpha = (g * self.att).sum(dim=-1)

        alpha = softmax(alpha, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j_fused.view(-1, self.heads, self.head_dim) * alpha.unsqueeze(-1)

        return out.view(-1, self.atom_dim)

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

        self.atom_embedding = nn.Linear(atom_dim, hidden_dim)

        self.atom_layers = nn.ModuleList()
        for _ in range(atom_layers):
            self.atom_layers.append(
                ModernAtomUpdateLayer(
                    atom_dim=hidden_dim,  
                    bond_dim=bond_dim,  
                    heads=heads,  
                    dropout=dropout  
                )
            )

        self.readout_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

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

        x = self.atom_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.atom_layers:
            x = layer(x, edge_index, edge_attr)

        gate_logits = self.readout_gate(x)
        atom_weights = softmax(gate_logits, batch)  

        mol_rep = global_add_pool(atom_weights * x, batch)

        atom_importance_list = []
        cpu_weights = atom_weights.detach().cpu().numpy()
        cpu_batch = batch.detach().cpu().numpy()
        num_graphs = int(cpu_batch.max()) + 1

        for g in range(num_graphs):
            mask = (cpu_batch == g)
            atom_importance_list.append(cpu_weights[mask])

        for mlp in self.mol_layers:
            mol_rep = mlp(mol_rep)

        return mol_rep, atom_importance_list

    def forward(self, data):
        mol_rep, atom_importance = self.encode(data)
        output = self.output_layer(mol_rep)
        return output, atom_importance

if __name__ == '__main__':

    smiles = 'O=C(C)Oc1ccccc1C(=O)O'  
    mol = Chem.MolFromSmiles(smiles)
    graph = mol_to_graph(mol)

    batch_graph = Batch.from_data_list([graph])

    input_atom_dim = batch_graph.x.shape[1]
    input_bond_dim = batch_graph.edge_attr.shape[1]

    model = GATv2(
        atom_dim=input_atom_dim,
        bond_dim=input_bond_dim,
        hidden_dim=64,  
        atom_layers=2, 
        mol_layers=1,
        dropout=0.4  
    )

    print(f"模型配置: Hidden=64, Layers=2, Heads=4, Dropout=0.4")
    print(f"参数量: {sum(p.numel() for p in model.parameters())}")

    model.eval()
    with torch.no_grad():
        pred, importance = model(batch_graph)

    print(f"\n输入分子: {smiles}")
    print(f"预测输出: {pred.item():.4f}")

    weights = importance[0]
    top_idx = weights.flatten().argmax()
    atom_sym = mol.GetAtomWithIdx(int(top_idx)).GetSymbol()
    print(f"模型认为最重要的原子: Index {top_idx} ({atom_sym}), Weight {weights[top_idx][0]:.4f}")
