import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class ResidualExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.3):
        super(ResidualExpert, self).__init__()

        if output_dim is None:
            output_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if hidden_dim != output_dim:
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_proj = nn.Identity()

        if input_dim != output_dim:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim)
            )
        else:
            self.projection = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        out = self.output_proj(out)
        res = self.projection(x)

        return self.relu(out + res)

class MultimodalCGC(nn.Module):
    def __init__(self,
                 graph_dim,
                 smiles_dim,
                 fp_dim,
                 use_graph=True,
                 use_smiles=True,
                 use_fp=True,
                 num_tasks=3,
                 num_classes=1,
                 num_graph_experts=3,
                 num_smiles_experts=3,
                 num_fp_experts=3,
                 num_specific_experts=2,
                 expert_hidden=128,  
                 task_expert_hidden=128,
                 dropout=0.3):
        super(MultimodalCGC, self).__init__()

        self.num_tasks = num_tasks
        self.expert_hidden = expert_hidden
        self.task_expert_hidden = task_expert_hidden
        self.num_specific_experts = num_specific_experts

        self.use_graph = use_graph
        self.use_smiles = use_smiles
        self.use_fp = use_fp

        if not any([use_graph, use_smiles, use_fp]):
            raise ValueError("At least one modality must be enabled!")

        if self.use_graph:
            if num_graph_experts <= 0: raise ValueError("num_graph_experts must be > 0")
            self.graph_experts = nn.ModuleList([
                ResidualExpert(input_dim=graph_dim, hidden_dim=expert_hidden, output_dim=expert_hidden, dropout=dropout)
                for _ in range(num_graph_experts)
            ])
        else:
            self.graph_experts = None

        if self.use_smiles:
            if num_smiles_experts <= 0: raise ValueError("num_smiles_experts must be > 0")
            self.smiles_experts = nn.ModuleList([
                ResidualExpert(input_dim=smiles_dim, hidden_dim=expert_hidden, output_dim=expert_hidden,
                               dropout=dropout)
                for _ in range(num_smiles_experts)
            ])
        else:
            self.smiles_experts = None

        if self.use_fp:
            if num_fp_experts <= 0: raise ValueError("num_fp_experts must be > 0")
            self.fp_experts = nn.ModuleList([
                ResidualExpert(input_dim=fp_dim, hidden_dim=expert_hidden, output_dim=expert_hidden, dropout=dropout)
                for _ in range(num_fp_experts)
            ])
        else:
            self.fp_experts = None

        self.joint_dim = 0
        if use_graph: self.joint_dim += graph_dim
        if use_smiles: self.joint_dim += smiles_dim
        if use_fp: self.joint_dim += fp_dim

        self.total_experts_for_gate = 0
        if use_graph: self.total_experts_for_gate += num_graph_experts
        if use_smiles: self.total_experts_for_gate += num_smiles_experts
        if use_fp: self.total_experts_for_gate += num_fp_experts

        self.total_experts_for_gate += num_specific_experts

        if self.num_specific_experts > 0:
            self.specific_experts = nn.ModuleList([
                nn.ModuleList([
                    ResidualExpert(
                        input_dim=self.joint_dim,
                        hidden_dim=task_expert_hidden,
                        output_dim=expert_hidden,
                        dropout=dropout
                    )
                    for _ in range(num_specific_experts)
                ]) for _ in range(num_tasks)
            ])
        else:
            self.specific_experts = None

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.joint_dim, self.total_experts_for_gate),
                nn.Softmax(dim=1)
            ) for _ in range(num_tasks)
        ])


        self.task_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden, expert_hidden // 2),
                nn.BatchNorm1d(expert_hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden // 2, num_classes)
            ) for _ in range(num_tasks)
        ])

    def forward(self, graph_feat=None, smiles_feat=None, fp_feat=None):
        all_expert_outputs = []
        active_feats = []

        if self.use_graph:
            if graph_feat is None: raise ValueError("Graph enabled but feat is None")
            active_feats.append(graph_feat)
            g_outs = [e(graph_feat).unsqueeze(2) for e in self.graph_experts]
            all_expert_outputs.append(torch.cat(g_outs, dim=2))

        if self.use_smiles:
            if smiles_feat is None: raise ValueError("SMILES enabled but feat is None")
            active_feats.append(smiles_feat)
            s_outs = [e(smiles_feat).unsqueeze(2) for e in self.smiles_experts]
            all_expert_outputs.append(torch.cat(s_outs, dim=2))

        if self.use_fp:
            if fp_feat is None: raise ValueError("FP enabled but feat is None")
            active_feats.append(fp_feat)
            f_outs = [e(fp_feat).unsqueeze(2) for e in self.fp_experts]
            all_expert_outputs.append(torch.cat(f_outs, dim=2))

        joint_feat = torch.cat(active_feats, dim=1)

        final_outputs = []

        for i in range(self.num_tasks):
            if self.num_specific_experts > 0:
                spec_outs = [e(joint_feat).unsqueeze(2) for e in self.specific_experts[i]]
                spec_outs_tensor = torch.cat(spec_outs, dim=2)
                current_task_experts = all_expert_outputs + [spec_outs_tensor]
            else:
                current_task_experts = all_expert_outputs

            all_experts_tensor = torch.cat(current_task_experts, dim=2)
            gate = self.gates[i](joint_feat).unsqueeze(1)  # [batch, 1, total_experts]
            gate_weights_for_analysis = gate.squeeze(1).mean(dim=0).detach().cpu().numpy()
            weighted_expert = torch.bmm(all_experts_tensor, gate.transpose(1, 2)).squeeze(2)
            logits = self.task_towers[i](weighted_expert)
            final_outputs.append((logits, gate_weights_for_analysis))

        return final_outputs

class Prediction_model(nn.Module):
    def __init__(self,
                 attentive_fp=None,
                 smiles_encoder=None,
                 fingerprint_mlp=None,
                 graph_embed_dim=None,
                 smiles_embed_dim=None,
                 fp_embed_dim=None,
                 use_graph=True,
                 use_smiles=True,
                 use_fp=True,
                 num_graph_experts=3,
                 num_smiles_experts=3,
                 num_fp_experts=3,
                 num_specific_experts=2,
                 expert_hidden=128,  
                 task_expert_hidden=128,  
                 num_tasks=3):
        super(Prediction_model, self).__init__()

        self.attentive_fp = attentive_fp
        self.smiles_encoder = smiles_encoder
        self.fingerprint_mlp = fingerprint_mlp

        self.use_graph = use_graph
        self.use_smiles = use_smiles
        self.use_fp = use_fp

        self.graph_dim = 0
        self.smiles_dim = 0
        self.fp_dim = 0

        if use_graph:
            if graph_embed_dim is not None:
                self.graph_dim = graph_embed_dim
            elif hasattr(attentive_fp, 'output_dim'):
                self.graph_dim = attentive_fp.output_dim
            else:
                self.graph_dim = 128
                print(f"Warning: Could not detect graph_dim, using default {self.graph_dim}")

        if use_smiles:
            if smiles_embed_dim is not None:
                self.smiles_dim = smiles_embed_dim
            elif hasattr(smiles_encoder, 'output_dim'):
                self.smiles_dim = smiles_encoder.output_dim
            elif hasattr(smiles_encoder, 'fc') and hasattr(smiles_encoder.fc, 'out_features'):
                self.smiles_dim = smiles_encoder.fc.out_features
            else:
                self.smiles_dim = 128
                print(f"Warning: Could not detect smiles_dim, using default {self.smiles_dim}")

        if use_fp:
            if fingerprint_mlp is None:
                raise ValueError("use_fp=True but fingerprint_mlp is None")
            if fp_embed_dim is not None:
                self.fp_dim = fp_embed_dim
            elif hasattr(fingerprint_mlp, 'output_dim'):
                self.fp_dim = fingerprint_mlp.output_dim
            elif hasattr(fingerprint_mlp, 'network'):
                try:
                    self.fp_dim = fingerprint_mlp.network[-1].out_features
                except:
                    self.fp_dim = 128
            else:
                try:
                    self.fp_dim = fingerprint_mlp[-1].out_features
                except:
                    raise ValueError("无法自动检测 fingerprint_mlp 维度，请传入 fp_embed_dim 参数。")

        print(
            f"Init CGC | Modalities: G({use_graph}:{self.graph_dim}) S({use_smiles}:{self.smiles_dim}) F({use_fp}:{self.fp_dim})")
        print(f"Experts Config | Shared Hidden: {expert_hidden} | Task Specific Hidden: {task_expert_hidden}")

        self.cgc = MultimodalCGC(
            graph_dim=self.graph_dim,
            smiles_dim=self.smiles_dim,
            fp_dim=self.fp_dim,
            use_graph=use_graph,
            use_smiles=use_smiles,
            use_fp=use_fp,
            num_graph_experts=num_graph_experts,
            num_smiles_experts=num_smiles_experts,
            num_fp_experts=num_fp_experts,
            num_specific_experts=num_specific_experts,
            expert_hidden=expert_hidden,
            task_expert_hidden=task_expert_hidden,  
            num_tasks=num_tasks,
            num_classes=1,
            dropout=0.25
        )

    def forward(self, data=None, fps=None, smiles_seq=None, modality_mask=None):
        graph_feat = None
        smiles_feat = None
        fp_feat = None
        atom_importance = None

        if self.use_graph:
            if data is None: raise ValueError("Graph enabled but data missing")
            graph_feat, atom_importance = self.attentive_fp(data)
            if self.training and modality_mask is not None and len(modality_mask) >= 1 and modality_mask[0] == 1:
                graph_feat = torch.zeros_like(graph_feat)

        if self.use_smiles:
            if smiles_seq is None: raise ValueError("SMILES enabled but seq missing")
            smiles_feat = self.smiles_encoder(smiles_seq)
            if self.training and modality_mask is not None and len(modality_mask) >= 2 and modality_mask[1] == 1:
                smiles_feat = torch.zeros_like(smiles_feat)

        if self.use_fp:
            if fps is None: raise ValueError("FP enabled but fps missing")
            fp_feat = self.fingerprint_mlp(fps)
            if self.training and modality_mask is not None and len(modality_mask) >= 3 and modality_mask[2] == 1:
                fp_feat = torch.zeros_like(fp_feat)

        task_outputs = self.cgc(graph_feat, smiles_feat, fp_feat)
        return task_outputs, atom_importance


class UncertaintyWeighting(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            log_var = self.log_vars[i]
            precision = torch.exp(-log_var)
            total_loss += precision * loss + log_var
        return total_loss
