import random
import argparse
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix, \
    matthews_corrcoef
from rdkit import Chem, rdBase
# 引入 MACCS 生成工具
from rdkit.Chem import MACCSkeys
import warnings

# === 引入核心模型文件 ===
from Multimodal_CGC_with_residual import Prediction_model, UncertaintyWeighting
from GAT_v2 import GATv2, mol_to_graph
from SMILES_encoder import SmilesTokenizer, SmilesTextCNN
from Fingerprint_prepare import FingerprintMLP

# 忽略警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
rdBase.DisableLog('rdApp.warning')


# ==========================================
# 0. 损失函数定义
# ==========================================
class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    def forward(self, logits, targets):
        valid_mask = (targets != -1).float().unsqueeze(1)
        safe_targets = targets.clone()
        safe_targets[safe_targets == -1] = 0
        target_float = safe_targets.float().unsqueeze(1)

        loss = self.bce(logits, target_float)
        loss = loss * valid_mask
        return loss.sum() / (valid_mask.sum() + 1e-8)


# ==========================================
# 1. 配置参数 (Config)
# ==========================================
@dataclass
class Config:
    # 路径配置
    excel_path: str = "data/0_fold_traindata.xlsx"
    val_path: str = "data//0_fold_valdata.xlsx"

    # 保存路径
    model_save_path: str = "model/model.pth"
    log_save_path: str = "log/model.xlsx"

    smiles_col: str = "SMILES"
    label_cols: Tuple[str] = ("GHS_oral", "GHS_dermal", "GHS_inhalation")
    task_names: Tuple[str] = ("Oral", "Dermal", "Inhal")

    # 训练参数
    batch_size: int = 64
    epochs: int = 300
    lr_base: float = 1e-3
    weight_decay: float = 1e-2

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    patience: int = 50

    # === 指纹数据增强配置 (软掩码) ===
    fp_mask_ratio: float = 0.1

    # === CGC 架构超参数 ===
    use_graph: bool = True
    use_smiles: bool = True
    use_fp: bool = True

    # 专家数量
    num_graph_experts: int = 2
    num_smiles_experts: int = 2
    num_fp_experts: int = 2
    num_specific_experts: int = 1

    cgc_expert_hidden: int = 64
    task_expert_hidden: int = 128
    num_tasks: int = 3

    # --- AttentiveFP (Graph) ---
    atom_dim: int = 46
    bond_dim: int = 10
    graph_hidden_dim: int = 128
    atom_layers: int = 3
    mol_layers: int = 2
    graph_output_dim: int = 64
    graph_dropout: float = 0.2
    heads: int = 4

    # --- SMILES TextCNN ---
    smiles_max_len: int = 150
    smiles_embedding_dim: int = 128
    cnn_kernel_sizes: Tuple[int] = (3, 5, 6, 7, 9)
    cnn_filters: int = 64
    smiles_out_dim: int = 64
    cnn_dropout: float = 0.2

    # --- Fingerprint MLP 参数 (MACCS) ---
    fp_input_dim: int = 167
    fp_hidden_dims: Tuple[int] = (256, 128)
    fp_output_dim: int = 64
    fp_dropout: float = 0.2

    # 学习率配置
    lr_graph = 1e-4
    lr_smiles = 5e-5
    lr_fp = 5e-5
    lr_cgc = 5e-5


cfg = Config()


# ==========================================
# 3. 工具函数 (Utils)
# ==========================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def process_dataframe(df, dataset_type="Train"):
    print(f"\n--- Processing {dataset_type} Data ---")
    if cfg.smiles_col not in df.columns:
        possible_cols = ["random_SMILES", "SMILES", "smiles", "canonical_smiles"]
        for c in possible_cols:
            if c in df.columns:
                cfg.smiles_col = c
                break

    df[list(cfg.label_cols)] = df[list(cfg.label_cols)].fillna(-1).astype(int)
    smiles_list_raw = df[cfg.smiles_col].astype(str).tolist()
    labels_raw = df[list(cfg.label_cols)].to_numpy()

    valid_smiles = []
    valid_labels = []

    print("Normalizing SMILES (RDKit Canonicalization)...")
    for smi, lab in zip(smiles_list_raw, labels_raw):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                can_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                valid_smiles.append(can_smi)
                valid_labels.append(lab)
        except:
            continue

    print(f"Valid {dataset_type} samples: {len(valid_smiles)}")
    return valid_smiles, np.array(valid_labels, dtype=np.int64)


def calculate_pos_weights(labels, num_tasks):
    pos_weights = []
    for i in range(num_tasks):
        col = labels[:, i]
        valid = col[col != -1]
        n_pos = (valid == 1).sum()
        n_neg = (valid == 0).sum()
        w = (n_neg / n_pos) if n_pos > 0 else 1.0
        pos_weights.append(torch.tensor([w], dtype=torch.float32))
        print(f"Task {i} Pos Weight: {w:.4f}")
    return pos_weights


# ==========================================
# 4. Dataset (Graph + SMILES + MACCS)
# ==========================================
class GraphSmilesDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, augment=False):
        self.smiles_list = smiles_list
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.augment = augment

        print("Validating Molecules...")
        self.valid_indices = []
        self.cached_ids = []

        for idx, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None: continue

            ids = self.tokenizer.encode([smi])
            if len(ids) == 0: continue

            try:
                t_ids = torch.tensor(ids, dtype=torch.long)
                if t_ids.dim() == 2: t_ids = t_ids.squeeze(0)
                self.cached_ids.append(t_ids)
                self.valid_indices.append(idx)
            except Exception as e:
                print(f"Skipping index {idx} due to encoding error: {e}")
                continue

        if len(self.valid_indices) < len(labels):
            self.labels = self.labels[self.valid_indices]
            self.smiles_list = [self.smiles_list[i] for i in self.valid_indices]

        print(f"Dataset Ready. Valid: {len(self.valid_indices)} | Cached IDs: {len(self.cached_ids)}")

    def __getitem__(self, idx):
        label = self.labels[idx]
        smi = self.smiles_list[idx]
        input_ids = self.cached_ids[idx]
        mol = Chem.MolFromSmiles(smi)

        # 1. 数据增强
        if self.augment:
            try:
                smi_rand = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
                mol_rand = Chem.MolFromSmiles(smi_rand)
                if mol_rand is not None:
                    mol = mol_rand
                    new_ids = self.tokenizer.encode([smi_rand])
                    t_ids = torch.tensor(new_ids, dtype=torch.long)
                    if t_ids.dim() == 2: t_ids = t_ids.squeeze(0)
                    if len(t_ids) <= cfg.smiles_max_len:
                        input_ids = t_ids
            except:
                pass

        # 2. 处理序列长度
        if len(input_ids) < cfg.smiles_max_len:
            pad = torch.zeros(cfg.smiles_max_len - len(input_ids), dtype=torch.long)
            input_ids = torch.cat([input_ids, pad])
        else:
            input_ids = input_ids[:cfg.smiles_max_len]

        # 3. 生成分子图
        graph_data = mol_to_graph(mol)

        # 4. 生成 MACCS 指纹
        try:
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            fp_vec = np.array(maccs_fp.ToList(), dtype=np.float32)

            # 指纹 Soft Masking (数据增强)
            if self.augment and cfg.fp_mask_ratio > 0.0:
                mask = (np.random.rand(*fp_vec.shape) > cfg.fp_mask_ratio).astype(np.float32)
                fp_vec = fp_vec * mask

        except:
            fp_vec = np.zeros((cfg.fp_input_dim,), dtype=np.float32)

        fp_tensor = torch.tensor(fp_vec, dtype=torch.float32)

        return graph_data, input_ids, fp_tensor, label

    def __len__(self):
        return len(self.valid_indices)


# ==========================================
# 5. 指标计算
# ==========================================
def calculate_metrics_dict(logits_list, labels_list, task_idx, prefix="val"):
    metrics = {}
    task_name = cfg.task_names[task_idx]
    if len(logits_list) == 0:
        return {f"{prefix}_{task_name}_{k}": 0.0 for k in ["loss", "acc", "prec", "rec", "spec", "mcc", "f1", "auc"]}

    logits = np.array(logits_list)
    labels = np.array(labels_list)
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    metrics[f"{prefix}_{task_name}_acc"] = accuracy_score(labels, preds)
    metrics[f"{prefix}_{task_name}_prec"] = precision_score(labels, preds, zero_division=0)
    metrics[f"{prefix}_{task_name}_rec"] = recall_score(labels, preds, zero_division=0)
    metrics[f"{prefix}_{task_name}_f1"] = f1_score(labels, preds, zero_division=0)
    metrics[f"{prefix}_{task_name}_mcc"] = matthews_corrcoef(labels, preds)

    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except:
        specificity = 0.0
    metrics[f"{prefix}_{task_name}_spec"] = specificity

    try:
        metrics[f"{prefix}_{task_name}_auc"] = roc_auc_score(labels, probs)
    except:
        metrics[f"{prefix}_{task_name}_auc"] = 0.5
    return metrics


# ==========================================
# 6. 验证/测试函数 (重构以支持不同前缀)
# ==========================================
def evaluate_detailed(model, loader, criterions, device, loss_weighting=None, prefix="val"):
    """
    通过 prefix 参数 ("val" 或 "test") 实现验证集和测试集的代码复用
    """
    model.eval()
    total_loss = 0
    task_losses_sum = [0.0] * cfg.num_tasks
    all_logits = [[] for _ in range(cfg.num_tasks)]
    all_labels = [[] for _ in range(cfg.num_tasks)]

    with torch.no_grad():
        for batch_data in loader:
            graphs = batch_data[0].to(device)
            input_ids = batch_data[1].to(device)
            fps = batch_data[2].to(device)
            labels = batch_data[3].to(device)

            outputs, _ = model(data=graphs, fps=fps, smiles_seq=input_ids, modality_mask=None)

            batch_loss_list = []
            for i, (out_tuple, crit) in enumerate(zip(outputs, criterions)):
                logits, _ = out_tuple
                loss = crit(logits, labels[:, i])
                batch_loss_list.append(loss)
                task_losses_sum[i] += loss.item()

                mask = (labels[:, i] != -1)
                if mask.sum() > 0:
                    all_logits[i].extend(logits[mask].cpu().numpy())
                    all_labels[i].extend(labels[mask, i].cpu().numpy())

            if loss_weighting:
                total_loss += loss_weighting(batch_loss_list).item()
            else:
                total_loss += sum([l.item() for l in batch_loss_list])

    avg_total_loss = total_loss / max(1, len(loader))
    results = {f"{prefix}_loss": avg_total_loss}

    avg_aucs = []
    avg_f1s = []
    avg_accs = []

    for i in range(cfg.num_tasks):
        task_name = cfg.task_names[i]
        results[f"{prefix}_{task_name}_loss"] = task_losses_sum[i] / max(1, len(loader))

        m = calculate_metrics_dict(all_logits[i], all_labels[i], i, prefix)
        results.update(m)

        avg_aucs.append(m[f"{prefix}_{task_name}_auc"])
        avg_f1s.append(m[f"{prefix}_{task_name}_f1"])
        avg_accs.append(m[f"{prefix}_{task_name}_acc"])

    results[f"{prefix}_avg_auc"] = np.mean(avg_aucs) if avg_aucs else 0.0
    results[f"{prefix}_avg_f1"] = np.mean(avg_f1s) if avg_f1s else 0.0
    results[f"{prefix}_avg_acc"] = np.mean(avg_accs) if avg_accs else 0.0

    return results


# ==========================================
# 辅助：打印权重函数
# ==========================================
def print_gate_analysis(epoch_gate_weights):
    print("\n[Gate Analysis] Average Modal Contribution:")
    idx = 0
    indices = {}

    if cfg.use_graph:
        indices['Graph'] = (idx, idx + cfg.num_graph_experts)
        idx += cfg.num_graph_experts

    if cfg.use_smiles:
        indices['SMILES'] = (idx, idx + cfg.num_smiles_experts)
        idx += cfg.num_smiles_experts

    if cfg.use_fp:
        indices['FP'] = (idx, idx + cfg.num_fp_experts)
        idx += cfg.num_fp_experts

    indices['Specific'] = (idx, idx + cfg.num_specific_experts)

    for task_i, weights in enumerate(epoch_gate_weights):
        task_name = cfg.task_names[task_i]
        info_str = f"  Task {task_name:<7}: "
        for name, (start, end) in indices.items():
            modal_weight = weights[start:end].sum()
            info_str += f"{name}={modal_weight:.3f} | "
        print(info_str)


# ==========================================
# 4. 训练函数
# ==========================================
def train_one_epoch(model, loader, optimizer, criterions, loss_weighting, device, epoch_idx):
    model.train()
    train_loss_sum = 0
    batches = 0
    train_task_losses = [0.0] * cfg.num_tasks
    train_logits_all = [[] for _ in range(cfg.num_tasks)]
    train_labels_all = [[] for _ in range(cfg.num_tasks)]

    running_gate_weights = [None] * cfg.num_tasks

    for batch_data in loader:
        graphs = batch_data[0].to(device)
        input_ids = batch_data[1].to(device)
        fps = batch_data[2].to(device)
        labels = batch_data[3].to(device)

        optimizer.zero_grad()
        outputs, _ = model(data=graphs, fps=fps, smiles_seq=input_ids, modality_mask=None)

        loss_list_tensor = []
        for i, (out_tuple, crit) in enumerate(zip(outputs, criterions)):
            logits, gate_w_batch = out_tuple

            # Gate weight handling
            curr_gate_w = gate_w_batch
            if isinstance(curr_gate_w, torch.Tensor):
                curr_gate_w = curr_gate_w.detach().cpu().numpy()
            if curr_gate_w.ndim > 1:
                curr_gate_w = curr_gate_w.mean(axis=0)

            if running_gate_weights[i] is None:
                running_gate_weights[i] = curr_gate_w
            else:
                running_gate_weights[i] += curr_gate_w

            loss = crit(logits, labels[:, i])
            loss_list_tensor.append(loss)
            train_task_losses[i] += loss.item()

            with torch.no_grad():
                mask = (labels[:, i] != -1)
                if mask.sum() > 0:
                    train_logits_all[i].extend(logits[mask].cpu().numpy())
                    train_labels_all[i].extend(labels[mask, i].cpu().numpy())

        total_loss_batch = loss_weighting(loss_list_tensor)
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        train_loss_sum += total_loss_batch.item()
        batches += 1

    avg_gate_weights = []
    if batches > 0:
        for w in running_gate_weights:
            if w is not None:
                avg_gate_weights.append(w / batches)
            else:
                avg_gate_weights.append(np.zeros(1))

    print_gate_analysis(avg_gate_weights)

    epoch_log = {"epoch": epoch_idx}
    epoch_log["train_total_loss"] = train_loss_sum / max(1, batches)
    train_metrics = {}

    for i, task_name in enumerate(cfg.task_names):
        avg_task_loss = train_task_losses[i] / max(1, batches)
        epoch_log[f"train_{task_name}_loss"] = avg_task_loss
        task_metrics = calculate_metrics_dict(train_logits_all[i], train_labels_all[i], i, prefix="train")
        epoch_log.update(task_metrics)
        train_metrics[f"train_{task_name}_loss"] = avg_task_loss
        train_metrics.update(task_metrics)

    return epoch_log, train_metrics


# ==========================================
# 7. 早停类
# ==========================================
class EarlyStopping:
    def __init__(self, patience=7, delta=0, path='checkpoint.pth', mode='max', trace_func=print):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.mode = mode
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        if self.mode == 'min':
            self.monitor_op = np.less
            self.min_delta = -self.delta
        elif self.mode == 'max':
            self.monitor_op = np.greater
            self.min_delta = self.delta
        else:
            raise ValueError(f"EarlyStopping mode must be 'min' or 'max', got {mode}")

    def __call__(self, val_score, model):
        score = val_score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif self.monitor_op(score - self.min_delta, self.best_score):
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, model):
        self.trace_func(f'Validation metric improved to {score:.4f}. Saving model ...')
        torch.save(model.state_dict(), self.path)


# ==========================================
# 8. 主程序
# ==========================================
def main():
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    print("=== Training Multimodal CGC (Graph + SMILES TextCNN + MACCS FP) with Gate Analysis ===")
    print(f"Config: Graph={cfg.use_graph}, SMILES={cfg.use_smiles}, FP={cfg.use_fp}")
    print(
        f"Experts: G={cfg.num_graph_experts}, S={cfg.num_smiles_experts}, F={cfg.num_fp_experts}, Spec={cfg.num_specific_experts}")

    # 加载数据集
    try:
        df_train = pd.read_excel(cfg.excel_path) if not cfg.excel_path.endswith('.csv') else pd.read_csv(cfg.excel_path)
        df_val = pd.read_excel(cfg.val_path) if not cfg.val_path.endswith('.csv') else pd.read_csv(cfg.val_path)
    except FileNotFoundError as e:
        print(f"Error: Data file not found. {e}")
        return

    train_smiles, train_labels = process_dataframe(df_train, "Train")
    val_smiles, val_labels = process_dataframe(df_val, "Val")

    print("Building Tokenizer from Training Data...")
    tokenizer = SmilesTokenizer(max_len=cfg.smiles_max_len)
    tokenizer.build_vocab(train_smiles)
    print(f"Vocab size: {len(tokenizer.vocab)}")

    train_dataset = GraphSmilesDataset(train_smiles, train_labels, tokenizer, augment=True)
    val_dataset = GraphSmilesDataset(val_smiles, val_labels, tokenizer, augment=False)

    pos_weights_list = calculate_pos_weights(train_labels, cfg.num_tasks)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print("Initializing Models (Random Weights)...")

    attentive_fp = GATv2(
        atom_dim=cfg.atom_dim,
        bond_dim=cfg.bond_dim,
        hidden_dim=cfg.graph_hidden_dim,
        atom_layers=cfg.atom_layers,
        mol_layers=cfg.mol_layers,
        output_dim=cfg.graph_output_dim,
        dropout=cfg.graph_dropout,
        heads=cfg.heads
    )

    smiles_encoder = SmilesTextCNN(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=cfg.smiles_embedding_dim,
        output_dim=cfg.smiles_out_dim,
        kernel_sizes=cfg.cnn_kernel_sizes,
        num_filters=cfg.cnn_filters,
        dropout=cfg.cnn_dropout
    )

    fingerprint_mlp = FingerprintMLP(
        input_dim=cfg.fp_input_dim,
        hidden_dims=cfg.fp_hidden_dims,
        output_dim=cfg.fp_output_dim,
        dropout=cfg.fp_dropout
    )

    model = Prediction_model(
        attentive_fp=attentive_fp,
        smiles_encoder=smiles_encoder,
        fingerprint_mlp=fingerprint_mlp,
        use_graph=cfg.use_graph,
        use_smiles=cfg.use_smiles,
        use_fp=cfg.use_fp,
        graph_embed_dim=cfg.graph_output_dim,
        smiles_embed_dim=cfg.smiles_out_dim,
        fp_embed_dim=cfg.fp_output_dim,
        num_graph_experts=cfg.num_graph_experts,
        num_smiles_experts=cfg.num_smiles_experts,
        num_fp_experts=cfg.num_fp_experts,
        num_specific_experts=cfg.num_specific_experts,
        num_tasks=cfg.num_tasks,
        expert_hidden=cfg.cgc_expert_hidden,
        task_expert_hidden=cfg.task_expert_hidden
    ).to(device)

    loss_weighting = UncertaintyWeighting(cfg.num_tasks).to(device)

    optimizer_params = []
    if model.attentive_fp is not None:
        optimizer_params.append({'params': model.attentive_fp.parameters(), 'lr': cfg.lr_graph})
    if model.smiles_encoder is not None:
        optimizer_params.append({'params': model.smiles_encoder.parameters(), 'lr': cfg.lr_smiles})
    if model.fingerprint_mlp is not None:
        optimizer_params.append({'params': model.fingerprint_mlp.parameters(), 'lr': cfg.lr_fp})
    optimizer_params.append({'params': model.cgc.parameters(), 'lr': cfg.lr_cgc})
    optimizer_params.append({'params': loss_weighting.parameters(), 'lr': 1e-3})

    optimizer = optim.AdamW(optimizer_params, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    criterions = []
    for i in range(cfg.num_tasks):
        pw = pos_weights_list[i].to(device)
        criterions.append(MaskedBCEWithLogitsLoss(pos_weight=pw).to(device))

    early_stopping = EarlyStopping(patience=cfg.patience, path=cfg.model_save_path, mode='max')
    history_log = []

    print(f"\nStart Training for {cfg.epochs} epochs...")

    for epoch in range(1, cfg.epochs + 1):
        epoch_log, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterions, loss_weighting, device, epoch
        )
        val_results = evaluate_detailed(model, val_loader, criterions, device, loss_weighting, prefix="val")
        avg_train_loss = epoch_log["train_total_loss"]

        print(f"\nEpoch {epoch:02d}/{cfg.epochs} | Loss: Tr {avg_train_loss:.4f} Val {val_results['val_loss']:.4f} | "
              f"Avg AUC {val_results['val_avg_auc']:.4f}")

        print("-" * 115)
        print(
            f"{'Task':<8} | {'Set':<5} | {'Loss':<7} | {'Acc':<7} | {'Prec':<7} | {'Rec':<7} | {'Spec':<7} | {'MCC':<7} | {'F1':<7} | {'AUC':<7}")
        print("-" * 115)

        metric_keys = ["loss", "acc", "prec", "rec", "spec", "mcc", "f1", "auc"]
        train_agg = {k: [] for k in metric_keys}
        val_agg = {k: [] for k in metric_keys}

        for task in cfg.task_names:
            t_vals = [train_metrics.get(f"train_{task}_{k}", 0) for k in metric_keys]
            for k, v in zip(metric_keys, t_vals): train_agg[k].append(v)
            print(
                f"{task:<8} | Train | {t_vals[0]:.4f}  | {t_vals[1]:.4f}  | {t_vals[2]:.4f}  | {t_vals[3]:.4f}  | {t_vals[4]:.4f}  | {t_vals[5]:.4f}  | {t_vals[6]:.4f}  | {t_vals[7]:.4f}")

            v_vals = [val_results.get(f"val_{task}_{k}", 0) for k in metric_keys]
            for k, v in zip(metric_keys, v_vals): val_agg[k].append(v)
            print(
                f"{'':<8} | Val   | {v_vals[0]:.4f}  | {v_vals[1]:.4f}  | {v_vals[2]:.4f}  | {v_vals[3]:.4f}  | {v_vals[4]:.4f}  | {v_vals[5]:.4f}  | {v_vals[6]:.4f}  | {v_vals[7]:.4f}")
            print("-" * 115)

        t_means = [np.mean(train_agg[k]) for k in metric_keys]
        v_means = [np.mean(val_agg[k]) for k in metric_keys]

        print(
            f"{'AVG':<8} | Train | {t_means[0]:.4f}  | {t_means[1]:.4f}  | {t_means[2]:.4f}  | {t_means[3]:.4f}  | {t_means[4]:.4f}  | {t_means[5]:.4f}  | {t_means[6]:.4f}  | {t_means[7]:.4f}")
        print(
            f"{'':<8} | Val   | {v_means[0]:.4f}  | {v_means[1]:.4f}  | {v_means[2]:.4f}  | {v_means[3]:.4f}  | {v_means[4]:.4f}  | {v_means[5]:.4f}  | {v_means[6]:.4f}  | {v_means[7]:.4f}")
        print("=" * 115)

        with torch.no_grad():
            sigmas = torch.exp(loss_weighting.log_vars).cpu().numpy()
            print(f"[Learned Sigmas] Oral: {sigmas[0]:.3f}, Dermal: {sigmas[1]:.3f}, Inhal: {sigmas[2]:.3f}")

        history_log.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_results["val_loss"],
            **train_metrics,
            **val_results
        })

        try:
            pd.DataFrame(history_log).to_excel(cfg.log_save_path, index=False)
        except PermissionError:
            pass

        scheduler.step()
        early_stopping(val_results["val_avg_auc"], model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Halting training.")
            break


if __name__ == "__main__":
    main()