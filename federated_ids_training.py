"""
Federated IoT-IDS Training Framework (FedAvg / FedProx / FedDyn)
==================================================================
English summary (full inline documentation below is in Arabic, the
author's working language during development and validation):

This module implements the validated "v3" federated-learning framework
used throughout the accompanying thesis and papers (FedTri-IDS,
FedGov_XAI, and the client-scale-generalization study). It performs:

  1. Leakage-free preprocessing of the CIC-ToN-IoT dataset (split before
     any fit; RobustScaler and Mutual-Information feature selection
     fitted on the training split only).
  2. "Double non-IID" client partitioning: clients are first grouped by
     network protocol (TCP / UDP / other), then split within each group
     using a Dirichlet(alpha) distribution over class labels.
  3. Local training and federated aggregation for three algorithms:
     FedAvg (sample-size-weighted averaging), FedProx (proximal term),
     and FedDyn (dynamic regularization with a persisted correction
     term h_i, per Acar et al., 2021).
  4. Per-client personalization (local fine-tuning) and a
     "personalization gain" metric (F1 after fine-tuning minus F1 of
     the un-personalized global model).
  5. Automatic saving of per-seed results, checkpoints, and a combined
     CSV summary, plus paired t-test / Wilcoxon signed-rank statistical
     comparisons between algorithms.

This is the exact framework whose outputs are reported in the papers'
result tables; the code has been validated and should not be modified
without re-running (and re-reporting) all downstream statistics.

Usage
-----
    python federated_ids_training.py --data-path data/CIC-ToN-IoT-V2.parquet \
        --output-dir outputs/fl_outputs_v2 --rounds 80 --seeds 42 123 2024

The dataset itself (CIC-ToN-IoT) is not included in this repository —
see the README for how to obtain it. Running the full experiment
(80 rounds x 3 algorithms x 3 seeds) is compute-intensive and was
originally run on Google Colab with GPU acceleration.
"""

# ============================================================================
# CELL 1 — الإعداد العام، الإعدادات الموحّدة FLConfig، قفل البذور
# ============================================================================
from __future__ import annotations

import os
import gc
import copy
import json
import time
import random
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score,
)
from scipy.stats import ttest_rel, wilcoxon

import matplotlib.pyplot as plt
import seaborn as sns

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
except ImportError:
    print("⚠️ لسنا داخل Colab — تخطي mount(). عدّلي cfg.file_path يدوياً إذا لزم.")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ الجهاز المستخدم: {DEVICE}")


def lock_all_seeds(seed: int = 42) -> None:
    """قفل كل مولدات الأرقام العشوائية لضمان قابلية إعادة الإنتاج."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class FLConfig:
    """كل معاملات التجربة في مكان واحد — يُحفظ تلقائياً بجانب النتائج."""
    # --- البيانات ---
    file_path: str = "data/CIC-ToN-IoT-V2.parquet"
    output_dir: str = "outputs/fl_outputs_v2"
    top_k: int = 27
    total_rows: int = 500_000
    mi_sample_size: int = 80_000
    test_size: float = 0.20
    val_size: float = 0.10
    data_seed: int = 42

    # --- تقسيم العملاء (Double Non-IID) ---
    dirichlet_alpha: float = 0.5
    split_seed: int = 42
    client_groups: dict = field(default_factory=lambda: {
        'tcp': [0, 1, 2, 3], 'udp': [4, 5, 6], 'other': [7, 8, 9]
    })
    # [C] استخدام أعمدة البروتوكول للتقسيم فقط وحذفها من مدخلات النموذج
    drop_protocol_features: bool = True
    # [V3-1] سقف حجم تدريب أي عميل (None لإلغائه — يعيد سيناريو العميل المهيمن)
    max_client_train_samples: int | None = 60_000
    # [V3-2] حصة مضمونة من كل صنف لكل عميل (حيثما توفر الصنف في مجموعته)
    min_class_quota_train: int = 500
    min_class_quota_eval: int = 100

    # --- التدريب الفيدرالي ---
    batch_size: int = 64
    eval_batch_size: int = 512
    rounds: int = 80
    patience: int = 20
    local_epochs: int = 1            # [E] عدد epochs المحلية لكل جولة
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_decay: float = 0.98           # [V3-4] تخميد lr يعالج التذبذب بعد الجولات الأولى
    # [V3-3] 'weighted' = FedAvg الأصلي (المرجّح) | 'uniform' = متوسط موحّد (للـ ablation)
    aggregation: str = 'weighted'
    participation_fraction: float = 1.0  # [E] نسبة العملاء المشاركين كل جولة
    grad_clip: float = 1.0

    # --- معاملات الخوارزميات ---
    mu: float = 0.01                 # FedProx
    feddyn_alpha: float = 0.1        # FedDyn

    # --- التخصيص والتكرار ---
    pers_epochs: int = 1             # epochs التخصيص المحلي بعد التدريب العالمي
    pers_lr: float = 1e-3
    seeds: tuple = (42, 123, 2024)   # يُفضَّل 5 seeds إن سمح الوقت
    save_checkpoints: bool = True

    @property
    def n_clients(self) -> int:
        return sum(len(v) for v in self.client_groups.values())


PROTOCOL_COL = "Protocol"
TARGET_COLUMN = "Label"

SCALING_FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Fwd Packets Length Total', 'Bwd Packets Length Total',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Max',
    'Fwd IAT Total', 'Bwd IAT Total', 'Packet Length Max', 'Packet Length Mean',
]

OTHER_NUMERIC_FEATURES = [
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'Fwd Header Length', 'Bwd Header Length', 'Down/Up Ratio',
]


# ============================================================================
# CELL 2 — تجهيز البيانات بدون تسرّب (Leakage-Free Preprocessing)
# ============================================================================
def add_engineered_features(df: pd.DataFrame, protocol_col: str = PROTOCOL_COL) -> pd.DataFrame:
    """
    هندسة ميزات رياضية بحتة (log / sqrt / مربعات / نسب) — لا يوجد أي fit،
    لذلك تطبيقها مستقلاً على train/val/test آمن تماماً.
    """
    df = df.copy()
    eps = 1e-9

    for c in df.columns:
        if c != protocol_col:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)

    def _clip_nonneg(s):
        return np.clip(s.astype(np.float64), 0, None)

    base_cols = [c for c in df.columns if c != protocol_col]
    for c in base_cols:
        x = _clip_nonneg(df[c])
        df[f"log_{c}"] = np.log1p(x)
        df[f"sqrt_{c}"] = np.sqrt(x + eps)

    sq_cols = [c for c in base_cols
               if any(k in c for k in ["Packet Length", "Flow", "Header", "Packets", "IAT"])]
    for c in sq_cols:
        x = df[c].astype(np.float64)
        df[f"sq_{c}"] = x * x

    if {"Packet Length Max", "Packet Length Min"} <= set(df.columns):
        df["pkt_len_range"] = df["Packet Length Max"] - df["Packet Length Min"]
    if {"Packet Length Std", "Packet Length Mean"} <= set(df.columns):
        df["pkt_len_cv"] = df["Packet Length Std"] / (df["Packet Length Mean"].abs() + eps)
    if {"Flow IAT Std", "Flow IAT Mean"} <= set(df.columns):
        df["iat_cv"] = df["Flow IAT Std"] / (df["Flow IAT Mean"].abs() + eps)
    if {"Fwd Packets Length Total", "Bwd Packets Length Total"} <= set(df.columns):
        df["total_bytes"] = df["Fwd Packets Length Total"] + df["Bwd Packets Length Total"]
        df["ratio_fwd_bwd_bytes"] = (
            (df["Fwd Packets Length Total"] + eps) / (df["Bwd Packets Length Total"] + eps)
        )
    if {"Fwd Header Length", "Bwd Header Length"} <= set(df.columns):
        df["total_header_len"] = df["Fwd Header Length"] + df["Bwd Header Length"]
        df["ratio_fwd_bwd_header"] = (
            (df["Fwd Header Length"] + eps) / (df["Bwd Header Length"] + eps)
        )

    flag_cols = [c for c in df.columns if "Flag Count" in c]
    if flag_cols:
        df["flags_sum"] = df[flag_cols].sum(axis=1)

    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    df.fillna(0.0, inplace=True)
    return df


def stratified_sample_df(df: pd.DataFrame, y: np.ndarray, total_rows: int, seed: int) -> pd.DataFrame:
    """عيّنة طبقية تحافظ على نسب الأصناف — لتوفير الذاكرة قبل المعالجة الثقيلة."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    props = counts / counts.sum()

    parts = []
    for cls, p in zip(classes, props):
        cls_idx = np.where(y == cls)[0]
        take = min(int(round(total_rows * p)), len(cls_idx))
        parts.append(rng.choice(cls_idx, size=take, replace=False))
    idx = np.concatenate(parts)
    rng.shuffle(idx)
    return df.iloc[idx].reset_index(drop=True)


def prepare_iot_dataset(cfg: FLConfig):
    """
    يرجع قاموساً فيه Tensors لكل من train/val/test بعد:
      1) تقسيم Train/Val/Test أولاً (قبل أي fit إحصائي) — يمنع التسرب.
      2) هندسة ميزات لا تحتاج fit — آمنة على كل جزء.
      3) RobustScaler يُدرَّب على Train فقط.
      4) One-Hot للبروتوكول بفئات Train فقط.
      5) اختيار top_k ميزة عبر Mutual Information من عيّنة Train فقط.
    ملاحظة: أعمدة البروتوكول تُحتفظ دائماً في X (لأن تقسيم العملاء يحتاجها)،
    لكن يمكن استبعادها من مدخلات النموذج لاحقاً عبر model_feature_idx.
    """
    print("🚀 تحميل البيانات ومعالجتها (بدون تسرّب)...")
    cols = SCALING_FEATURES + [PROTOCOL_COL] + OTHER_NUMERIC_FEATURES + [TARGET_COLUMN]
    df = pd.read_parquet(cfg.file_path, columns=cols)
    df.dropna(inplace=True)
    print(f"✅ تم التحميل والتنظيف: {df.shape}")

    le = LabelEncoder()
    y_all = le.fit_transform(df[TARGET_COLUMN].values)

    df_small = stratified_sample_df(df, y_all, total_rows=cfg.total_rows, seed=cfg.data_seed)
    del df
    gc.collect()

    y_small = le.transform(df_small[TARGET_COLUMN].values)
    X_small = df_small.drop(columns=[TARGET_COLUMN]).copy()
    del df_small
    gc.collect()

    # 1) التقسيم أولاً — قبل أي fit إحصائي
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_small, y_small, test_size=cfg.test_size, random_state=cfg.data_seed, stratify=y_small
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=cfg.val_size,
        random_state=cfg.data_seed, stratify=y_train_full
    )
    del X_small, y_small, X_train_full, y_train_full
    gc.collect()

    # 2) هندسة الميزات (بدون fit)
    X_train = add_engineered_features(X_train)
    X_val = add_engineered_features(X_val)
    X_test = add_engineered_features(X_test)

    # 3) القياس — fit على Train فقط
    scaler = RobustScaler()
    num_cols = [c for c in X_train.columns if c != PROTOCOL_COL]
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # 4) One-Hot للبروتوكول — فئات Train فقط
    train_cats = sorted(pd.Series(X_train[PROTOCOL_COL]).dropna().unique().tolist())
    for split_df in (X_train, X_val, X_test):
        split_df[PROTOCOL_COL] = pd.Categorical(split_df[PROTOCOL_COL], categories=train_cats)

    X_train_oh = pd.get_dummies(X_train, columns=[PROTOCOL_COL], prefix=PROTOCOL_COL, dummy_na=False)
    X_val_oh = pd.get_dummies(X_val, columns=[PROTOCOL_COL], prefix=PROTOCOL_COL, dummy_na=False)
    X_test_oh = pd.get_dummies(X_test, columns=[PROTOCOL_COL], prefix=PROTOCOL_COL, dummy_na=False)
    X_train_oh, X_val_oh = X_train_oh.align(X_val_oh, join="left", axis=1, fill_value=0)
    X_train_oh, X_test_oh = X_train_oh.align(X_test_oh, join="left", axis=1, fill_value=0)

    feature_names = X_train_oh.columns.tolist()
    proto_cols = [c for c in feature_names if c.startswith(f"{PROTOCOL_COL}_")]

    # 5) اختيار الميزات عبر MI — من عيّنة Train فقط
    rng = np.random.default_rng(cfg.data_seed)
    take = min(cfg.mi_sample_size, len(y_train))
    mi_idx = rng.choice(len(y_train), size=take, replace=False)
    X_mi = X_train_oh.iloc[mi_idx].to_numpy(dtype=np.float32, copy=False)
    y_mi = np.asarray(y_train)[mi_idx]
    mi = mutual_info_classif(X_mi, y_mi, random_state=cfg.data_seed)
    mi_series = pd.Series(mi, index=feature_names).sort_values(ascending=False)

    # أعمدة البروتوكول ضرورية للتقسيم — تُحفظ دائماً في X
    must_keep = proto_cols
    ranked = [c for c in mi_series.index if c not in must_keep]
    selected = must_keep + ranked[: max(0, cfg.top_k - 0)]  # top_k ميزة نمذجة + أعمدة البروتوكول

    # [C] فهارس ميزات النموذج: مع/بدون أعمدة البروتوكول حسب الإعداد
    if cfg.drop_protocol_features:
        model_features = [c for c in selected if c not in proto_cols]
    else:
        model_features = selected
    model_feature_idx = [selected.index(c) for c in model_features]

    print(f"✅ عدد المرشحين={len(feature_names)} | أعمدة X={len(selected)} "
          f"| مدخلات النموذج={len(model_features)} "
          f"(drop_protocol_features={cfg.drop_protocol_features})")
    print("أعلى 10 ميزات حسب Mutual Information:\n", mi_series.head(10))

    def to_tensor(df_oh, y):
        x_t = torch.from_numpy(df_oh[selected].to_numpy(dtype=np.float32, copy=True))
        y_t = torch.tensor(np.asarray(y), dtype=torch.long)
        return x_t, y_t

    X_train_t, y_train_t = to_tensor(X_train_oh, y_train)
    X_val_t, y_val_t = to_tensor(X_val_oh, y_val)
    X_test_t, y_test_t = to_tensor(X_test_oh, y_test)

    del X_train, X_val, X_test, X_train_oh, X_val_oh, X_test_oh
    gc.collect()

    print(f"✅ input_dim (نموذج)={len(model_feature_idx)} | حجم train/val/test = "
          f"{len(y_train_t)}/{len(y_val_t)}/{len(y_test_t)}")

    return dict(
        X_train=X_train_t, y_train=y_train_t,
        X_val=X_val_t, y_val=y_val_t,
        X_test=X_test_t, y_test=y_test_t,
        label_encoder=le, selected_features=selected,
        model_feature_idx=model_feature_idx, mi_scores=mi_series,
    )


# ============================================================================
# CELL 3 — تقسيم العملاء (Double Non-IID) + بناء الـ Loaders + التشخيص
# ============================================================================
def get_protocol_masks(X: torch.Tensor, feature_names: list) -> dict:
    """
    فهارس الصفوف لكل مجموعة (TCP=Protocol_6 / UDP=Protocol_17 / Other=الباقي).
    'other' تُعرَّف كمتمّمة (كل ما ليس TCP ولا UDP) — يضمن تغطية كل الصفوف
    حتى لو كان One-Hot لصف ما كله أصفار (بروتوكول غير مرئي في Train).
    """
    tcp_col = f"{PROTOCOL_COL}_6"
    udp_col = f"{PROTOCOL_COL}_17"
    if tcp_col not in feature_names or udp_col not in feature_names:
        raise ValueError(
            "لم يتم العثور على Protocol_6 (TCP) و/أو Protocol_17 (UDP) ضمن أعمدة X. "
            "تأكدي أن أعمدة البروتوكول ضمن must_keep في prepare_iot_dataset."
        )
    tcp_i = feature_names.index(tcp_col)
    udp_i = feature_names.index(udp_col)

    is_tcp = X[:, tcp_i] > 0
    is_udp = X[:, udp_i] > 0
    is_other = ~(is_tcp | is_udp)

    return {
        'tcp': torch.where(is_tcp)[0],
        'udp': torch.where(is_udp)[0],
        'other': torch.where(is_other)[0],
    }


def partition_clients_protocol_dirichlet(
    X: torch.Tensor, y: torch.Tensor, feature_names: list,
    alpha: float, seed: int, client_groups: dict,
    min_class_quota: int = 0,
):
    """
    Double Non-IID: فصل تام حسب البروتوكول (كل عميل بروتوكول واحد) +
    Dirichlet غير متوازن للفئات داخل كل مجموعة. مولّد عشوائي محلي seeded.

    [V3-2] min_class_quota: قبل توزيع Dirichlet، يُضمن لكل عميل حصة ثابتة
    من كل صنف *متوفر في مجموعة بروتوكوله*. يحافظ على non-IID في الباقي
    لكنه يمنع عملاء تقييم أحاديي الصنف عندما يكون الصنفان موجودين أصلاً.
    إن كانت المجموعة كلها صنفاً واحداً (مثل 'other' في CIC-ToN-IoT) فلا
    يمكن للحصة أن تخلق الصنف الغائب — وهذه خاصية بيانات تُوثَّق كما هي.
    """
    n_clients = sum(len(v) for v in client_groups.values())
    rng = np.random.default_rng(seed)
    masks = get_protocol_masks(X, feature_names)

    client_indices = {i: [] for i in range(n_clients)}
    class_labels = torch.unique(y).tolist()

    for group_idx, (proto_name, clients) in enumerate(client_groups.items()):
        # [V3.1] نسب Dirichlet تُرسم من مولد ثابت لكل (مجموعة، صنف) — مستقلة عن
        # حجم البيانات، فتتطابق نسب أصناف كل عميل عبر train/val/test.
        proto_indices = masks[proto_name]
        n_group_clients = len(clients)
        if len(proto_indices) == 0:
            print(f"⚠️ تحذير: مجموعة البروتوكول '{proto_name}' بلا عينات في هذا الجزء.")
            continue

        y_sub = y[proto_indices]
        for c_label in class_labels:
            class_mask = torch.where(y_sub == c_label)[0]
            global_class_indices = proto_indices[class_mask]
            n_samples = len(global_class_indices)
            if n_samples == 0:
                continue

            perm = torch.from_numpy(rng.permutation(n_samples))
            global_class_indices = global_class_indices[perm]

            # [V3-2] الحصة المضمونة أولاً (إن توفرت عينات كافية)
            pos = 0
            q = min(min_class_quota, n_samples // n_group_clients)
            if q > 0:
                for client_id in clients:
                    client_indices[client_id].append(global_class_indices[pos: pos + q])
                    pos += q

            remaining = global_class_indices[pos:]
            n_rem = len(remaining)
            if n_rem == 0:
                continue

            # الباقي يوزَّع Dirichlet كالمعتاد (يحافظ على non-IID)
            # [V3.1] مولد النسب ثابت لكل (seed, مجموعة, صنف) عبر كل الـ splits
            prop_rng = np.random.default_rng(
                (seed * 1_000_003 + group_idx * 10_007 + int(c_label) * 101) % (2**32)
            )
            proportions = prop_rng.dirichlet([alpha] * n_group_clients)
            start_idx = 0
            for idx, client_id in enumerate(clients):
                if idx == n_group_clients - 1:
                    client_samples = remaining[start_idx:]
                else:
                    take = int(proportions[idx] * n_rem)
                    client_samples = remaining[start_idx: start_idx + take]
                    start_idx += take
                if len(client_samples):
                    client_indices[client_id].append(client_samples)

    return client_indices, client_groups


def build_client_loaders(X, y, client_indices, model_feature_idx,
                         batch_size=64, shuffle=True, seed=42, max_samples=None):
    """
    يبني DataLoader لكل عميل (على ميزات النموذج فقط model_feature_idx)
    + جدول إحصائي. الخلط داخل الـ loaders مضبوط بمولّد seeded [F].
    [V3-1] max_samples: سقف عشوائي طبقي تقريبي لحجم بيانات العميل — يمنع
    هيمنة عميل ضخم واحد على التجميع المرجّح (يُستخدم للتدريب فقط).
    """
    col_idx = torch.tensor(model_feature_idx, dtype=torch.long)
    loaders, stats_rows = [], []
    for client_id in sorted(client_indices.keys()):
        parts = client_indices[client_id]
        if parts:
            idx = torch.cat(parts)
            g_np = np.random.default_rng(seed + 1000 + client_id)
            idx = idx[torch.from_numpy(g_np.permutation(len(idx)))]
            if max_samples is not None and len(idx) > max_samples:
                # الخلط تم قبل القص، والقص يحافظ تقريبياً على نسب الأصناف
                idx = idx[:max_samples]
        else:
            idx = torch.tensor([], dtype=torch.long)

        X_c = X[idx][:, col_idx] if len(idx) else torch.empty(0, len(col_idx))
        y_c = y[idx] if len(idx) else torch.empty(0, dtype=torch.long)

        gen = torch.Generator()
        gen.manual_seed(seed + client_id)
        loaders.append(DataLoader(
            TensorDataset(X_c, y_c), batch_size=batch_size,
            shuffle=shuffle and len(y_c) > 0, generator=gen,
        ))

        n0 = int((y_c == 0).sum()) if len(y_c) else 0
        n1 = int((y_c == 1).sum()) if len(y_c) else 0
        stats_rows.append({'client_id': client_id, 'n_samples': int(len(y_c)),
                           'n_benign': n0, 'n_attack': n1})

    return loaders, pd.DataFrame(stats_rows)


def diagnose_client_splits(train_stats: pd.DataFrame, test_stats: pd.DataFrame) -> pd.DataFrame:
    """
    فحص تشخيصي إلزامي قبل الثقة بأي نتيجة: يكشف العملاء بعينات اختبار
    قليلة/غير متوازنة (المفسّر الأشهر لدقة 100% الزائفة) والعملاء الفارغين.
    """
    merged = train_stats.merge(test_stats, on='client_id', suffixes=('_train', '_test'))
    print("\n📋 توزيع بيانات العملاء (تدريب مقابل اختبار):")
    print(merged.to_string(index=False))
    for _, row in merged.iterrows():
        if row['n_samples_train'] == 0:
            print(f"🛑 العميل {row['client_id']}: بلا بيانات تدريب إطلاقاً — سيُتخطى في كل الجولات. "
                  f"فكّري في رفع alpha أو تغيير split_seed.")
        if row['n_samples_test'] < 200 or row['n_benign_test'] == 0 or row['n_attack_test'] == 0:
            print(f"⚠️ العميل {row['client_id']}: بيانات اختبار قليلة/غير متوازنة "
                  f"(n={row['n_samples_test']}, benign={row['n_benign_test']}, "
                  f"attack={row['n_attack_test']}). أي دقة 100% هنا تُفسَّر بصغر العينة "
                  f"وليست دليل تفوّق — وثّقي هذا صراحة في الأطروحة.")
    return merged


# ============================================================================
# CELL 4 — النموذج ودوال التقييم والأوزان
# ============================================================================
class IotIDSModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def evaluate_model(model, X, y, device, batch_size=512):
    """يرجع acc/prec/rec/f1/fpr + AUC [F]. يتعامل بأمان مع صنف مفقود."""
    model.eval()
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    probs_all, preds_all, trues_all = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs_all.append(torch.softmax(logits, dim=1)[:, 1].cpu())
            preds_all.append(torch.argmax(logits, dim=1).cpu())
            trues_all.append(yb)
    probs = torch.cat(probs_all).numpy()
    preds = torch.cat(preds_all).numpy()
    trues = torch.cat(trues_all).numpy()

    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, average='macro', zero_division=0)
    rec = recall_score(trues, preds, average='macro', zero_division=0)
    f1 = f1_score(trues, preds, average='macro', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(trues, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    try:
        auc_v = roc_auc_score(trues, probs)
    except ValueError:
        auc_v = float('nan')  # صنف واحد فقط في العينة
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'fpr': fpr, 'auc': auc_v}


def calculate_local_class_weights(labels_tensor: torch.Tensor, device) -> torch.Tensor:
    if len(labels_tensor) == 0:
        return torch.ones(2, device=device)
    counts = torch.bincount(labels_tensor, minlength=2).float().clamp(min=1)
    weights = len(labels_tensor) / (2.0 * counts)
    return torch.clamp(weights, min=0.1, max=10.0).to(device)


def loader_size(loader: DataLoader) -> int:
    return len(loader.dataset)


# ============================================================================
# CELL 5 — التدريب المحلي والتجميع (FedAvg / FedProx / FedDyn)
# ============================================================================
def _train_local(model, global_model, loader, device, mode, cfg: FLConfig,
                 prev_grad=None, lr=None):
    """تدريب محلي موحّد. يدعم local_epochs [E] ويعيد state_dict."""
    lr = cfg.lr if lr is None else lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    model.train()

    all_labels = torch.cat([yb for _, yb in loader]).long()
    class_weights = calculate_local_class_weights(all_labels, device)

    for _ in range(cfg.local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device).long()
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb, weight=class_weights)

            if mode == 'fedprox':
                prox = sum((w - w_t).pow(2).sum()
                           for w, w_t in zip(model.parameters(), global_model.parameters()))
                loss = loss + (cfg.mu / 2.0) * prox
            elif mode == 'feddyn':
                lin_pen, quad_pen = 0.0, 0.0
                for p, g_p, pg in zip(model.parameters(), global_model.parameters(), prev_grad):
                    lin_pen = lin_pen + torch.sum(p * pg)
                    quad_pen = quad_pen + torch.sum((p - g_p) ** 2)
                loss = loss - lin_pen + (cfg.feddyn_alpha / 2.0) * quad_pen

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

    return model.state_dict()


def aggregate_weighted(global_model, client_states, client_sizes):
    """
    [A] تجميع مرجّح بعدد عينات كل عميل (FedAvg الأصلي) — يُستخدم لـ
    FedAvg و FedProx. مرّري client_sizes=None للمتوسط الموحّد (FedDyn).
    """
    keys = global_model.state_dict().keys()
    if client_sizes is None:
        w = torch.ones(len(client_states))
    else:
        w = torch.tensor(client_sizes, dtype=torch.float64)
    w = (w / w.sum()).tolist()

    new_w = {}
    for k in keys:
        stacked = torch.stack([cs[k].float() * wi for cs, wi in zip(client_states, w)])
        agg = stacked.sum(0)
        orig_dtype = client_states[0][k].dtype
        if orig_dtype in (torch.int64, torch.int32):
            new_w[k] = agg.round().to(orig_dtype)
        else:
            new_w[k] = agg.to(orig_dtype)
    return new_w


def aggregate_feddyn(global_model, client_states, global_grad, feddyn_alpha):
    """تجميع FedDyn: متوسط موحّد + تصحيح الحالة العالمية h (كما في الورقة)."""
    current = global_model.state_dict()
    avg_w = aggregate_weighted(global_model, client_states, client_sizes=None)

    new_w = {}
    with torch.no_grad():
        for k in current.keys():
            if "num_batches_tracked" in k or "running" in k:
                new_w[k] = avg_w[k]
                continue
            diff = avg_w[k] - current[k]
            global_grad[k] -= feddyn_alpha * diff
            new_w[k] = avg_w[k] - (1.0 / feddyn_alpha) * global_grad[k]
    return new_w, global_grad


# ============================================================================
# CELL 6 — محرك التجربة الموحّد (Global + Personalization Gain)
# ============================================================================
def _concat_loaders(loaders):
    xs = [torch.cat([xb for xb, _ in ld]) for ld in loaders if loader_size(ld) > 0]
    ys = [torch.cat([yb for _, yb in ld]) for ld in loaders if loader_size(ld) > 0]
    return torch.cat(xs), torch.cat(ys)


def run_federated_experiment(algo: str, input_dim: int,
                             train_loaders, val_loaders, test_loaders,
                             cfg: FLConfig, seed: int, device=DEVICE, verbose=True):
    assert algo in ('fedavg', 'fedprox', 'feddyn')
    lock_all_seeds(seed)  # نفس التهيئة العشوائية لكل الخوارزميات عند نفس الـ seed

    n_clients = cfg.n_clients
    global_model = IotIDSModel(input_dim).to(device)
    total_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)

    train_sizes = [loader_size(ld) for ld in train_loaders]
    active_pool = [cid for cid in range(n_clients) if train_sizes[cid] > 0]  # [B]
    if len(active_pool) < n_clients:
        print(f"⚠️ عملاء بلا بيانات تدريب سيُتخطون نهائياً: "
              f"{sorted(set(range(n_clients)) - set(active_pool))}")

    if algo == 'feddyn':
        if cfg.participation_fraction < 1.0:
            print("⚠️ FedDyn مع مشاركة جزئية يتطلب معالجة خاصة لحالة h — "
                  "يُنصح بإبقاء participation_fraction=1.0 مع FedDyn.")
        global_grad = {k: torch.zeros_like(v).to(device)
                       for k, v in global_model.state_dict().items()
                       if "num_batches_tracked" not in k and "running" not in k}
        client_grads = {cid: [torch.zeros_like(p).to(device)
                              for p in global_model.parameters()]
                        for cid in active_pool}

    # Validation مجمّعة — تُستخدم فقط لاختيار أفضل جولة (وليس Test)
    val_X, val_y = _concat_loaders(val_loaders)

    history = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'fpr': [], 'auc': []}
    best_f1, best_round, best_state = 0.0, 0, None
    patience_counter, completed_rounds = 0, 0
    part_rng = np.random.default_rng(seed + 7)

    for r in range(cfg.rounds):
        completed_rounds += 1
        lr_r = cfg.lr * (cfg.lr_decay ** r)

        # [E] اختيار العملاء المشاركين هذه الجولة
        if cfg.participation_fraction < 1.0:
            m = max(1, int(round(cfg.participation_fraction * len(active_pool))))
            round_clients = sorted(part_rng.choice(active_pool, size=m, replace=False).tolist())
        else:
            round_clients = active_pool

        client_states, round_sizes = [], []
        global_snapshot = copy.deepcopy(global_model).to(device)

        for cid in round_clients:
            local_model = copy.deepcopy(global_model).to(device)
            if algo == 'feddyn':
                state = _train_local(local_model, global_snapshot, train_loaders[cid],
                                     device, mode='feddyn', cfg=cfg,
                                     prev_grad=client_grads[cid], lr=lr_r)
                with torch.no_grad():
                    for i_p, (p, g_p) in enumerate(zip(local_model.parameters(),
                                                       global_snapshot.parameters())):
                        client_grads[cid][i_p] -= cfg.feddyn_alpha * (p - g_p)
            else:
                state = _train_local(local_model, global_snapshot, train_loaders[cid],
                                     device, mode=algo, cfg=cfg, lr=lr_r)
            client_states.append(state)
            round_sizes.append(train_sizes[cid])

        if algo == 'feddyn':
            new_w, global_grad = aggregate_feddyn(global_model, client_states,
                                                  global_grad, cfg.feddyn_alpha)
        else:
            # [A]+[V3-3] مرجّح (FedAvg الأصلي) أو موحّد (ablation)
            sizes = round_sizes if cfg.aggregation == 'weighted' else None
            new_w = aggregate_weighted(global_model, client_states, sizes)
        global_model.load_state_dict(new_w)

        # التقييم لاختيار أفضل جولة — Validation فقط
        metrics = evaluate_model(global_model, val_X, val_y, device, cfg.eval_batch_size)
        for k in history:
            history[k].append(metrics[k])

        if verbose:
            print(f"[{algo.upper()} | seed={seed}] Round [{r + 1:02d}/{cfg.rounds}] "
                  f"| Val F1: {metrics['f1']:.4f} | Best: {max(history['f1']):.4f} "
                  f"| Val FPR: {metrics['fpr']:.4f} | Val AUC: {metrics['auc']:.4f}")

        if metrics['f1'] > best_f1:
            best_f1, best_round = metrics['f1'], r + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in global_model.state_dict().items()}  # [G] على CPU
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"⏹️ Early stopping عند الجولة {r + 1}")
                break

    if best_state is not None:
        global_model.load_state_dict(best_state)

    # ---- التقييم النهائي العالمي: مرة واحدة فقط، على Test المعزول ----
    test_X, test_y = _concat_loaders(test_loaders)
    global_test_metrics = evaluate_model(global_model, test_X, test_y, device, cfg.eval_batch_size)

    # ---- التخصيص + قياس مكسب التخصيص [D] ----
    personalized_rows = []
    for cid in range(n_clients):
        if loader_size(train_loaders[cid]) == 0 or loader_size(test_loaders[cid]) == 0:
            print(f"⚠️ العميل {cid}: بيانات تدريب أو اختبار محلية غير كافية — تخطي التخصيص.")
            continue

        cx = torch.cat([xb for xb, _ in test_loaders[cid]])
        cy = torch.cat([yb for _, yb in test_loaders[cid]])

        # 1) أداء النموذج العالمي على test المحلي (قبل التخصيص)
        m_global = evaluate_model(global_model, cx, cy, device, cfg.eval_batch_size)

        # 2) التخصيص: fine-tune على Train المحلي، اختبار على Test المحلي
        client_model = copy.deepcopy(global_model).to(device)
        opt = torch.optim.Adam(client_model.parameters(), lr=cfg.pers_lr)
        client_model.train()
        all_labels = torch.cat([yb for _, yb in train_loaders[cid]]).long()
        weights = calculate_local_class_weights(all_labels, device)
        for _ in range(cfg.pers_epochs):
            for xb, yb in train_loaders[cid]:
                xb, yb = xb.to(device), yb.to(device).long()
                opt.zero_grad()
                loss = F.cross_entropy(client_model(xb), yb, weight=weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(client_model.parameters(), cfg.grad_clip)
                opt.step()

        m_pers = evaluate_model(client_model, cx, cy, device, cfg.eval_batch_size)
        personalized_rows.append({
            'client_id': cid, 'n_test_samples': int(len(cy)),
            # [V3-5] macro-F1 مضلّل لعميل بصنف واحد — نميّزه صراحة
            'has_both_classes': bool((cy == 0).any() and (cy == 1).any()),
            **{k: v for k, v in m_pers.items()},
            'global_f1_on_client': m_global['f1'],
            'pers_gain_f1': m_pers['f1'] - m_global['f1'],
        })

    pdf = pd.DataFrame(personalized_rows)
    # [V3-5] العدالة تُحسب على العملاء ثنائيي الأصناف؛ أحاديو الصنف
    # يُقرَّرون منفصلين بالـ Accuracy/FPR (لأن macro-F1 لهم مضلّل)
    both = pdf[pdf['has_both_classes']] if 'has_both_classes' in pdf else pdf
    single = pdf[~pdf['has_both_classes']] if 'has_both_classes' in pdf else pdf.iloc[0:0]
    base = both if len(both) else pdf
    fairness = {
        'n_clients_evaluated': int(len(pdf)),
        'n_clients_both_classes': int(len(both)),
        'mean_f1': base['f1'].mean(), 'std_f1': base['f1'].std(),
        'worst_f1': base['f1'].min(), 'best_f1': base['f1'].max(),
        'gap': base['f1'].max() - base['f1'].min(),
        'mean_acc': base['acc'].mean(), 'mean_prec': base['prec'].mean(),
        'mean_rec': base['rec'].mean(), 'mean_fpr': base['fpr'].mean(),
        'mean_pers_gain_f1': base['pers_gain_f1'].mean(),   # [D]
        'worst_pers_gain_f1': base['pers_gain_f1'].min(),
        'single_class_mean_acc': single['acc'].mean() if len(single) else float('nan'),
        'single_class_mean_fpr': single['fpr'].mean() if len(single) else float('nan'),
    }
    if len(single):
        print(f"ℹ️ عملاء بصنف اختبار واحد (تُستثنى من إحصاءات F1 للعدالة): "
              f"{single['client_id'].tolist()} — يُقرَّرون بالـ Accuracy/FPR فقط.")

    # ---- زمن الاستدلال وسعة المعالجة وحمل الاتصال ----
    global_model.eval()
    dummy = torch.randn(1, input_dim).to(device)
    with torch.no_grad():
        for _ in range(200):  # إحماء
            _ = global_model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(2000):
            _ = global_model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    elapsed = time.time() - t0
    latency_us = (elapsed / 2000.0) * 1e6
    throughput = 1.0 / (elapsed / 2000.0)

    param_size = sum(p.nelement() * p.element_size() for p in global_model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in global_model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    avg_participants = (len(active_pool) if cfg.participation_fraction >= 1.0
                        else cfg.participation_fraction * len(active_pool))
    data_per_round_mb = 2 * model_size_mb * avg_participants
    total_overhead_gb = (data_per_round_mb * completed_rounds) / 1024

    return {
        'algo': algo, 'seed': seed, 'total_params': total_params,
        'best_round': best_round, 'completed_rounds': completed_rounds,
        'history': history, 'global_test': global_test_metrics,
        'personalized_df': pdf, 'fairness': fairness,
        'latency_us': latency_us, 'throughput': throughput,
        'model_size_mb': model_size_mb, 'total_overhead_gb': total_overhead_gb,
        'final_state': global_model.state_dict(),
    }


# ============================================================================
# CELL 7 — تشغيل متعدد الخوارزميات × الـ seeds + التحليل الإحصائي + الحفظ
# ============================================================================
def run_all_experiments(input_dim, train_loaders, val_loaders, test_loaders, cfg: FLConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2, default=str)

    all_rows, raw_results = [], {}
    for algo in ('fedavg', 'fedprox', 'feddyn'):
        for seed in cfg.seeds:
            print(f"\n===== تشغيل {algo.upper()} | seed={seed} =====")
            res = run_federated_experiment(
                algo=algo, input_dim=input_dim,
                train_loaders=train_loaders, val_loaders=val_loaders,
                test_loaders=test_loaders, cfg=cfg, seed=seed,
            )
            raw_results[(algo, seed)] = res

            if cfg.save_checkpoints:  # [G]
                ckpt = os.path.join(cfg.output_dir, f"best_{algo}_seed{seed}.pt")
                torch.save(res['final_state'], ckpt)

            res['personalized_df'].to_csv(
                os.path.join(cfg.output_dir, f"personalized_{algo}_seed{seed}.csv"), index=False)

            all_rows.append({
                'algo': algo, 'seed': seed,
                'best_round': res['best_round'], 'completed_rounds': res['completed_rounds'],
                **{f"global_{k}": v for k, v in res['global_test'].items()},
                **{f"pers_{k}": v for k, v in res['fairness'].items()},
                'latency_us': res['latency_us'], 'throughput': res['throughput'],
                'overhead_gb': res['total_overhead_gb'],
            })

    results_df = pd.DataFrame(all_rows)
    csv_path = os.path.join(cfg.output_dir, "federated_results_all_seeds.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"💾 تم حفظ النتائج الكاملة إلى: {csv_path}")
    return results_df, raw_results


def summarize_and_test(results_df: pd.DataFrame, metric='global_f1',
                       proposed='feddyn', baseline='fedprox'):
    """Mean±Std + paired t-test + Wilcoxon [H] بين المقترح وأقوى baseline."""
    num_cols = [c for c in results_df.columns if c not in ('algo', 'seed')]
    summary = results_df.groupby('algo')[num_cols].agg(['mean', 'std'])
    print("\n📊 ملخص Mean ± Std عبر كل الـ seeds:")
    print(summary)

    prop = results_df[results_df.algo == proposed].sort_values('seed')[metric].values
    base = results_df[results_df.algo == baseline].sort_values('seed')[metric].values

    if len(prop) == len(base) and len(prop) >= 3:
        t_stat, p_t = ttest_rel(prop, base)
        print(f"\n🧪 Paired t-test على '{metric}' ({proposed} مقابل {baseline}): "
              f"t={t_stat:.3f}, p={p_t:.4f}")
        try:
            w_stat, p_w = wilcoxon(prop, base)
            print(f"🧪 Wilcoxon signed-rank: W={w_stat:.3f}, p={p_w:.4f}")
        except ValueError as e:
            print(f"⚠️ Wilcoxon غير ممكن: {e}")
        print("⚠️ مع عدد seeds قليل (<5) القوة الإحصائية محدودة — لا تكتبي "
              "'دلالة إحصائية' إلا مع p<0.05 وعدد seeds كافٍ (5 فأكثر يُفضَّل).")
    else:
        print("⚠️ لا يمكن إجراء الاختبارات (عدد seeds غير متطابق أو أقل من 3).")
    return summary


# ============================================================================
# CELL 8 — الرسوم البيانية
# ============================================================================
def plot_convergence(raw_results: dict, cfg: FLConfig, seed_to_plot=None):
    seed_to_plot = seed_to_plot or cfg.seeds[0]
    plt.figure(figsize=(7, 4.5))
    colors = {'fedavg': 'gray', 'fedprox': 'darkblue', 'feddyn': 'darkred'}
    for (algo, seed), res in raw_results.items():
        if seed != seed_to_plot:
            continue
        plt.plot(res['history']['f1'], label=algo.upper(), color=colors.get(algo))
    plt.xlabel("Round"); plt.ylabel("Validation F1-Score")
    plt.title(f"Convergence Comparison (seed={seed_to_plot})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, 'convergence_comparison.png'), dpi=300)
    plt.show()


def plot_personalization_fairness(raw_results: dict, cfg: FLConfig, seed_to_plot=None):
    """Boxplot لتوزيع F1 المخصّص عبر العملاء لكل خوارزمية — يوضح العدالة."""
    seed_to_plot = seed_to_plot or cfg.seeds[0]
    rows = []
    for (algo, seed), res in raw_results.items():
        if seed != seed_to_plot:
            continue
        for _, r in res['personalized_df'].iterrows():
            rows.append({'algo': algo.upper(), 'f1': r['f1'],
                         'pers_gain_f1': r['pers_gain_f1']})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(data=df, x='algo', y='f1', ax=axes[0])
    axes[0].set_title(f'Per-Client Personalized F1 (seed={seed_to_plot})')
    sns.boxplot(data=df, x='algo', y='pers_gain_f1', ax=axes[1])
    axes[1].axhline(0, color='grey', linestyle='--')
    axes[1].set_title('Personalization Gain (F1 after − before)')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, 'personalization_fairness.png'), dpi=300)
    plt.show()


def plot_confusion_roc(model, X_test, y_test, device, algo_name: str,
                       cfg: FLConfig, cmap='Blues'):
    model.eval()
    loader = DataLoader(TensorDataset(X_test, y_test), batch_size=512, shuffle=False)
    all_probs, all_preds, all_trues = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            all_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
            all_preds.append(torch.argmax(logits, dim=1).cpu())
            all_trues.append(yb)

    probs_np = torch.cat(all_probs).numpy()
    preds_np = torch.cat(all_preds).numpy()
    trues_np = torch.cat(all_trues).numpy()

    cm = confusion_matrix(trues_np, preds_np, labels=[0, 1])
    fpr, tpr, _ = roc_curve(trues_np, probs_np)
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes[0],
                xticklabels=['Benign (0)', 'Attack (1)'],
                yticklabels=['Benign (0)', 'Attack (1)'])
    axes[0].set_title(f'Confusion Matrix — {algo_name}')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

    axes[1].plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.4f}')
    axes[1].plot([0, 1], [0, 1], linestyle='--', color='grey')
    axes[1].set_title(f'ROC Curve — {algo_name}')
    axes[1].set_xlabel('False Positive Rate'); axes[1].set_ylabel('True Positive Rate')
    axes[1].legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, f'{algo_name.lower()}_confusion_roc.png'), dpi=300)
    plt.show()



# ============================================================================
# CLI ENTRY POINT — run the full FedAvg / FedProx / FedDyn comparison
# ============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and compare FedAvg / FedProx / FedDyn on CIC-ToN-IoT (double non-IID)."
    )
    parser.add_argument("--data-path", type=str, default="data/CIC-ToN-IoT-V2.parquet",
                        help="Path to the CIC-ToN-IoT parquet file (see README for source).")
    parser.add_argument("--output-dir", type=str, default="outputs/fl_outputs_v2",
                        help="Directory for results, checkpoints, and plots.")
    parser.add_argument("--rounds", type=int, default=80, help="Federated communication rounds.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience (rounds).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024],
                        help="Random seeds to repeat each algorithm with.")
    parser.add_argument("--top-k", type=int, default=27, help="Number of MI-selected features.")
    parser.add_argument("--total-rows", type=int, default=500_000,
                        help="Stratified sample size drawn from the full dataset before splitting.")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5,
                        help="Dirichlet concentration parameter for the non-IID client split.")
    parser.add_argument("--max-client-train-samples", type=int, default=60_000,
                        help="Cap on any single client's training set size (prevents one client "
                             "from dominating weighted aggregation).")
    args = parser.parse_args()

    cfg = FLConfig(
        file_path=args.data_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        total_rows=args.total_rows,
        dirichlet_alpha=args.dirichlet_alpha,
        rounds=args.rounds,
        patience=args.patience,
        max_client_train_samples=args.max_client_train_samples,
        seeds=tuple(args.seeds),
    )
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 1) Data: leakage-free preprocessing (minutes, not hours)
    data = prepare_iot_dataset(cfg)
    input_dim = len(data["model_feature_idx"])

    # 2) Client partitioning (double non-IID: protocol groups + Dirichlet skew)
    train_idx, groups = partition_clients_protocol_dirichlet(
        data["X_train"], data["y_train"], data["selected_features"],
        alpha=cfg.dirichlet_alpha, seed=cfg.split_seed, client_groups=cfg.client_groups,
        min_class_quota=cfg.min_class_quota_train)
    val_idx, _ = partition_clients_protocol_dirichlet(
        data["X_val"], data["y_val"], data["selected_features"],
        alpha=cfg.dirichlet_alpha, seed=cfg.split_seed, client_groups=cfg.client_groups,
        min_class_quota=cfg.min_class_quota_eval)
    test_idx, _ = partition_clients_protocol_dirichlet(
        data["X_test"], data["y_test"], data["selected_features"],
        alpha=cfg.dirichlet_alpha, seed=cfg.split_seed, client_groups=cfg.client_groups,
        min_class_quota=cfg.min_class_quota_eval)

    # 3) Per-client DataLoaders
    fi = data["model_feature_idx"]
    train_loaders, train_stats = build_client_loaders(
        data["X_train"], data["y_train"], train_idx, fi,
        batch_size=cfg.batch_size, shuffle=True, seed=cfg.split_seed,
        max_samples=cfg.max_client_train_samples)
    val_loaders, val_stats = build_client_loaders(
        data["X_val"], data["y_val"], val_idx, fi,
        batch_size=cfg.eval_batch_size, shuffle=False, seed=cfg.split_seed)
    test_loaders, test_stats = build_client_loaders(
        data["X_test"], data["y_test"], test_idx, fi,
        batch_size=cfg.eval_batch_size, shuffle=False, seed=cfg.split_seed)

    # Mandatory diagnostic check before trusting any downstream result
    diagnose_client_splits(train_stats, test_stats)

    # 4) Run FedAvg / FedProx / FedDyn across all seeds, save CSV + checkpoints
    results_df, raw_results = run_all_experiments(
        input_dim, train_loaders, val_loaders, test_loaders, cfg)

    # 5) Summary statistics + paired significance tests (FedDyn vs. strongest baseline)
    summarize_and_test(results_df, metric="global_f1", proposed="feddyn", baseline="fedprox")

    # 6) Plots
    plot_convergence(raw_results, cfg)
    plot_personalization_fairness(raw_results, cfg)

    print(f"\n\u2705 Done. Results, checkpoints, and plots saved under: {cfg.output_dir}")


if __name__ == "__main__":
    main()
