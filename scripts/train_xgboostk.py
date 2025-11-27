import pickle
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.calibration import cross_val_predict
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_squared_error, precision_score, recall_score, roc_auc_score, roc_curve, root_mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier
from select_feature_github import build_feature_matrix_from_fasta, prepare_feature_encodings, clean_and_normalize_sequences, convert_to_fasta_str, save_temp_fasta
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
import warnings
import os
from lazypredict.Supervised import LazyClassifier
import warnings
import lightgbm as lgb
import optuna.trial as trial
from optuna.samplers import TPESampler
import optuna
from optuna.visualization import plot_optimization_history
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from optuna.visualization import plot_param_importances
import optuna.visualization as vis
from joblib import dump, load
import shap
import lightgbm as lgb
from sklearn.metrics import RocCurveDisplay, auc
import sys
from pathlib import Path
from sklearn.utils.validation import check_is_fitted
from operator import itemgetter
import argparse
from Bio import SeqIO
from datetime import datetime




from sequence_io import project_root, read_fasta, write_fasta, get_file_path
sys.path.insert(0, os.path.join(project_root, "utils"))


random.seed(1) 


def split_dataset(id_aa_pairs, all_features_vectors, label_df, test_ratio=0.2):
    """
    Split dataset into training and testing sets using stratified sampling 
    to maintain class balance between positive and negative samples.

    Parameters
    ----------
    id_aa_pairs : list of tuple
        List of (sequence_id, amino_acid_sequence).
    all_features_vectors : list of np.ndarray
        List of feature vectors (each corresponding to one sequence).
    label_df : dataFrame
        DataFrame containing 'id' and 'label' columns (aligned with the FASTA entries).
    test_ratio : float, optional
        Proportion of the dataset to include in the test split (default: 0.2).

    Returns
    -------
    X_train : np.ndarray
        Training feature vectors.
    y_train : list
        Training labels.
    X_test : np.ndarray
        Testing feature vectors.
    y_test : list
        Testing labels.
    test_ids : list
        IDs of sequences in the test set.
    test_seqs : list
        Corresponding amino acid sequences in the test set.

    Example
    -------
    Input: 2180 samples ({0:1090, 1:1090}), test_ratio=0.2  
    Output:
        Train: 1744 samples ({0:872, 1:872})
        Test :  436 samples ({0:218, 1:218})
    """

    seq_ids, seq_seqs = zip(*id_aa_pairs) 
    feature_df = pd.DataFrame(all_features_vectors)
    feature_df.columns = feature_df.columns.astype(str)  # 🔹 把欄名全轉字串
    feature_df['id'] = seq_ids
    feature_df['sequence'] = seq_seqs

    # 對齊標籤
    df = feature_df.merge(label_df, on='id', how='inner')


    # 保證 train/test 類別比例一致
    train_df, test_df = train_test_split(
        df, test_size=test_ratio, stratify=df['label'], random_state=1
    )


    # 保留特徵
    drop_cols = ['label', 'id', 'sequence']
    train_set, test_set = train_df.drop(columns=drop_cols), test_df.drop(columns=drop_cols)
    train_labels, test_labels = train_df['label'].tolist(), test_df['label'].tolist()
    test_ids, test_seqs = test_df['id'].tolist(), test_df['sequence'].tolist()



    return train_set, train_labels, test_set, test_labels, test_ids, test_seqs


def build_labeled_dataset(fasta_path, label_csv, overall_params, feature_encodings):
    """
    Build a labeled dataset.

    This function reads a FASTA file and a label CSV file, extracts sequence features,
    and aligns them by their IDs. If any IDs do not match between the FASTA and label file,
    only the common entries will be kept.

    Args:
        fasta_path : str
            Path to the FASTA file containing protein sequences.
        label_csv : str
            Path to the label CSV file containing 'id' and 'label' columns.
        overall_params : dict
            Global configuration parameters for feature extraction, including
            normalization min/max values saved from the training phase.
        feature_encodings : dict
            Feature encoding lookup tables (from prepare_feature_encodings()).

    Returns:
        id_aa_pairs : list of tuple
            List of (sequence_id, amino_acid_sequence).
        all_features_vectors : list of list
            Extracted feature vectors for each sequence.
        label_df : dataFrame
            DataFrame containing 'id' and 'label' columns (aligned with the FASTA entries).

    Notes:
        - Automatically keeps only IDs present in both FASTA and label files.
        - Feature vectors remain in the same order as id_aa_pairs.
    """
    sequences = list(SeqIO.parse(fasta_path, 'fasta'))
    padded_sequences = clean_and_normalize_sequences(sequences)
    
    fasta_str = convert_to_fasta_str(padded_sequences)
    temp_fasta_path = save_temp_fasta(fasta_str)

    # 特徵提取
    id_list, protein_seqs, all_features_vectors = build_feature_matrix_from_fasta(
        temp_fasta_path, overall_params, feature_encodings
    )
    id_aa_pairs = list(zip(id_list, protein_seqs))

    # 讀取標籤
    label_df = pd.read_csv(label_csv)
    if not {'id', 'label'}.issubset(label_df.columns):
        raise ValueError('The label file must contain the columns: id, label.')

    # 檢查 fasta 和 label id 是否一致
    fasta_ids = set(id_list)
    label_ids = set(label_df['id'])
    common_ids = fasta_ids & label_ids


    if fasta_ids != label_ids:
        missing_in_fasta = label_ids - fasta_ids
        missing_in_label = fasta_ids - label_ids

        if missing_in_fasta:
            print(f'[Warning] There are {len(missing_in_fasta)} IDs in the label file that were not found in the FASTA file:')
            print('   └─ ' + ', '.join(sorted(missing_in_fasta)))
        if missing_in_label:
            print(f'[Warning] There are {len(missing_in_label)} IDs in the FASTA file that were not found in the label file:')
            print('   └─ ' + ', '.join(sorted(missing_in_label)))

        print(f'\n[Info] Automatically took the intersection — {len(common_ids)} entries retained.\n')

        # 依據交集 ID 過濾 FASTA、標籤與特徵資料
        filtered_ids = [seq_id for seq_id in id_list if seq_id in common_ids]

        filtered_pairs = [(seq_id, seq) for seq_id, seq in id_aa_pairs if seq_id in common_ids]

        filtered_features = []
        for (seq_id, _), vec in zip(id_aa_pairs, all_features_vectors):
            if seq_id in common_ids:
                filtered_features.append(vec)

        filtered_labels = label_df[label_df['id'].isin(common_ids)].reset_index(drop=True)

        id_list = filtered_ids
        id_aa_pairs = filtered_pairs
        all_features_vectors = filtered_features
        label_df = filtered_labels


    return id_aa_pairs, all_features_vectors, label_df




optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_lightgbm(trial):
    X_train, X_valid, y_train, y_valid = train_test_split(train_set, train_labels, test_size=0.2,random_state=1)
    lgb_param={
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "num_leaves": trial.suggest_int('num_leaves', 10, 300),  
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),  
        "max_depth": trial.suggest_int("max_depth", 3, 12),  
        "min_child_samples":trial.suggest_int("min_child_samples", 10, 20),
        "min_child_weight":trial.suggest_float("min_child_weight", 1e-3, 3.0, log=True),
        "subsample":trial.suggest_float("subsample", 0.7, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 0, 3),  
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),  
        'verbosity': -1,
        'random_state': 1
    }
    lgb_model = lgb.LGBMClassifier(**lgb_param, verbose=-1)
    lgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    accuracy = lgb_model.score(X_valid, y_valid)
    return accuracy


def save_optuna_results(study, trial_history, filename):
    """
    儲存 Optuna 調參歷史與最佳結果
    """
    
    base_dir = os.path.join(project_root, 'best_score')
    os.makedirs(base_dir, exist_ok=True)

    results_path = os.path.join(base_dir, filename)


    history_df = pd.DataFrame(trial_history)
    with open(results_path, 'w') as results_file:
        results_file.write('Trial_Number Best_Score\n')
        history_df.to_csv(results_file, sep=' ', header=False, index=False)

        results_file.write("\n=== Best Trial ===\n")
        results_file.write(f"Trial Number: {study.best_trial.number}\n")
        results_file.write(f"Best Score: {study.best_trial.value:.6f}\n")
        results_file.write("Best Params:\n")
        for k, v in study.best_trial.params.items():
            results_file.write(f"  {k}: {v}\n")




def run_optuna(objective_func, n_trials, model_type=None):
    """
    跑 Optuna 調參，回傳包含歷史與最佳參數的 study 物件
    """
    trial_history = []
    best_trial_number = 0
    best_score_so_far = float('-inf') # 用負無限大初始化，確保第一個 trial 一定能更新

    def objective(trial):
        """
        Optuna 正式目標函式，用來呼叫指定模型的調參邏輯
        """
        nonlocal best_trial_number, best_score_so_far
        trial_score = objective_func(trial)

        if trial_score > best_score_so_far:
            best_trial_number = trial.number
            best_score_so_far = trial_score

        trial_history.append({
            'Trial_Number': int(best_trial_number),
            'Best_Score': best_score_so_far
        })
        return trial_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{model_type}_optuna_{timestamp}.txt"

    
    save_optuna_results(study, trial_history, filename=filename)
    return study


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    計算分類模型的多種評估指標：
    F1、Accuracy、AUC、Sensitivity、Specificity
    """

    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    # AUC 若提供預測機率且非 NaN
    if y_prob is not None and not np.all(np.isnan(y_prob)):
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = np.nan

    sensitivity = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = np.nan  # 當分母為 0 時，避免除以零錯誤
        
    return f1, acc, auc, sensitivity, specificity

def get_scores(model, X, cv=None, y=None):
    """
    取得模型的預測分數：
    - 若指定 cv 與 y，執行交叉驗證並回傳每筆樣本的預測分數
    - 若未指定 cv，則直接對輸入資料（如測試集）做預測

    args：
        model : 已訓練的模型
        X     : 特徵矩陣
        y     : 標籤
        cv    : 若提供，執行 cross_val_predict（交叉驗證）

    回傳：
        array 或 None — 模型預測分數（越大代表越可能是正類）
    """

    # 使用交叉驗證情況
    if cv is not None and y is not None:
        if hasattr(model, "predict_proba"):
            return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        elif hasattr(model, "decision_function"):
            return cross_val_predict(model, X, y, cv=cv, method="decision_function")
        else:
            return None
    # 不使用交叉驗證，直接整體預測(test set)
    else:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            return model.decision_function(X)
        else:
            return None


def train_and_evaluate_models(train_set, train_labels, test_set, test_labels, tuned_models, run_baseline=True):
    """Train tuned models + optional LazyPredict baseline models."""

    X_train, X_val, y_train, y_val = train_test_split(
        train_set, train_labels, test_size=0.2, random_state=1
    )

    model_dict = {}

    # === LazyPredict baseline ===
    if run_baseline:
        
        try:
            clf = LazyClassifier(verbose=0, ignore_warnings=True)
            models, predictions = clf.fit(X_train, X_val, y_train, y_val)
            print(f"LazyPredict finished: {len(models)} models trained.")
        except Exception as e:
            print(f"LazyPredict failed: {e}")
            models = pd.DataFrame()

        if models.empty:
            print("No valid baseline results generated.")
        else:
            model_dict.update(clf.models)

    # === Tuned models ===
    for name, model in tuned_models.items():
        model.fit(X_train, y_train)
        model_dict[name] = model

    # === 評估 ===
    evaluation_results = []
    for classifier_name, model_wrapper in model_dict.items():
        model = getattr(model_wrapper, 'model', model_wrapper)

        # Train CV
        y_pred_train = cross_val_predict(model, X_train, y_train, cv=5)
        y_prob_train = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else None
        f1_train, acc_train, auc_train, sens_train, spec_train = compute_metrics(
            y_train, y_pred_train, y_prob_train
        )

        # Independent Test
        y_pred_test = model.predict(test_set)
        y_prob_test = model.predict_proba(test_set)[:, 1] if hasattr(model, "predict_proba") else None
        f1_test, acc_test, auc_test, sens_test, spec_test = compute_metrics(
            test_labels, y_pred_test, y_prob_test
        )

        evaluation_results.append({
            'model': classifier_name,
            'train_f1': f1_train,
            'train_acc': acc_train,
            'train_auc': auc_train,
            'train_sens': sens_train,
            'train_spec': spec_train,
            'test_f1': f1_test,
            'test_acc': acc_test,
            'test_auc': auc_test,
            'test_sens': sens_test,
            'test_spec': spec_test,
        })

    if len(evaluation_results) == 0:
        print("No valid model results generated.")
        return pd.DataFrame()

    return pd.DataFrame(evaluation_results)



def save_results_report(results_df, sort_key='test_auc', save_path=None, model_type=None):
    """Save formatted evaluation report to file."""

    if results_df.empty:
        print('No results to save — DataFrame is empty.')
        return

    if save_path is None:
        save_path = os.path.join(project_root, 'results_baseline')
    os.makedirs(save_path, exist_ok=True)

    if sort_key not in results_df.columns:
        print(f"sort_key '{sort_key}' not found. Available columns: {results_df.columns.tolist()}")
        sort_key = results_df.columns[0]

    results_sorted = results_df.sort_values(by=sort_key, ascending=False).to_dict('records')


    output_lines = []
    for i, res in enumerate(results_sorted, 1):
        classifier_name = res['model']
        # 每個模型自己判斷 baseline / tuned（顯示用）
        is_tuned = 'optuna' in classifier_name.lower()
        status = 'Optuna tuned' if is_tuned else 'Baseline'
        rank_label = f'#{i}  ' if len(results_sorted) > 1 else ''

        block = [
            f"\n{rank_label}{classifier_name}  ({status})",
            "-" * 60,
            "Train (Cross-validated)",
            f"  F1 Score     : {res['train_f1']:.4f}",
            f"  Accuracy     : {res['train_acc']:.4f}",
            f"  AUC          : {res['train_auc']:.4f}",
            f"  Sensitivity  : {res['train_sens']:.4f}",
            f"  Specificity  : {res['train_spec']:.4f}",
            "\nIndependent Test",
            f"  F1 Score     : {res['test_f1']:.4f}",
            f"  Accuracy     : {res['test_acc']:.4f}",
            f"  AUC          : {res['test_auc']:.4f}",
            f"  Sensitivity  : {res['test_sens']:.4f}",
            f"  Specificity  : {res['test_spec']:.4f}",
            "-" * 60
        ]
        output_lines.append('\n'.join(block))

    output_text = '\n'.join(output_lines)
    print(output_text)


    # 儲存
    if model_type:
        # 有指定 model_type 時，使用帶 timestamp 的檔名（通常是 tuned 模型）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f'{model_type}_model_evaluation_{timestamp}.txt'
    else:
        # 沒有 model_type 時，使用通用檔名（通常是 baseline）
        filename = 'all_model_evaluation.txt'

    text_file = os.path.join(save_path, filename)

    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(output_text)

    print(f"\nResults saved to: {text_file}")


# lgb_tuned_model = model_dict["LightGBM_tuned"]
# xgb_tuned_model = model_dict["XGBoost_tuned"]
# ada_tuned_model = model_dict["AdaBoost_tuned"]
# rf_tuned_model = model_dict["RandomForest_tuned"]
# et_tuned_model = model_dict["ExtraTrees_tuned"]



# dump(lgb_tuned_model,'../save_models/LightGBM_model.txt')
# dump(xgb_tuned_model,'../save_models/XGBoost_model.joblib')
# dump(ada_tuned_model, '../save_models/AdaBoost_model.joblib')
# dump(rf_tuned_model, '../save_models/RandomForest_model.joblib')
# dump(et_tuned_model, '../save_models/ExtraTrees_model.joblib')



# xgb_model = load('../save_models/XGBoost_model.joblib')
# lgb_model = load('../save_models/LightGBM_model.txt')
# ada_model = load('save_models/AdaBoost_model.joblib')
# rf_model = load('save_models/RandomForest_model.joblib')
# ns_model = load('save_models/NuSVC_model.joblib')


# from lazy_predict_v1 import test_set,test_labels
# from create_comparison_tools_testing import 

# y_pred = xgb_model.predict(test_set)
# y_prob = xgb_model.predict_proba(test_set)[:, 1]


# accuracy = accuracy_score(test_labels, y_pred)
# f1 = f1_score(test_labels, y_pred)
# auc = roc_auc_score(test_labels, y_prob)

# conf_matrix = confusion_matrix(test_labels, y_pred)
# tn, fp, fn, tp = conf_matrix.ravel()

# sensitivity = tp / (tp + fn)  
# specificity = tn / (tn + fp)  

# print(f"準確率 (ACC): {accuracy:.4f}")
# print(f"F1 分數: {f1:.4f}")
# print(f"AUC: {auc:.4f}")
# print(f"敏感性 (Recall): {sensitivity:.4f}")
# print(f"特異性: {specificity:.4f}")


def check_n_jobs(n_jobs):
    if n_jobs == -1:
        return cpu_count()
    elif n_jobs < 1:
        print(f'Invalid n_jobs={n_jobs}, using 1 instead.')
        return 1
    return n_jobs



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Train machine learning models for small protein prediction.\n'
            'This script supports both baseline training and parameter-optimized modes, '
            'and outputs training/testing scores for model performance comparison.' 
        ),
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=40)
    )

    
    # === 模式選擇 ===
    parser.add_argument('--mode', type=str, choices=['auto', 'manual'], default='auto', metavar='MODE',
                        help='auto: auto split train/test | manual: use pre-split data')
    
    # === Auto mode 參數 ===
    parser.add_argument('--input', type=str, metavar='FILE', help='Path to the input FASTA file (used in auto mode)')
    parser.add_argument(
        '--label_csv', type=str,
        metavar='FILE',
        help=(
            'Label CSV file (used in auto mode)\n'
            'Should contain at least two columns:\n'
            '  - id: sequence ID (must match FASTA headers)\n'
            '  - label: class label (e.g., 0 or 1)'
        )
    )
    parser.add_argument(
        '--test_ratio', type=float, default=0.2,
        metavar='RATIO',
        help=(
            'Proportion of data used for testing in auto mode (0–1)\n'
            'Default: 0.2 (i.e., 20%% for testing, 80%% for training)'
        )
    )
    
    # === Manual mode 參數 ===
    parser.add_argument('--train_fasta', type=str, metavar='FILE', help='Path to the training FASTA file (used in manual mode)')
    parser.add_argument('--train_label_csv', type=str, metavar='FILE', help='Path to the training label CSV file (used in manual mode)')
    parser.add_argument('--test_fasta', type=str, metavar='FILE', help='Path to the testing FASTA file (used in manual mode)')
    parser.add_argument('--test_label_csv', type=str, metavar='FILE', help='Path to the testing label CSV (used in manual mode)')
    
    # === 共用參數 ===
    parser.add_argument('--tune', action='store_true', help='Enable Optuna hyperparameter tuning (optional)')
    parser.add_argument('--n_trials', type=int, default=300, metavar='N', help='Number of Optuna optimization trials (used when --tune is enabled)')
    parser.add_argument('--run_baseline', action='store_true', help='Run LazyPredict baseline models')
    parser.add_argument("--n_jobs", type=int, default=1, metavar='NUM_CORES',
                    help="Number of CPU cores to use (use -1 for all cores)")

    parser.add_argument('--save_model', action='store_true',
                    help='Save the trained model (only saves tuned model if --tune is enabled)')
    
    args = parser.parse_args()               

    n_jobs = check_n_jobs(args.n_jobs)
    
    # === 參數驗證 ===
    if args.mode == 'auto':
        if not args.input or not args.label_csv:
            parser.error("--mode auto requires both --input and --label_csv")
    elif args.mode == 'manual':
        if not all([args.train_fasta, args.train_label_csv, args.test_fasta, args.test_label_csv]):
            parser.error("--mode manual requires --train_fasta, --train_label_csv, --test_fasta, and --test_label_csv")

    # === 檢查檔案存在性（執行階段）===
    if args.mode == 'auto':
        file_list = [args.input, args.label_csv]
    elif args.mode == 'manual':
        file_list = [args.train_fasta, args.train_label_csv, args.test_fasta, args.test_label_csv]

    # 統一檢查缺失檔案
    missing_files = [f for f in file_list if not os.path.isfile(f)]
    if missing_files:
        sys.exit(
            "[Error] Missing files:\n" +
            "\n".join(f" - {os.path.abspath(f)}" for f in missing_files)
        )


    # === Step 1: 特徵生成 ===
    overall_params = load(get_file_path('overall_params.pkl'))
    feature_encodings = prepare_feature_encodings()

    print('\n' + '='*60)
    if args.mode == 'auto':
        print('[Mode: auto] Splitting dataset into train/test')
        print('='*60)

        id_aa_pairs, all_features_vectors, label_df = build_labeled_dataset(
            args.input, args.label_csv, overall_params, feature_encodings
        )

        train_set, train_labels, test_set, test_labels, test_ids, test_seqs = split_dataset(
            id_aa_pairs, all_features_vectors, label_df, test_ratio=args.test_ratio
        )

    elif args.mode == 'manual':
        print('[Mode: manual] Using user-provided train/test datasets')
        print('='*60)

        # === 處理 Train set ===
        train_pairs, train_vec, train_label_df = build_labeled_dataset(
            args.train_fasta, args.train_label_csv, overall_params, feature_encodings
        )
        
        train_set = pd.DataFrame(train_vec)
        train_labels = train_label_df['label'].tolist()
        train_ids = train_label_df['id'].tolist()
        

        # === 處理 Test set ===
        test_pairs, test_vec, test_label_df = build_labeled_dataset(
            args.test_fasta, args.test_label_csv, overall_params, feature_encodings
        )
        

        test_set = pd.DataFrame(test_vec)
        test_labels = test_label_df['label'].tolist()
        test_ids = test_label_df['id'].tolist()
        test_seqs = [seq for _, seq in test_pairs]
        


    # === Step 3: Baseline (LazyPredict) ===
    if args.run_baseline:
        print('\nRunning baseline models...')
        baseline_df = train_and_evaluate_models(
            train_set, train_labels, test_set, test_labels, 
            tuned_models={}, run_baseline=True
        )
        save_results_report(baseline_df, sort_key='test_auc', 
                       save_path=get_file_path('results_baseline'))
        print('Baseline done.\n')
        
        if not args.tune:
            print('Baseline only mode: skipping LightGBM training.')
            sys.exit(0)

    # === Step 4: Optuna 調參 ===
    if args.tune:
        print(f'\nRunning Optuna tuning ({args.n_trials} trials)...')
        study = run_optuna(objective_lightgbm, n_trials=args.n_trials, model_type='lightgbm')
        best_params = study.best_trial.params
    else:
        print('Using fixed optimized parameters (from previous tuning).')
        best_params = {
            'n_estimators': 870,
            'learning_rate': 0.01954138735720925,
            'max_depth': 6,
            'num_leaves': 192,
            'min_child_samples': 15,
            'min_child_weight': 0.05620211863643883,
            'subsample': 0.7247653405678081,
            'subsample_freq': 1,
            'colsample_bytree': 0.8032160595365208,
            'random_state': 1,
            'verbose': -1
        }

    # === Step 5: 訓練模型 ===
    tuned_models = {'LGBMClassifier_optuna': lgb.LGBMClassifier(**best_params)}
    print('\nTraining LGBMClassifier with Optuna best parameters...')
    tuned_df = train_and_evaluate_models(train_set, train_labels, test_set, test_labels, tuned_models, run_baseline=False)
    save_results_report(tuned_df, sort_key='test_auc', 
                   save_path=get_file_path('results_tuned'),
                   model_type='lightgbm')

    # === 新增模型存檔 ===
    if args.save_model:

        base_dir = os.path.join(project_root, 'save_models')
        os.makedirs(base_dir, exist_ok=True)

        model_type = 'lightgbm'

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        save_path = f"{base_dir}/{model_type}_model_{timestamp}.joblib"

        print(f"\nSaving model to: {save_path}")
        classifier_name = list(tuned_models.keys())[0]
        dump(tuned_models[classifier_name], save_path)
        print("Model saved.")

    print('\nLGBMClassifier training complete.')


#     estimator = DecisionTreeClassifier(
#         max_depth=5, 
#         min_samples_split= 11, 
#         min_samples_leaf= 6, 
#         max_features= 'sqrt', 
#         criterion= 'entropy', 
#         random_state=1
#     )

#     tuned_models = {
#         'LightGBM_tuned': lgb.LGBMClassifier(
#             # 要調參傳入 **best_params 就可以了
#             n_estimators= 870,
#             learning_rate= 0.01954138735720925,
#             max_depth= 6,
#             num_leaves= 192,
#             min_child_samples= 15,
#             min_child_weight= 0.05620211863643883,
#             subsample= 0.7247653405678081,
#             subsample_freq= 1,
#             colsample_bytree= 0.8032160595365208,
#             random_state=1,
#             verbose=-1
#         ),
#         'XGBoost_tuned': XGBClassifier(
#             n_estimators= 508,
#             learning_rate= 0.020874111994222226,
#             max_depth= 9,
#             reg_alpha= 0.668054555692508,
#             reg_lambda= 0.07805161320175043,
#             min_child_weight= 7,
#             gamma= 1.8523253993611626,
#             subsample= 0.8117152339795167,
#             colsample_bytree= 0.8792570655370885,
#             objective ='binary:logistic',
#             eval_metric='logloss',     # 建議加這個避免 warning
#             random_state=1,
#             n_jobs=-1
#         ),
#         'RandomForest_tuned': RandomForestClassifier(
#             n_estimators= 878,
#             max_depth= 6,
#             min_samples_split= 12,
#             min_samples_leaf= 5,
#             max_features= 0.6,
#             bootstrap= True,
#             random_state=1,
#             n_jobs=-1
#         ),
#         'AdaBoost_tuned': AdaBoostClassifier(
#             estimator=estimator,
#             n_estimators= 461,
#             learning_rate= 0.04274484403814473,
#             random_state=1
#         ),
#         'ExtraTrees_tuned': ExtraTreesClassifier(
#             n_estimators= 229,
#             max_depth= 19,
#             max_features= 0.7,
#             random_state=1,
#             n_jobs=-1
#         )
#     }