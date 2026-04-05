import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.calibration import cross_val_predict
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from select_feature import build_feature_matrix_from_fasta, prepare_feature_encodings, clean_and_normalize_sequences, convert_to_fasta_str, save_temp_fasta
from sklearn.model_selection import train_test_split
import os
from lazypredict.Supervised import LazyClassifier
import warnings
import lightgbm as lgbm
import optuna.trial as trial
import optuna
from joblib import dump, load
import sys
from pathlib import Path
import argparse
from Bio import SeqIO
from datetime import datetime



project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from sequence_io import read_fasta, write_fasta, get_file_path
from data import split_dataset, build_labeled_dataset
from training_utils import train_and_evaluate_models, save_results_report
from tuning import run_optuna, objective_lightgbm
from training_pipeline import parse_and_validate_args, load_auto_dataset, load_manual_dataset, load_datasets, run_training_pipeline, check_n_jobs


if __name__ == '__main__':

    random.seed(1) 

    args = parse_and_validate_args()
    n_jobs = check_n_jobs(args.n_jobs)
    

    default_params = {
        'objective': 'binary',
        'learning_rate': 0.0948986295292432,
        'n_estimators': 220,
        'num_leaves': 16,
        'max_depth': 7,
        'min_child_samples': 41,
        'colsample_bytree': 0.6374707667806355,
        'subsample': 0.687647357184827,
        'reg_alpha': 0.0016164194312992622,
        'reg_lambda': 2.9894222199228095,
        'random_state': 1,
        'verbose': -1
    }
    
    run_training_pipeline(
        args,
        model_class=lgbm.LGBMClassifier,
        default_params=default_params
    )

