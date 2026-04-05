import argparse
import os
import sys
import pandas as pd
from data import build_labeled_dataset, split_dataset, get_or_create_feature_params
from select_feature import prepare_feature_encodings, build_feature_matrix_from_fasta, clean_and_normalize_sequences, convert_to_fasta_str, save_temp_fasta
from joblib import dump
from sequence_io import get_file_path, project_root
from datetime import datetime
from training_utils import train_and_evaluate_models, save_results_report
from tuning import run_optuna
from multiprocessing import cpu_count



def parse_and_validate_args():
    """
    Parse and validate command-line arguments shared across all training scripts
    (used by all five executables).

    This function centralizes argument definitions and validation logic so that
    all training workflows (auto/manual) follow a consistent interface.
    Returns a populated argparse.Namespace object.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Train machine learning models for small protein prediction.\n'
            'This script supports both baseline training and parameter-optimized modes, '
            'and outputs training/testing scores for model performance comparison.' 
        ),
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=40)
    )

    
    # Select training mode (auto/manual)
    parser.add_argument('--mode', type=str, choices=['auto', 'manual'], default='auto', metavar='MODE',
                        help='auto: auto split train/test | manual: use pre-split data')
    
    # Arguments for auto mode
    parser.add_argument('--fasta', type=str, metavar='FILE', help='Path to the input FASTA file (used in auto mode)')
    parser.add_argument(
        '--label-csv', type=str,
        metavar='FILE',
        help=(
            'Label CSV file (used in auto mode)\n'
            'Should contain at least two columns:\n'
            '  - id: sequence ID (must match FASTA headers)\n'
            '  - label: class label (e.g., 0 or 1)'
        )
    )
    parser.add_argument(
        '--test-ratio', type=float, default=0.2,
        metavar='RATIO',
        help=(
            'Proportion of data used for testing in auto mode (0–1)\n'
            'Default: 0.2 (i.e., 20%% for testing, 80%% for training)'
        )
    )
    
    # Arguments for manual mode
    parser.add_argument('--train-fasta', type=str, metavar='FILE', help='Path to the training FASTA file (used in manual mode)')
    parser.add_argument('--train-label-csv', type=str, metavar='FILE', help='Path to the training label CSV file (used in manual mode)')
    parser.add_argument('--test-fasta', type=str, metavar='FILE', help='Path to the testing FASTA file (used in manual mode)')
    parser.add_argument('--test-label-csv', type=str, metavar='FILE', help='Path to the testing label CSV (used in manual mode)')
    
    # Arguments shared by both modes
    parser.add_argument('--tune', action='store_true', help='Enable Optuna hyperparameter tuning (optional)')
    parser.add_argument('--n-trials', type=int, default=300, metavar='N', help='Number of Optuna optimization trials (used when --tune is enabled)')
    parser.add_argument('--run-baseline', action='store_true', help='Run baseline model comparison using LazyPredict')
    parser.add_argument("--n-jobs", type=int, default=1, metavar='NUM_CORES',
                    help="Number of CPU cores to use (use -1 for all cores)")

    parser.add_argument('--save-model', action='store_true',
                    help='Save the trained model (only saves tuned model if --tune is enabled)')

    s4pred_path = os.path.join(project_root, 'tools/s4pred')
    parser.add_argument('--s4pred-path', type=str, default= s4pred_path, metavar='DIR',
                    help='Directory containing s4pred files')
    
    args = parser.parse_args()

    # Validate input arguments
    if args.mode == 'auto':
        if not args.fasta or not args.label_csv:
            parser.error('--mode auto requires both --fasta and --label-csv')
    elif args.mode == 'manual':
        if not all([args.train_fasta, args.train_label_csv, args.test_fasta, args.test_label_csv]):
            parser.error('--mode manual requires --train-fasta, --train-label_csv, --test-fasta, and --test-label-csv')

    # Verify that required files exist
    if args.mode == 'auto':
        file_list = [args.fasta, args.label_csv]
    elif args.mode == 'manual':
        file_list = [args.train_fasta, args.train_label_csv, args.test_fasta, args.test_label_csv]

    # Check for missing files
    missing_files = [f for f in file_list if not os.path.isfile(f)]
    if missing_files:
        sys.exit(
            '[Error] Missing files:\n' +
            '\n'.join(f' - {os.path.abspath(f)}' for f in missing_files)
        )

    # Validate s4pred path
    if not os.path.isdir(args.s4pred_path):
        sys.exit(f"[Error] s4pred-path does not exist: {os.path.abspath(args.s4pred_path)}")

    weights_dir = os.path.join(args.s4pred_path, "weights")
    if not os.path.isdir(weights_dir):
        sys.exit(f"[Error] Missing weights/ folder inside: {os.path.abspath(args.s4pred_path)}")

    return args



def load_auto_dataset(args, overall_params, feature_encodings):
    """"
    Builds a dataset from a single input and performs an internal train/test split.

    This mode is used when the user provides a single FASTA file and a
    corresponding label CSV. The function generates features for all samples
    and then splits the dataset into training and testing subsets using the
    provided test_ratio.

    Args:
        args (argparse.Namespace):
            Parsed command-line arguments. Required fields:
                fasta (str): Input FASTA file.
                label_csv (str): Label CSV file.
                test_ratio (float): Fraction of samples assigned to the test set.
        overall_params (dict):
            Global configuration parameters for feature extraction, including
            normalization min/max values saved from the training phase.
        feature_encodings (dict):
            Feature encoding lookup tables (from prepare_feature_encodings()).

    Returns:
        tuple: A tuple containing:
            train_set (np.ndarray): 
                Training feature matrix with shape (n_samples, n_features).
            train_labels (np.ndarray): 
                Training labels.
            test_set (np.ndarray): 
                Testing feature matrix with shape (n_samples, n_features).
            test_labels (np.ndarray): 
                Testing labels.
            test_ids (list[str]): 
                Sequence IDs in the test split.
            test_seqs (list[str]): 
                Amino acid sequences in the test split.
    """

    print('\n' + '='*60)
    print('[Mode: auto] Splitting dataset into train/test')
    print('='*60)

    # Build dataset from a single FASTA + label file
    id_aa_pairs, all_features_vectors, label_df = build_labeled_dataset(
        args.fasta,
        args.label_csv,
        overall_params,
        feature_encodings,
        args.s4pred_path
    )

    # Internal train/test split
    train_set, train_labels, test_set, test_labels, test_ids, test_seqs = \
        split_dataset(
            id_aa_pairs,
            all_features_vectors,
            label_df,
            test_ratio=args.test_ratio
        )

    return train_set, train_labels, test_set, test_labels, test_ids, test_seqs


def load_manual_dataset(args, overall_params, feature_encodings):
    """
    Loads train and test datasets separately using user-provided files.

    This mode is used when the user directly provides separate FASTA and CSV
    files for both training and testing data. Features are generated for each
    dataset independently, and no splitting is performed.

    Args:
        args (argparse.Namespace):
            Parsed command-line arguments. Required fields:
                train_fasta (str): Training FASTA file.
                train_label_csv (str): Training label CSV.
                test_fasta (str): Testing FASTA file.
                test_label_csv (str): Testing label CSV.
        overall_params (dict):
            Global configuration parameters for feature extraction, including
            normalization min/max values saved from the training phase.
        feature_encodings (dict):
            Feature encoding lookup tables (from prepare_feature_encodings()).

    Returns:
        tuple: A tuple containing:
            train_set (pd.DataFrame): 
                Training feature matrix with shape (n_samples, n_features).
            train_labels (list[int]): 
                Training labels.
            test_set (pd.DataFrame): 
                Testing feature matrix with shape (n_samples, n_features).
            test_labels (list[int]): 
                Testing labels.
            test_ids (list[str]): 
                Sequence IDs in the test split.
            test_seqs (list[str]): 
                Amino acid sequences in the test split.
    """

    print("\n" + "="*60)
    print("[Mode: manual] Using user-provided train/test datasets")
    print("="*60)

    # Build training set
    train_pairs, train_vec, train_label_df = build_labeled_dataset(
        args.train_fasta,
        args.train_label_csv,
        overall_params,
        feature_encodings,
        args.s4pred_path
    )
    train_set = pd.DataFrame(train_vec)
    train_labels = train_label_df['label'].tolist()

    # Build test set
    test_pairs, test_vec, test_label_df = build_labeled_dataset(
        args.test_fasta,
        args.test_label_csv,
        overall_params,
        feature_encodings,
        args.s4pred_path
    )
    test_set = pd.DataFrame(test_vec)
    test_labels = test_label_df['label'].tolist()
    test_ids = test_label_df['id'].tolist()
    test_seqs = [seq for _, seq in test_pairs]

    return train_set, train_labels, test_set, test_labels, test_ids, test_seqs


def load_datasets(args, overall_params, feature_encodings):
    """
    Dispatches dataset loading to auto or manual mode.

    This function chooses the correct dataset-loading routine based on
    args.mode. It does not perform dataset construction itself.

    Args:
        args (argparse.Namespace):
            Parsed arguments containing:
                mode (str): Either "auto" or "manual".
                fasta (str, optional): Used only in auto mode.
                label_csv (str, optional): Used only in auto mode.
                test_ratio (float, optional): Used only in auto mode.
                train_fasta (str, optional): Used only in manual mode.
                train_label_csv (str, optional): Used only in manual mode.
                test_fasta (str, optional): Used only in manual mode.
                test_label_csv (str, optional): Used only in manual mode.
        overall_params (dict):
            Global configuration parameters for feature extraction, including
            normalization min/max values saved from the training phase.
        feature_encodings (dict):
            Feature encoding lookup tables (from prepare_feature_encodings()).


    Returns:
        tuple: A tuple containing:
            load_manual_dataset(), containing:
                train_set (np.ndarray or pd.DataFrame):
                    Training feature matrix with shape (n_samples, n_features).
                    - np.ndarray in 'auto' mode
                    - pd.DataFrame in 'manual' mode
                train_labels (np.ndarray or list[int]):
                    Training labels. 
                    - np.ndarray in 'auto' mode 
                    - list[int] in 'manual' mode
                test_set (np.ndarray or pd.DataFrame): 
                    Testing feature matrix with shape (n_samples, n_features).
                    - np.ndarray in 'auto' mode
                    - pd.DataFrame in 'manual' mode
                test_labels (np.ndarray or list[int]): 
                    Testing labels.
                    - np.ndarray in 'auto' mode
                    - list[int] in 'manual' mode
                test_ids (list[str]): 
                    Sequence IDs in the test split.
                test_seqs (list[str]): 
                    Amino acid sequences in the test split.
    """

    if args.mode == 'auto':
        return load_auto_dataset(args, overall_params, feature_encodings)

    elif args.mode == 'manual':
        return load_manual_dataset(args, overall_params, feature_encodings)

    else:
        raise ValueError(f'Unknown mode: {args.mode}. Expected "auto" or "manual".')


def run_training_pipeline(
    args,
    model_class,                   
    default_params=None  
):
    """
    Executes the SPROTify training workflow from raw sequences to a production-ready model.

    Core & Optional Components:
        - Feature generation and setting up scaling rules (Min-Max values)
        - Baseline model evaluation (optional)
        - Optuna hyperparameter tuning (optional)
        - Model training using best parameters
        - Model saving (optional)

    Args:
        args (Namespace): 
            Command-line arguments controlling the pipeline behavior
            (e.g., whether to tune, run baseline, save model).
        model_class (type): 
            The model class to be used (e.g., LGBMClassifier). 
            Pass the class name itself, not a created model.
        default_params (dict, optional): Pre-defined optimized parameters to use
            when tuning is disabled.

    Returns:
        pandas.DataFrame: A DataFrame containing evaluation metrics (e.g., ACC, AUC) 
            for either the baseline models or the final optimized model.
    """

    model_name = model_class.__name__

    # ----- Feature Generation -----
    feature_encodings = prepare_feature_encodings()

    # Feature scaling (Min-Max values)
    overall_params, params_path = get_or_create_feature_params(args, feature_encodings)
    
    train_set, train_labels, test_set, test_labels, test_ids, test_seqs = \
        load_datasets(args, overall_params, feature_encodings)


    # ----- Baseline Evaluation (LazyPredict) -----
    if args.run_baseline:
        print('\nRunning baseline models...')
        baseline_df, _ = train_and_evaluate_models(
            train_set, train_labels,
            test_set, test_labels,
            tuned_models={}, run_baseline=True
        )
        save_results_report(
            baseline_df, sort_key='train_auc',
            save_path=get_file_path('results_baseline')
        )
        print('\nBaseline done.')

        if not args.tune:
            if args.save_model:
                print("\n[Warning] Baseline mode does not save models. --save_model will be ignored.")
            print('\nBaseline only mode: evaluation completed. Skipping hyperparameter tuning and final model training.')
            return baseline_df

    # ----- Hyperparameter Tuning (Optuna) -----
    if args.tune:

        print(f'\nRunning Optuna tuning ({args.n_trials} trials)...')
        study = run_optuna(
            model_name=model_name,
            train_set=train_set,
            train_labels=train_labels,
            n_trials=args.n_trials
        )
        
        best_model  = study.best_trial.user_attrs["trained_model"]

        model_key = f'{model_name}_optuna'
        print(f'\nTraining {model_name} with Optuna best parameters...')

    else:
        print('Using fixed optimized parameters (from previous tuning).')
        best_params = default_params

        best_model = model_class(**best_params)
        model_key = f'{model_name}'
        print(f'\nTraining {model_name} with the fixed parameters...')

    # ----- Final Model Training (using optimized hyperparameters) -----
    tuned_models = {
        model_key: best_model
    }

    tuned_df, final_trained_models = train_and_evaluate_models(
        train_set, train_labels,
        test_set, test_labels,
        tuned_models, run_baseline=False
    )

    save_results_report(
        tuned_df,
        sort_key=None,  # No sorting needed for a single model
        save_path=get_file_path('results_tuned'),
        model_name=model_key
    )

    # ----- Model Saving -----
    if args.save_model:
        base_dir = os.path.join(project_root, 'save_models')
        os.makedirs(base_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        save_path = f'{base_dir}/{model_key}_{timestamp}.joblib'

        final_model = final_trained_models[model_key]
        dump(final_model, save_path)
        print(f"\nSaved model to: {save_path}")

    print(f'\n{model_name} training complete.')

    return tuned_df


def check_n_jobs(n_jobs):
    """
    Normalize n_jobs:
    -1 → use all CPU cores
    <1 → fallback to 1
    otherwise return n_jobs itself
    """
    if n_jobs == -1:
        return cpu_count()
    elif n_jobs < 1:
        print(f'Invalid n_jobs={n_jobs}, using 1 instead.')
        return 1
    return n_jobs

