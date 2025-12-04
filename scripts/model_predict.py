import argparse
from collections import Counter
from email import parser
import os
import shlex
import subprocess
from isoelectric import ipc
from joblib import load, cpu_count
import numpy as np
import peptides
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
from Bio import SeqIO
import sys
from lazypredict.Supervised import LazyClassifier
from Bio.Seq import Seq
from io import StringIO
from Bio.SeqRecord import SeqRecord
import tempfile
import argparse
import importlib
from isoelectric import ipc


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, os.path.join(project_root, "utils"))
sys.path.insert(0, os.path.join(project_root, "tools/s4pred"))
from seq_translation import *
from sequence_io import project_root, read_fasta, write_fasta, get_file_path
from select_feature import *
from training_pipeline import check_n_jobs



def predict_model(model, input_file, output_file, overall_params, feature_encodings):
    """
    Performs prediction on FASTA sequences using a trained model.

    Reads FASTA sequences, normalizes them to fixed length, extracts features,
    and generates predictions with probability scores.

    Args:
        model (BaseEstimator):
            A trained classifier that implements predict() and predict_proba().
            Examples: LGBMClassifier, XGBClassifier, AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier.
        input_file (str): 
            Path to the input FASTA file.
        output_file (str): 
            Path to the output CSV file.
        overall_params (dict):
            Global configuration parameters for feature extraction, including
            normalization min/max values saved from the training phase.
        feature_encodings (dict):
            Feature encoding lookup tables (from prepare_feature_encodings()).

    Returns:
        None

    """
    # Read and normalize sequences from input FASTA file
    sequences = list(SeqIO.parse(input_file, 'fasta'))
    padded_sequences = clean_and_normalize_sequences(sequences)
    
    # Convert to FASTA format and save temporarily
    fasta_str = convert_to_fasta_str(padded_sequences)
    temp_fasta_path = save_temp_fasta(fasta_str)

    # Extract features from sequences
    sequence_ids, protein_seq, feature_matrix = build_feature_matrix_from_fasta(
        temp_fasta_path, overall_params, feature_encodings
    )

    # Predict using trained model
    y_pred = model.predict(feature_matrix)
    y_prob = model.predict_proba(feature_matrix)[:, 1]

    protein_seq = [seq.replace('X', '') for seq in protein_seq]

    sprotify_df = pd.DataFrame({
        'sequence_id': sequence_ids,
        'protein_seq': protein_seq,
        'class': y_pred,
        'prediction_probability': y_prob,
    })
    sprotify_df.to_csv(output_file, index=False)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description=(
        'SPROTify: a machine learning-based tool for small protein prediction.\n'
        'This script loads trained models to predict potential small proteins '
        'from input data and exports the results to a CSV file.'
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model-type', type=str, choices=['xgb', 'lgbm', 'ada', 'rf', 'et'], default='lgbm', metavar='TYPE', help='Model type to use (choices: xgb, lgbm, ada, rf or et)')
    parser.add_argument('--model-path', type=str, default='', metavar='FILE', help='Path to trained model file (optional; auto-select by model type if empty)')
    parser.add_argument('--input', type=str, required=True, metavar='FILE', help='Path to the input FASTA file')
    parser.add_argument('--output', type=str, required=True, metavar='FILE', help='Path to the output CSV file')
    parser.add_argument("--n_jobs", type=int, default=1, metavar='NUM_CORES',
                    help="Number of CPU cores to use (use -1 for all cores)")

    args = parser.parse_args()

    n_jobs = check_n_jobs(args.n_jobs)

    default_models = {
        'xgb': get_file_path('models/xgboost_model.joblib'),
        'lgbm': get_file_path('models/lightgbm_model.joblib'),
        'ada': get_file_path('models/adaboost_model.joblib'),
        'rf': get_file_path('models/randomforest_model.joblib'),
        'et': get_file_path('models/extratrees_model.joblib'),
    }

    if not args.model_path:
        default_path = default_models.get(args.model_type)
        args.model_path = get_file_path(default_path)

    if not os.path.isfile(args.input):
        sys.exit(f'[Error] Input file not found: {os.path.abspath(args.input)}')

    if not os.path.isfile(args.model_path):
        sys.exit(f'[Error] Model not found: {os.path.abspath(args.model_path)}')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print(f'Using model file: {args.model_path}')



    model = load(args.model_path)

    overall_params = load(get_file_path('overall_params.pkl'))
    
    feature_encodings = prepare_feature_encodings()
    
    predict_model(model, args.input, args.output, overall_params, feature_encodings)


