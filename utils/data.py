import pandas as pd
import os
from select_feature import build_feature_matrix_from_fasta, prepare_feature_encodings, clean_and_normalize_sequences, convert_to_fasta_str, save_temp_fasta
from sklearn.model_selection import train_test_split
from Bio import SeqIO
from sequence_io import project_root
from joblib import dump, load


def split_dataset(id_aa_pairs, all_features_vectors, label_df, test_ratio=0.2):
    """
    Split the dataset into training and testing sets using stratified sampling
    to preserve class balance between positive and negative samples.

    Args:
        id_aa_pairs (list[tuple]):
            A list of (sequence_id, amino_acid_sequence) pairs.
        all_features_vectors (array-like):
            A sequence of feature vectors extracted from each sequence.
        label_df (pandas.DataFrame):
            A DataFrame containing two required columns:
            - 'id' : sequence IDs
            - 'label' : class labels (e.g., 0/1)
        test_ratio (float, optional):
            Fraction of samples assigned to the test split.
            Defaults to 0.2.

    Returns:
        tuple: A 6-element tuple containing:
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

    Example:
        Suppose the dataset contains 2180 samples (positive: 1090, negative: 1090)
        with test_ratio = 0.2:

        - Training set → 1744 samples (872 positive, 872 negative)
        - Testing set  → 436 samples (218 positive, 218 negative)
    """

    seq_ids, seq_seqs = zip(*id_aa_pairs)

    feature_df = pd.DataFrame(all_features_vectors)
    feature_df.columns = feature_df.columns.astype(str)  # avoid issues with numeric column names in pandas operations
    feature_df["id"] = seq_ids
    feature_df["sequence"] = seq_seqs

    # Merge features with labels (inner join excludes unlabeled samples)
    df = feature_df.merge(label_df, on="id", how="inner")

    df["label"] = df["label"].astype(int)

    # Stratified split to maintain class ratio
    train_df, test_df = train_test_split(
        df, test_size=test_ratio, stratify=df["label"], random_state=1
    )

    # Drop non-feature columns
    drop_cols = ["label", "id", "sequence"]
    train_set = train_df.drop(columns=drop_cols).to_numpy()
    test_set = test_df.drop(columns=drop_cols).to_numpy()

    train_labels = train_df["label"].to_numpy()
    test_labels = test_df["label"].to_numpy()

    test_ids = test_df["id"].tolist()
    test_seqs = test_df["sequence"].tolist()

    return train_set, train_labels, test_set, test_labels, test_ids, test_seqs


def build_labeled_dataset(fasta_path, label_csv, overall_params, feature_encodings, s4pred_path):
    """
    Build a labeled dataset by extracting features from a FASTA file and aligning
    them with labels from a CSV file.

    This function reads protein sequences from a FASTA file, applies preprocessing
    and feature extraction, and merges them with label information provided in
    a CSV file. If some sequence IDs do not appear in both the FASTA and label
    file, only the intersection of IDs is retained.

    Args:
        fasta_path (str):
            Path to the FASTA file containing protein sequences.
        label_csv (str):
            Path to the label CSV file containing 'id' and 'label' columns.
        overall_params (dict):
            Global configuration parameters for feature extraction, including
            normalization min/max values saved from the training phase.
        feature_encodings (dict):
            Feature encoding lookup tables (from prepare_feature_encodings()).

    Returns:
        id_aa_pairs (list[tuple]):
            A list of (sequence_id, amino_acid_sequence) pairs.
        all_features_vectors (Sequence[np.ndarray]):
            A sequence of feature vectors extracted from each sequence.
        label_df (pandas.DataFrame):
            A DataFrame containing two required columns:
            - 'id' : sequence IDs
            - 'label' : class labels (e.g., 0/1)

    Notes:
        Only IDs present in both the FASTA and label CSV files are retained.
    """
    sequences = list(SeqIO.parse(fasta_path, 'fasta'))
    padded_sequences = clean_and_normalize_sequences(sequences)
    
    fasta_str = convert_to_fasta_str(padded_sequences)
    temp_fasta_path = save_temp_fasta(fasta_str)

    # Extract features from sequences
    id_list, protein_seqs, all_features_vectors = build_feature_matrix_from_fasta(
        temp_fasta_path, overall_params, feature_encodings, s4pred_path
    )
    id_aa_pairs = list(zip(id_list, protein_seqs))

    # Load labels
    label_df = pd.read_csv(label_csv)
    if not {'id', 'label'}.issubset(label_df.columns):
        raise ValueError('The label file must contain the columns: id, label.')

    # Validate ID consistency between FASTA and label files
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

        # Identify valid indices where sequences exist in both FASTA and label files
        valid_indices = [i for i, (sid, _) in enumerate(id_aa_pairs) if sid in common_ids]
    
        id_aa_pairs = [id_aa_pairs[i] for i in valid_indices]
        all_features_vectors = [all_features_vectors[i] for i in valid_indices]
        
        # Keep only IDs with corresponding sequences and reset the index
        label_df = label_df[label_df['id'].isin(common_ids)].reset_index(drop=True)

        label_df['label'] = label_df['label'].astype(int)

    return id_aa_pairs, all_features_vectors, label_df


def get_or_create_feature_params(args, feature_encodings):
    """
    Manages unified feature standards (Max/Min) for protein sequences.
    
    Ensures consistent feature scaling by performing a global scan of:
    - Amino acid composition 
    - Physicochemical properties (pi, hydrophobicity)
    - Sequence length
    
    Returns:
        overall_params (dict): Dictionary of global feature benchmarks.
        params_path (str): Storage path for the .pkl parameter file.
    """
    SPROTIFY_FILES = ['train_set.fasta', 'test_set.fasta', 'full_dataset.fasta']
    
    if args.mode == 'manual':
        input_file = args.train_fasta
        base_name = os.path.basename(input_file).split('.')[0]
        param_name = f"{base_name}_with_test_combined"
    else:
        input_file = args.fasta
        param_name = os.path.basename(input_file).split('.')[0]

    current_filename = os.path.basename(input_file)
    
    # Use pre-saved scaling values for SPROTify dataset
    if current_filename in SPROTIFY_FILES:
        params_path = os.path.join(project_root, 'overall_params.pkl')
    else:
        # Generating user-specific parameter configuration
        params_path = os.path.join(project_root, f'{param_name}_params.pkl')

    # Check for existing parameter files (.pkl)
    if os.path.exists(params_path):
        overall_params = load(params_path)
        return overall_params, params_path
    
    # Re-calculate if missing
    print(f"\n[Init] Feature parameters not found. Calculating global Max/Min for sequence features...")
    overall_params = {}
    
    if args.mode == 'manual':
        scaling_list = [args.train_fasta, args.test_fasta]
    else:
        scaling_list = [args.fasta]

    # Data fusion and preprocessing
    combined_sequences = []
    for s in scaling_list:
        if s and os.path.exists(s):
            combined_sequences.extend(list(SeqIO.parse(s, 'fasta')))
    
    padded_sequences = clean_and_normalize_sequences(combined_sequences)
    fasta_str = convert_to_fasta_str(padded_sequences)
    temp_fasta_path = save_temp_fasta(fasta_str)

    _ = build_feature_matrix_from_fasta(temp_fasta_path, overall_params, feature_encodings, args.s4pred_path)

    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    dump(overall_params, params_path)
    print(f"Saved global feature parameters to: {params_path}")
    
    return overall_params, params_path