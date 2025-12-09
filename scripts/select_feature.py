import argparse
from collections import Counter
from email import parser
import os
import shlex
import subprocess
from isoelectric import ipc
from joblib import load
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



project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "utils"))

from seq_translation import *



def normalize_aa_category_values_by_max(aa_categories, max_category_value):

    """
    Normalizes amino acid category indices to [0, 1] range by dividing by max category value.

    In addition to the 20 standard amino acids, the mapping may include:
      - 'U': the non-standard amino acid selenocysteine.
      - 'X': used as a padding token.

    Args:
        aa_categories (dict[int, list[str] or str]):
            A mapping from category index to one or more amino acid letters.
            Example:
                {
                    0: 'X',
                    1: ["A", "I", "L", "V"],
                    2: ['N','Q']
                }
        max_category_value (int):
            The highest category index used for normalization.

    Returns:
        dict[str, float]:
            A dictionary mapping each amino acid (e.g., "A") to its
            normalized score (category_value / max_category_value),
            rounded to 2 decimals.
    """
    normalized_values = {}
    for cat, aas in aa_categories.items():
        if isinstance(aas, list):
            for aa in aas:
                normalized_values[aa] = round((cat / max_category_value),2)
        else:
            normalized_values[aas] = round((cat / max_category_value),2)
    return normalized_values


def min_max_scale(values):

    """
    Apply Min–Max normalization to a list of numeric values.
    Scales values into the range [0, 1].
    This normalization is suitable for data that may include negative values.
    """
    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        return [0 for _ in values]
    
    return [min(round((v - min_val) / (max_val - min_val), 2), 1) for v in values]


def normalize_aa_categories_globally(aa_categories):

    """
    Globally normalizes amino acid values across all categories using Min–Max scaling.
    This function collects all amino acid values from every category into a single list,
    applies a shared Min–Max normalization, and maps the normalized values back to each
    amino acid. As a result, values from different categories become directly comparable
    on the same [0, 1] scale.

    Args:
        aa_categories (dict[int, dict[str, float]]):
            Mapping from category IDs to amino acid property dictionaries.
            Example:
                {
                    0: {'X': -5.5},
                    1: {'A': 1.8, 'C': 2.5, 'I': 4.5},
                    2: {'G': -0.4, 'H': -3.2}
                }
    Returns:
        dict[str, float]:
            A dictionary mapping amino acid letters to their globally normalized values.

    """

    aa_list = []
    for category in aa_categories.values():
        for aa, value in category.items():
            aa_list.append((aa, value))

    
    all_values = [v for _, v in aa_list]
    normalized_values = min_max_scale(all_values)

    normalized_aa_dict = {
        aa: norm_val for (aa, _), norm_val in zip(aa_list, normalized_values)
    }

    return normalized_aa_dict


def create_property_dict(phy_categories, hyd_categories, vol_categories, che_categories, 
                         cha_categories, hyd_don_categories, pol_categories, aac_categories, size_categories):

    """
    Creates a combined amino acid property dictionary.
    The function normalizes amino acid values across nine property categories.
    Some categories use max-based normalization, while others use global Min–Max
    scaling. The final output maps each amino acid to a list of normalized
    property values in a fixed feature order.

    Args:
        phy_categories (dict): Physicochemical category mapping.
        hyd_categories (dict): Hydropathy category mapping.
        vol_categories (dict): Volume category mapping.
        che_categories (dict): Chemical category mapping.
        cha_categories (dict): Charge category mapping.
        hyd_don_categories (dict): Hydrogen-donor category mapping.
        pol_categories (dict): Polarity category mapping.
        aac_categories (dict): Amino acid composition category mapping.
        size_categories (dict): Molecular size category mapping.

    Returns:
        dict[str, list[float]]: Mapping of amino acids to normalized
        feature lists in the following order:
            [
                physicochemical,
                hydropathy,
                volume,
                chemical,
                charge,
                hydrogen_donor,
                polarity,
                aa_composition,
                size
            ]
    """
    phy_values = normalize_aa_category_values_by_max(phy_categories, 12)
    
    hyd_values = normalize_aa_categories_globally(hyd_categories)
    vol_values = normalize_aa_categories_globally(vol_categories)

    che_values = normalize_aa_category_values_by_max(che_categories, 8)
    cha_values = normalize_aa_category_values_by_max(cha_categories, 4)
    hyd_don_values = normalize_aa_category_values_by_max(hyd_don_categories, 5)
    pol_values = normalize_aa_category_values_by_max(pol_categories, 3)
    aac_values = normalize_aa_category_values_by_max(aac_categories, 21)
    size_values = normalize_aa_categories_globally(size_categories)

    
    properties = {}
    for aa in set(phy_values.keys()) | set(hyd_values.keys()) | set(vol_values.keys()) | set(che_values.keys()) | set(cha_values.keys()) | set(hyd_don_values.keys()) | set(pol_values.keys()) | set(aac_values.keys())| set(size_values.keys()):
        properties[aa] = [phy_values.get(aa, 0), hyd_values.get(aa, 0), vol_values.get(aa, 0), che_values.get(aa, 0), cha_values.get(aa, 0), hyd_don_values.get(aa, 0), pol_values.get(aa, 0), aac_values.get(aa, 0), size_values.get(aa, 0)]
    
    return properties

# physicochemical
phy_categories = {
    0: 'X',
    1: ['A','I','L','V'],
    2: ['N','Q'],
    3: ['C','M'],
    4: ['S','T'],
    5: ['R','H','K'],
    6: ['D','E'],
    7: 'F',
    8: 'W',
    9: 'Y',
    10: 'P',
    11: 'G',
    12: 'U'
}


# hydropathy
hyd_categories = {
    0: {'X': -5.5},
    1: {'A': 1.8, 'C': 2.5, 'I': 4.5, 'L': 3.8, 'M': 1.9, 'F': 2.8, 'W': -0.9, 'V': 4.2},
    2: {'G': -0.4, 'H': -3.2, 'P': -1.6, 'S': -0.8, 'T': -0.7 ,'Y': -1.3},
    3: {'R': -4.5, 'N': -3.5, 'D': -3.5, 'Q': -3.5, 'E': -3.5, 'K': -3.9},
    4: {'U': 4.7} # max = 4.7 - (-5.5) = 10.2
}

# volume
vol_categories = {
    0: {'X': 40.0},
    1: {'A': 88.6, 'G': 60.1, 'S': 89.0},
    2: {'N': 114.1, 'D': 111.1, 'C': 108.5, 'P': 112.7, 'T': 116.1},
    3: {'Q': 143.8, 'E': 138.4, 'H': 153.2, 'V': 140.0},
    4: {'R': 173.4, 'I': 166.7, 'L': 166.7, 'K': 168.6, 'M': 162.9},
    5: {'F': 189.9, 'W': 227.8, 'Y': 193.6},
    6: {'U': 235.5} # max = 235.5 - 40 = 195.5
}

# chemical
che_categories = {
    0: 'X',
    1: ['A','G','I','L','P','V'],
    2: ['F','W','Y'],
    3: ['C','M'],
    4: ['S','T'],
    5: ['R','H','K'],
    6: ['D','E'],
    7: ['N','Q'],
    8: 'U'
}

#charge
cha_categories = {
    0: 'X',
    1: ['R','H','K'],
    2: ['D','E'],
    3: ['A','N','C','Q','G','I','L','M','F','P','S','T','W','Y','V'],
    4: 'U',
}

# Hydrogen donor or acceptor atoms
hyd_don_categories = {
    0: 'X',
    1: ['R','K','W'],
    2: ['D','E'],
    3: ['N','Q','H','S','T','Y'],
    4: ['A','C','G','I','L','M','F','P','V'],
    5: 'U'
}

# polarity
pol_categories = {
    0: 'X',
    1: ['R','N','D','Q','E','H','K','S','T','Y'],
    2: ['A','C','G','I','L','M','F','P','W','V'],
    3: 'U'
}


# amino acid composition
aac_categories = {
    0: 'X',
    1: 'A',
    2: 'R',
    3: 'N',
    4: 'D',
    5: 'C',
    6: 'Q',
    7: 'E',
    8: 'G',
    9: 'H',
    10: 'I',
    11: 'L',
    12: 'K',
    13: 'M',
    14: 'F',
    15: 'P',
    16: 'S',
    17: 'T',
    18: 'W',
    19: 'Y',
    20: 'V',
    21: 'U',
}

# size
size_categories = {
    0: {'X': 55},
    1: {'A': 89, 'C': 121, 'I': 131, 'L': 131, 'M': 149, 'F': 165, 'W': 204, 'V': 117},
    2: {'G': 75, 'H': 155, 'P': 115, 'S': 105, 'T': 119 ,'Y': 181},
    3: {'R': 174, 'N': 132, 'D': 133, 'Q': 146, 'E': 147, 'K': 146},
    4: {'U': 212} # max = 212 - 55 = 157
}

# secondary structure
sec_categories = {
    1: 'E',
    2: 'H',
    3: 'C',
}



def predict_s4pred_structure(fasta_file):

    """
    Predicts secondary structure for multiple protein sequences using S4Pred.
    
    The function extracts and returns three types of information from the
    S4Pred formatted output:
        - Conf: Confidence score for each residue position
        - Pred: Predicted secondary structure (H, E, C)
        - AA:   Original amino acid sequence
    
    Args:
        fasta_file (list[list]): 
            A list of sequence entries from loadfasta(), where each entry contains:
            [name (str), iseq (list[int]), seq (str)].

    tuple: A tuple containing:
            conf_list (list[str]): Confidence scores for each sequence
            pred_list (list[str]): Predicted secondary structures for each sequence
            sequence_list (list[str]): Original amino acid sequences

    """

    from run_model import predict_sequence, format_horiz
    
    conf_list = []  
    pred_list = []  
    sequence_list = []  

    for seq in fasta_file:
        
        sequence_id = seq[0]  
        encoded_seq_for_model = seq[1] 
        original_aa_sequence = seq[2]  
        # ss -> predicted structure (H, E, C), ss_conf -> confidence scores
        ss, ss_conf = predict_sequence([sequence_id, encoded_seq_for_model, original_aa_sequence])

        formatted_prediction = format_horiz(seq, ss, ss_conf)

        confidence_parts = []
        structure_parts = []
        aa_parts = []

        for line in formatted_prediction:
            line = line.strip()
            if line.startswith('Conf:'):
                confidence_parts.append(line.split('Conf:')[1].strip())
            elif line.startswith('Pred:'):
                structure_parts.append(line.split('Pred:')[1].strip())
            elif line.startswith('AA:'):
                aa_parts.append(line.split('AA:')[1].strip())


        conf_list.append(''.join(confidence_parts))
        pred_list.append(''.join(structure_parts))
        sequence_list.append(''.join(aa_parts))
        

    return conf_list, pred_list, sequence_list




def normalize_overall_values_by_max(values, max_val):
    """
    Normalizes values to [0, 1] range for overall sequence calculations.

    Args:
        values (list): Numeric values to normalize.
        max_val (float): Maximum value for normalization.

    Returns:
        list: Normalized values in [0, 1] range, capped at 1.0. 
              Returns zeros if max_val is 0.
    """
    if max_val == 0:
        return [0 for _ in values]
    return [min(v / max_val, 1) for v in values]


def normalize_overall_values_min_max_with_fixed_range(values, min_val, max_val):
    """
    Performs min-max normalization using fixed min and max values for overall sequence calculations.

    Args:
        values (list): 
            Numeric values to normalize.
        min_val (float): 
            Fixed minimum value (from training set).
        max_val (float): 
            Fixed maximum value (from training set).

    Returns:
        list: Normalized values in [0, 1] range, capped at 1.0. 
              Returns zeros if min_val == max_val.
    """

    if max_val == min_val:
        return [0 for _ in values]
    return [min((v - min_val) / (max_val - min_val), 1) for v in values]




def prepare_feature_encodings():
    """
    Initializes amino acid property dictionaries and structure score mappings.
    
    This function creates the lookup tables needed for feature extraction.
    Call once at program start.
    
    Returns:
        dict: Dictionary containing:
              sec_values (dict): Secondary structure scores for each amino acid.
              properties (dict): Physicochemical properties for each amino acid.
              amino_acids (str): Standard 21 amino acid letters (20 standard + U).
    """
    sec_values = normalize_aa_category_values_by_max(sec_categories, 3)


    properties = create_property_dict(
        phy_categories, hyd_categories, vol_categories,
        che_categories, cha_categories, hyd_don_categories,
        pol_categories, aac_categories, size_categories
    )

    amino_acids = 'ACDEFGHIKLMNPQRSTVWYU'

    return {
        'sec_values': sec_values,
        'properties': properties,
        'amino_acids': amino_acids
    }


def encode_sequence(sequence, scores, properties):

    """
    Encodes a protein sequence into a local feature vector.
    For each amino acid, the feature vector contains:
        - Physicochemical property values (from properties dict)
        - Corresponding secondary structure score

    Args:
        sequence (str): Protein sequence (amino acid letters).
        scores (list): Secondary structure scores for each residue.
        properties (dict): Mapping of amino acids to their property vectors.

    Returns:
        numpy.ndarray: Flattened feature vector for the entire sequence.
    """
    feature_vector = []
    feature_len = len(next(iter(properties.values()))) # Get feature vector length from first amino acid properties
    
    for aa, score in zip(sequence, scores):
        if aa in properties:
            feature_vector.extend(properties[aa])
        else:
            feature_vector.extend([0] * feature_len)
        feature_vector.append(score)     
    return np.array(feature_vector)


def compute_overall_features(sequence, amino_acids, overall_params):
    """
    Computes overall feature vector for a protein sequence.

    This function calculates amino acid composition, isoelectric point,
    hydrophobicity (GRAVY score), and sequence length features from a
    protein sequence.

    Args:
        sequence (str):
            Protein sequence (amino acid letters).
        amino_acids (List[str]): 
            List of standard amino acids for frequency calculation.
        overall_params (Dict[str, float]): 
            Dictionary containing normalization parameters with
            keys 'iso_max', 'gravy_min', and 'gravy_max'.

    Returns:
        List[float]: Feature vector containing:
            - 21 amino acid frequencies (normalized, 20 standard + U)
            - Normalized isoelectric point
            - Normalized GRAVY score (hydrophobicity)
            - Normalized sequence length (divided by 99)
            Total dimensionality: 24 features.
    """

    # Remove unknown residues 'X' for feature calculation
    seq_no_x = sequence.replace('X', '')
    seq_len_no_x = len(seq_no_x)

    # Calculate amino acid composition frequencies
    aa_count = Counter(seq_no_x)
    aa_freq_vector = [aa_count.get(aa, 0) / seq_len_no_x for aa in amino_acids]
    aa_freq_vector = [round(v, 2) for v in aa_freq_vector]

    # Calculate isoelectric point and normalize to [0, 1]
    iso = ipc.predict_isoelectric_point(seq_no_x)
    iso_value = round(
        normalize_overall_values_by_max([iso], overall_params["iso_max"])[0], 2
    )

    # Calculate GRAVY score (Grand Average of Hydropathy) and normalize
    gravy = peptides.Peptide(seq_no_x).hydrophobicity(scale="KyteDoolittle")
    gravy_value = round(
        normalize_overall_values_min_max_with_fixed_range(
            [gravy],
            overall_params["gravy_min"],
            overall_params["gravy_max"]
        )[0], 2
    )

    # Normalize sequence length (max length assumed to be 99)
    len_value = round(seq_len_no_x / 99, 2)

    return aa_freq_vector + [iso_value, gravy_value, len_value]


def build_feature_matrix_from_fasta(fasta_file, overall_params, feature_encodings, s4pred_path):
    """
    Reads FASTA file and converts to feature matrix.

    Args:
        fasta_file (str): 
            Path to FASTA file.
        overall_params (Dict[str, float]): 
            Fixed parameters needed for normalizing overall sequence features.
        feature_encodings (Dict[str, Union[Dict, str]]): Feature encoding lookup tables 
            (prepared by prepare_feature_encodings()).
    Returns:
        tuple: A 3-element tuple (sequence_ids, sequences, feature_matrix).
            sequence_ids: List of sequence ID strings.
            sequences: List of protein sequence strings.
            feature_matrix: NumPy array of shape (n_sequences, n_features).
    """

    sys.path.insert(0, s4pred_path)

    importlib.invalidate_caches()
    original_argv = sys.argv.copy()
    sys.argv = ["run_model.py", fasta_file, "--outfmt", "horiz"]

    from utilities import loadfasta
    from run_model import predict_sequence, format_horiz

    sys.argv = original_argv
    sequences = list(SeqIO.parse(fasta_file, 'fasta'))

    # S4Pred secondary structure prediction
    fasta_data = loadfasta(fasta_file)
    conf_scores, sec_structs, aa_sequence = predict_s4pred_structure(fasta_data)

    sec_values = feature_encodings['sec_values']
    properties = feature_encodings['properties']
    amino_acids = feature_encodings['amino_acids']

    id_list, protein_seqs = [], []
    aa_features, overall_features = [], []

    for seq_record, conf, pred, aa in zip(sequences, conf_scores, sec_structs, aa_sequence):
        seq_id = seq_record.id
        sequence = str(aa)

        # Secondary structure scores
        sec_score_list = [sec_values.get(p, 0) for p in pred]

        # Amino acid features
        aa_feat_vector = encode_sequence(sequence, sec_score_list, properties)
        aa_features.append(aa_feat_vector)

        # Overall sequence features
        overall_feat_vector = compute_overall_features(sequence, amino_acids, overall_params)
        overall_features.append(overall_feat_vector)

        id_list.append(seq_id)
        protein_seqs.append(sequence)

    aa_features = np.array(aa_features)
    overall_features = np.array(overall_features)

    # Concatenate amino acid and overall features to form feature matrix
    all_features_vectors = np.concatenate((aa_features, overall_features), axis=1)

    return id_list, protein_seqs, all_features_vectors

def is_dna(seq):
    """
    Checks if the input sequence is a DNA sequence.
    """
    return set(str(seq.upper())) <= {'A', 'T', 'C', 'G'}


def clean_and_normalize_sequences(sequences):

    """
    Cleans and normalizes DNA/protein sequences to fixed length.
    
    Args:
        sequences (Iterable[SeqRecord]):
            Iterable of BioPython SeqRecord objects containing DNA or protein sequences.
    
    Returns:
        List[Tuple[str, str]]: List of tuples containing:
            - Sequence ID (str)
            - Normalized protein sequence of length 99 (str)

    Raises:
        ValueError: If protein sequence length exceeds 100 amino acids.

    Warnings:
        Prints warnings for:
            - DNA sequences not divisible by 3
            - Internal stop codons in DNA
            - Missing terminal stop codon in DNA
    """
    FIXED_LENGTH = 99
    cleaned_data = []
    all_warn_msgs = []

    for record in sequences:
        seq = str(record.seq)
        warn_msgs = []

        if is_dna(seq):
            # Check if DNA length is divisible by 3
            if len(seq) % 3 != 0:
                warn_msgs.append(f"[WARNING] {record.id} - DNA sequence length is not divisible by 3, translation may be inaccurate.")

            protein_seq = translation(seq)

            # Check for internal stop codons (potential multiple ORFs)
            if "*" in protein_seq[:-1]:
                warn_msgs.append(f"[WARNING] {record.id} - DNA sequence contains internal stop codons, indicating potential additional ORFs.")

            # Remove terminal stop codon if present
            if protein_seq.endswith("*"):
                protein_seq = protein_seq.rstrip("*")
            else:
                warn_msgs.append(f"[WARNING] {record.id} - DNA sequence does not end with a stop codon.")

        else:  # Protein input
            protein_seq = seq
            if "*" in seq:
                protein_seq = seq.rstrip("*")

        # Truncate if length is exactly 100 (remove last amino acid)
        if len(protein_seq) == 100:
            protein_seq = protein_seq[:-1]
        elif len(protein_seq) > 100:
            raise ValueError(
                f"[Error] {record.id} - Protein sequence length is {len(protein_seq)}, which exceeds the 100 amino acid limit."
            )

        # Pad with 'X' to reach fixed length of 99
        if len(protein_seq) < FIXED_LENGTH:
            protein_seq += "X" * (FIXED_LENGTH - len(protein_seq))
        

        cleaned_data.append((record.id, protein_seq))
        all_warn_msgs.extend(warn_msgs)

    for msg in all_warn_msgs:
        print(msg)

    return cleaned_data


def convert_to_fasta_str(id_seq_pairs):
    """
    Converts (ID, sequence) pairs into a FASTA-formatted string.
    """
    fasta_io = StringIO()
    records = []

    for id, seq in id_seq_pairs:
        seq_obj = Seq(seq)
        record = SeqRecord(seq_obj, id=id)
        records.append(record)

    SeqIO.write(records, fasta_io, 'fasta')
    return fasta_io.getvalue()


def save_temp_fasta(fasta_str):
    """
    Saves a FASTA string as a temporary file and returns the file path.
    """
    temp = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.fasta')
    temp.write(fasta_str)
    temp.close()
    return temp.name
