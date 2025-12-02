# SPROTify

**SPROTify** is a machine learning (ML)–based tool designed to determine whether an amino acid sequence is a small protein.  
The tool integrates five classification models—**LGBMClassifier, BaggingClassifier, XGBClassifier, ExtraTreesClassifier, and SVC**—allowing users to choose a model according to their needs.  
It provides a command-line interface (CLI) that supports:

- **Small protein prediction** directly from amino acid sequences  
- **Model training**, including training, testing, and performance evaluation, with optional performance comparison across models  

As the biological functions of many small proteins remain insufficiently characterized, SPROTify helps researchers identify candidate sequences potentially associated with specific diseases or experimental conditions.


## Installation

**Requirements:** Python 3.9+ (tested on Python 3.10.12)
```bash
# clone the repository
git clone https://github.com/CSBM-Lab/SPROTify.git
cd SPROTify

# Install dependencies
pip install -r requirements.txt
```

## Input Format

Depending on the task, this tool supports **two input settings**:

1. **Prediction setting** – FASTA file only  
2. **Training & evaluation setting** – FASTA file with an additional label file

---

### Common Input (FASTA)

A FASTA file containing **protein or DNA sequences** is required for both modes.


### Prediction Setting (FASTA only)

Used for predicting whether the input sequences are small proteins.

### Training & Evaluation Setting (FASTA + Label File)

Used for model training, testing, and performance evaluation.

### FASTA Example

```text
>seq1
MTHHPISDHEATLRCWALGFYPVEITLTQ
>seq2
MAIRGPAALSAALYLH
```

### Label File Format (CSV)

```csv
id,label
seq1,positive
seq2,negative
```

| Column | Required | Description |
|--------|----------|-------------|
| `id`   |  Yes | Sequence identifier (must match FASTA headers) |
| `label`|  Yes | True labels for training/testing/evaluation |

### Notes
- When DNA sequences are provided, they will be **automatically translated into protein sequences** before analysis.
- If separate training and testing datasets are not provided, the tool will **automatically perform train/test splitting**.

## Usage

### Prediction

**Command:**
```bash
python script/sm_model_predict.py --input input.fasta --output output.csv
```

**Required Arguments:**

- `--input`: Path to the input FASTA file 
- `--output`: Path to the output CSV file 

**Optional Arguments:**

- `--model-type`: Model type to use (choices: `xgb`, `lgbm`(default), `ada`, `rf` or `et`)
- `--model-path`: Path to trained model file (auto-select by model type if empty)
- `--n_jobs`: Number of CPU cores to use (default: 1, use -1 for all cores)

### Model Selection

Available pre-trained models for prediction:

| Model Type | Default Path |
|------------|-------------|
| `xgb` | `pretrained_models/xgboost_model.joblib` |
| `lgbm` | `pretrained_models/lightgbm_model.joblib` |
| `ada` | `pretrained_models/adaboost_model.joblib` |
| `rf` | `pretrained_models/randomforest_model.joblib` |
| `et` | `pretrained_models/extratrees_model.joblib` |

### Examples

**Basic usage:**
```bash
# Default lightgbm model
python script/sm_model_predict.py --input dataset/test_set.fasta --output model_result/lgbm_testing.csv
```

**Try different models:**
```bash
# xgboost model
python script/sm_model_predict.py --model-type xgb --input dataset/test_set.fasta --output model_result/xgb_testing.csv

# adaboost model  
python script/sm_model_predict.py --model-type ada --input dataset/test_set.fasta --output model_result/ada_testing.csv
```

**Performance optimization:**
```bash
# Use all CPU cores for faster prediction
python script/sm_model_predict.py --model-type xgb --input dataset/test_set.fasta --output model_result/xgb_testing.csv --n_jobs -1
```

---

### Training & Evaluation

Train and evaluate models on your dataset.

**Training Modes:**
- `--mode auto` (default): Automatically split data into train/test sets
- `--mode manual`: Use pre-split datasets

**The repository contains five model implementations:**
- `script/train_lightgbm.py` (recommended, fastest)
- `script/train_xgboost.py`
- `script/train_adaboost.py`
- `script/train_randomforest.py`
- `script/train_extratrees.py`

### Notes
- The five training scripts share identical command-line arguments. To switch models, just change the script name - everything else stays the same.

---

#### Auto Mode ((Default))
Automatically splits your data into training (80%) and testing (20%) sets.

**Command:**
```bash
python script/train_lightgbm.py --input input.fasta --label_csv input_label.csv 
```

**Required Arguments:**

- `--input`: Path to the input FASTA file
- `--label_csv`: Path to the label CSV file with columns `id` and `label`

**Optional Arguments:**

- `--test_ratio`: Proportion of data for testing (default: 0.2)
- `--tune`: Enable Optuna hyperparameter tuning
- `--n_trials`: Number of Optuna optimization trials (default: 300, used when `--tune` is enabled)
- `--run_baseline`: Run LazyPredict baseline model comparison
- `--save_model`: Save the trained model (recommended)
- `--n_jobs`: Number of CPU cores to use (default: 1, use -1 for all cores)


**Basic usage:**
```bash
# Train and save LightGBM model
python script/train_lightgbm.py --input dataset/sequences.fasta --label_csv dataset/labels.csv --save_model
```

**Different models:**
```bash
# XGBoost
python script/xgb.py --input dataset/sequences.fasta --label_csv dataset/labels.csv --save_model

# AdaBoost
python script/ada.py --input dataset/sequences.fasta --label_csv dataset/labels.csv --save_model
```

**Advanced training:**
```bash
# Custom test split (70/30 instead of 80/20)
python script/lgbm.py --input dataset/sequences.fasta --label_csv dataset/labels.csv --test_ratio 0.3 --save_model

# Hyperparameter tuning with all CPU cores
python script/xgb.py --input dataset/sequences.fasta --label_csv dataset/labels.csv --tune --n_trials 100 --save_model --n_jobs -1

# Run baseline model comparison
python script/lgbm.py --input dataset/sequences.fasta --label_csv dataset/labels.csv --run_baseline --save_model

# Combine multiple options
python script/xgb.py --input dataset/sequences.fasta --label_csv dataset/labels.csv --test_ratio 0.25 --tune --n_trials 50 --save_model --n_jobs -1
```

---