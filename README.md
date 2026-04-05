<div align="center">
<img src="images/logo_mod.png" width="30%" />
</div>

# SPROTify

[![DOI](https://zenodo.org/badge/1105209892.svg)](https://doi.org/10.5281/zenodo.17982429)

SPROTify is a machine learning–based tool for accurate small-protein prediction using features derived from amino acid sequences and secondary structure information. Small proteins have emerged as important regulators in diverse biological processes, including signal transduction, metabolism, stress response, and disease progression. However, many remain unannotated or experimentally uncharacterized due to their short length and low abundance.
SPROTify is trained on a curated dataset of experimentally validated small proteins, with multiple algorithms assessed through 5-fold cross-validation and further optimized via hyperparameter tuning.
The tool integrates three classification models—LGBMClassifier, XGBClassifier, and AdaBoostClassifier—allowing users to select the most suitable model based on their analytical goals.
By enabling accurate identification of small proteins, SPROTify supports research into their potential roles in disease mechanisms, regulatory pathways, and condition-specific biological functions.

SPROTify includes two main modules:

- **Small protein prediction**: Predict small proteins directly from amino acid sequences. 
- **Model training**: Train and test models on custom datasets while evaluating and comparing performance across models.

## Installation

1. Download SPROTify and install required Python packages (Python 3.9+)

```bash
# Clone the repository
git clone https://github.com/CSBM-Lab/SPROTify.git
cd SPROTify
``` 
```bash
# Install dependencies with pip
pip3 install -r requirements.txt
```
*or*
```bash
# Install dependencies with conda environment (for Windows)
conda env create -f env_sprotify.yaml
conda activate sprotify
```

2. Install [s4pred](https://github.com/psipred/s4pred)

- You may use a pre-installed version of [s4pred](https://github.com/psipred/s4pred),
please ensure the **weights** folder containing five models exists within the [s4pred](https://github.com/psipred/s4pred) directory.

  **Note**: Replace the `s4pred/run_model.py` with modified version from `SPROTify/tools/run_model.py` (Backup the original version if needed.)

- If [s4pred](https://github.com/psipred/s4pred) is not already installed, use the following commands to install it:

```bash
git clone https://github.com/psipred/s4pred.git tools/s4pred
wget -P tools/s4pred http://bioinfadmin.cs.ucl.ac.uk/downloads/s4pred/weights.tar.gz
tar -xvzf tools/s4pred/weights.tar.gz -C tools/s4pred
```

3. SPROTify requires a modified version of [s4pred](https://github.com/psipred/s4pred).
After cloning the original repository, copy the patched file from `tools/` to overwrite the original.

```bash
cp tools/run_model.py tools/s4pred/
```

## Input file

SPROTify supports input in FASTA format, compatible with both nucleotide (DNA) and protein (amino acid) sequences.

```text
>seq1
MTHHPISDHEATLRCWALGFYPVEITLTQ
>seq2
MAIRGPAALSAALYLH
```
 
If users want to perform model training, a separate CSV file containing labels is required. The file should follow the format below:

```csv
id,label
seq1,positive
seq2,negative
```

The CSV file must include two columns **id** and **label**. The **id** must correspond to the headers in the FASTA file, while the **label** is used to distinguish positive and negative samples.

Users may provide separate FASTA and CSV files for training and testing; alternatively, they can provide a single set of files, which SPROTify will randomly split into training and test sets.

## SPROTify's commands

### Prediction
This module is for the prediction based on constructed models. Execution requires only a single step.

```bash
python3 scripts/model_predict.py --input INPUT_FASTA_PATH --output OUTPUT_PATH
```

**Required arguments**

- `--input`: Path to the input FASTA file.
- `--output`: Path to the output file. If the output folder does not exist, it will be created automatically.

**Additional arguments**

- `--model-type`: Model type. (options: `xgb`, `lgbm`(default) or `ada`)
- `--model-path`: Path to trained model file.
- `--s4pred-path`: the path of [s4pred](https://github.com/psipred/s4pred) folder. (default: `tools/s4pred`)
- `--params-file`: Path to the .pkl file storing the feature min-max values calculated from our training data. (default: `overall_params.pkl`) 

If `--model-type` and `--model-path` both are assigned, only `--model-path` will take effect.

**Example**

Perform prediction using default lightgbm model. The example files used below are stored in the **dataset** directory.

```bash
python3 scripts/model_predict.py --input dataset/test_set.fasta --output model_result/lgbm_testing.csv
```

If users want to use a pre-installed [s4pred](https://github.com/psipred/s4pred), 
please check the above `2. Install s4pred` section, and specify the [s4pred](https://github.com/psipred/s4pred) directory path with `--s4pred-path` parameter.

```bash
S4PRED_PATH="/path/of/your/s4pred"
# replace the "/path/of/your/s4pred" to the path of your s4pred folder
python3 scripts/model_predict.py --input dataset/test_set.fasta --output model_result/lgbm_testing.csv --s4pred-path $S4PRED_PATH
```

Users can select an alternative model by using the `--model-type` flag.

```bash
# Using xgboost model
python3 scripts/model_predict.py --model-type xgb --input dataset/test_set.fasta --output model_result/xgb_testing.csv
# Using adaboost model  
python3 scripts/model_predict.py --model-type ada --input dataset/test_set.fasta --output model_result/ada_testing.csv
```

Default execution uses `overall_params.pkl`. For custom models, provide the files generated during training (the model saved via --save-model and its matching auto-generated .pkl file):

```bash
MODEL_PATH="./save_models/your_model.joblib"
PARAMS_PATH="./your_params.pkl"

python3 scripts/model_predict.py --model-path $MODEL_PATH --input INPUT_FASTA_PATH --output OUTPUT_PATH --params-file $PARAMS_PATH
```

### Training and evaluation (train a model with user data)

This module enables users to train custom prediction models using their own datasets.

Depending on the machine learning algorithm selected, SPROTify provides different scripts for users. The paths to each program are listed below:

- `scripts/train_lightgbm.py` (Recommended, fastest and high-precision)
- `scripts/train_xgboost.py`
- `scripts/train_adaboost.py`

Additionally, depending on whether the training and test sets are provided separately or a single dataset is supplied for SPROTify to randomly split, two modes (**auto mode** and **manual mode**) are available.

#### 1. Auto mode (default)
Automatically splits your data into training and test sets. Training and model building can be executed in a single, streamlined step.

```bash
python3 scripts/train_lightgbm.py --fasta INPUT_FASTA_PATH --label_csv INPUT_LABEL_PATH 
```

**Required arguments**

- `--fasta`: Path to the input FASTA file.
- `--label-csv`: Path to the label CSV file with columns `id` and `label`.

**Additional arguments**

- `--test-ratio`: Proportion of data for testing. (default: 0.2)
- `--tune`: Enable hyperparameter optimization using [Optuna](https://github.com/optuna/optuna).
- `--n-trials`: Number of [Optuna](https://github.com/optuna/optuna) optimization trials. (default: 300; This option is available only when `--tune` flag is enabled.)
- `--run-baseline`: Perform model comparison across all methods provided by [LazyPredict](https://github.com/shankarpandala/lazypredict) without hyperparameter tuning.
- `--save-model`: Save the trained model for later prediction. Files will be automatically stored in the `save_models/` directory.
- `--mode`: Dataset assignment methods. (options: `auto`(default), `manual`)
- `--s4pred-path`: the path of [s4pred](https://github.com/psipred/s4pred) folder. (default: `tools/s4pred`)

**Example**

Using default lightgbm model to train the data. 
The following example demonstrates how to build the model with SPROTify randomly splitting the input dataset, 
corresponding to the `--mode auto` setting. 
Since `--mode` defaults to `auto`, the `--mode auto` is omitted in the commands below.
The example files used below are stored in the **dataset** directory.

```bash
python3 scripts/train_lightgbm.py --fasta dataset/full_dataset.fasta --label-csv dataset/full_true_labels.csv --save-model
```

If users want to use a pre-installed [s4pred](https://github.com/psipred/s4pred), 
please check the above `2. Install s4pred` section, and specify the [s4pred](https://github.com/psipred/s4pred) directory path with `--s4pred-path` parameter.

```bash
S4PRED_PATH="/path/of/your/s4pred"
# replace the "/path/of/your/s4pred" to the path of your s4pred folder
python3 scripts/train_lightgbm.py --fasta dataset/full_dataset.fasta --label-csv dataset/full_true_labels.csv --save-model --s4pred-path $S4PRED_PATH
```

Users can also select other models to train the data.

```bash
# Using xgboost model
python3 scripts/train_xgboost.py --fasta dataset/full_dataset.fasta --label-csv dataset/full_true_labels.csv --save_model

# Using adaboost model
python3 scripts/train_adaboost.py --fasta dataset/full_dataset.fasta --label-csv dataset/full_true_labels.csv --save-model
```
By default, 80% of the dataset is used for training and 20% for testing. Users can modify this ratio using the `--test-ratio`.

```bash
# Custom training/test set split (90/10)
python3 scripts/train_lightgbm.py --fasta dataset/full_dataset.fasta --label-csv dataset/full_true_labels.csv --test-ratio 0.1 --save-model
```

If users want to perform hyperparameter tuning during model training, they can enable `--tune` and set `--n-trials`. However, please note that this will significantly increase the runtime.
When `--tune` is used, the script will output txt files containing the evaluation metrics of the tuned model and the best hyperparameters found.

```bash
python3 scripts/train_xgboost.py --fasta dataset/full_dataset.fasta --label-csv dataset/full_true_labels.csv --tune --n-trials 100 --save-model
```

For the above command, the following files will be generated.
- **Evaluation metrics**: `results_tuned/XGBClassifier_evaluation.txt`
- **Best hyperparameters**: `best_score/XGBClassifier_optuna.txt`

SPROTify also provides an fucntion (`--run-baseline`) that allows users to train and test models using all algorithms available in [LazyPredict](https://github.com/shankarpandala/lazypredict).
This command will generate txt files containing the accuracy of each machine learning method.
The output will be saved to `results_baseline/all_model_evaluation.txt`.

Please note that these results are intended for a preliminary comparison of algorithms and reflect performance without hyperparameter optimization, thus the `--save-model` function will not be executed.

```bash
# Preliminary evaluation only, no model will be built
python3 scripts/train_lightgbm.py --fasta dataset/full_dataset.fasta --label-csv dataset/full_true_labels.csv --run-baseline
```
Additionally, a matching `<name>_params.pkl` file is automatically generated in the root directory upon the first run of a new dataset. This file is consistently reused for all future tasks(including tuning) to ensure that both training and prediction data are processed using the same criteria.

#### 2. Manual mode

Train and build models based on user-defined training and test sets. 
Training and model building can be executed in a single, streamlined step.
All the functionalities available in **auto mode** can also be executed in **manual mode**. 
Users simply need to provide separate files for the training and test datasets, and set the `--mode` to `manual`.

```bash
python3 scripts/train_lightgbm.py --mode manual \
  --train-fasta TRAIN_FASTA_PATH --train-label-csv TRAIN_LABEL_PATH \
  --test-fasta TEST_FASTA_PATH --test-label-csv TEST_LABEL_PATH
```

**Required argument**
- `--train-fasta`: Path to the training FASTA file.
- `--train-label-csv`: Path to the training label CSV file.
- `--test-fasta`: Path to the testing FASTA file.
- `--test-label-csv`: Path to the testing label CSV.
- `--mode`: Dataset assignment methods. (options: `auto`(default), `manual`).

**Additional arguments**
- `--tune`: Enable hyperparameter optimization using [Optuna](https://github.com/optuna/optuna).
- `--n-trials`: Number of [Optuna](https://github.com/optuna/optuna) optimization trials. (default: 300; This option is available only when `--tune` flag is enabled.)
- `--run-baseline`: Run baseline model comparison using [LazyPredict](https://github.com/shankarpandala/lazypredict).
- `--save-model`: Save the trained model for later prediction. Files will be automatically stored in the `save_models/` directory.
- `--s4pred-path`: the path of [s4pred](https://github.com/psipred/s4pred) folder. (default: `tools/s4pred`)

**Example**

Using default lightgbm model to train the data. The example files used below are stored in the **dataset** directory.

```bash
# Train and save lightgbm model
python3 scripts/train_lightgbm.py --mode manual \
  --train-fasta dataset/train_set.fasta --train-label-csv dataset/train_true_labels.csv \
  --test-fasta dataset/test_set.fasta --test-label-csv dataset/test_true_labels.csv \
  --save-model
```

If users want to use a pre-installed [s4pred](https://github.com/psipred/s4pred), 
please check the above `2. Install s4pred` section, and specify the [s4pred](https://github.com/psipred/s4pred) directory path with `--s4pred-path` parameter.

```bash
S4PRED_PATH="/path/of/your/s4pred"
# replace the "/path/of/your/s4pred" to the path of your s4pred folder
python3 scripts/train_lightgbm.py --mode manual \
  --train-fasta dataset/train_set.fasta --train-label-csv dataset/train_true_labels.csv \
  --test-fasta dataset/test_set.fasta --test-label-csv dataset/test_true_labels.csv \
  --save-model --s4pred-path $S4PRED_PATH
```

Users can also select other models to train the data.

```bash
# Using xgboost model
python3 scripts/train_xgboost.py --mode manual \
  --train-fasta dataset/train_set.fasta --train-label-csv dataset/train_true_labels.csv \
  --test-fasta dataset/test_set.fasta --test-label-csv dataset/test_true_labels.csv \
  --save-model

# Using adaboost model
python3 scripts/train_adaboost.py --mode manual \
  --train-fasta dataset/train_set.fasta --train-label-csv dataset/train_true_labels.csv \
  --test-fasta dataset/test_set.fasta --test-label-csv dataset/test_true_labels.csv \
  --save-model
```

If users want to perform hyperparameter tuning during model training, they can enable `--tune` and set `--n-trials`. 
However, please note that this will significantly increase the runtime.
When `--tune` is used in Manual mode, the output files and folder structure are the same as in **auto mode**.

```bash
python3 scripts/train_xgboost.py --mode manual \
  --train-fasta dataset/train_set.fasta --train-label-csv dataset/train_true_labels.csv \
  --test-fasta dataset/test_set.fasta --test-label-csv dataset/test_true_labels.csv \
  --tune --n-trials 100 --save-model
```

Same as **auto mode**, users can also perform preliminary comparison of different algorithms provided by 
[LazyPredict](https://github.com/shankarpandala/lazypredict). The output files and directory structure are the same as in auto mode.

Please note that these results are intended for a preliminary comparison of algorithms and reflect performance without hyperparameter optimization, thus the `--save-model` function will not be executed.
The output files and folder structure are the same as in **auto mode**.

```bash
# Preliminary evaluation only, no model will be built
python3 scripts/train_lightgbm.py --mode manual \
  --train-fasta dataset/train_set.fasta --train-label-csv dataset/train_true_labels.csv \
  --test-fasta dataset/test_set.fasta --test-label-csv dataset/test_true_labels.csv \
  --run-baseline
```


