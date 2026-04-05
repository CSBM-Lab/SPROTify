import os
from datetime import datetime
import pandas as pd
import lightgbm as lgbm
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import optuna
from sequence_io import project_root



optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_lightgbm(train_set, train_labels, trial):
    """
    Objective function for Optuna hyperparameter tuning of LGBMClassifier.

    Args:
        train_set (np.ndarray or pd.DataFrame):
            Training feature matrix with shape (n_samples, n_features).
        train_labels (np.ndarray or list[int]):
            Training labels.
        trial (optuna.trial.Trial): 
            Optuna trial object for suggesting hyperparameters.

    Returns:
        float: Validation accuracy of the trained LGBMClassifier.
    """
    X_train, X_valid, y_train, y_valid = train_test_split(train_set, train_labels, test_size=0.2,random_state=1,shuffle=True)
    lgbm_param={
        'objective': 'binary',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),  
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True), 
        'verbosity': -1,
        'random_state': 1
    }
    lgbm_model = lgbm.LGBMClassifier(**lgbm_param, verbose=-1)
    lgbm_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

    # Store trained parameters and model in trial attributes
    trial.set_user_attr("trained_params", lgbm_param)
    trial.set_user_attr("trained_model", lgbm_model)

    accuracy = lgbm_model.score(X_valid, y_valid)
    return accuracy

def objective_xgboost(train_set, train_labels, trial):
    """
    Objective function for Optuna hyperparameter tuning of XGBClassifier.
    """
    
    X_train, X_valid, y_train, y_valid = train_test_split(train_set, train_labels, test_size=0.2,random_state=1)
    xgb_param={
        
        'objective': 'binary:logistic',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),  
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),  
        'max_depth': trial.suggest_int('max_depth', 3, 12),     
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),         
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True), 
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20), 
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),       
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 1
    }
    xgb_model = XGBClassifier(**xgb_param)
    xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],verbose=False)

    trial.set_user_attr("trained_params", xgb_param)
    trial.set_user_attr("trained_model", xgb_model)

    accuracy = xgb_model.score(X_valid, y_valid)
    return accuracy


def objective_adaboost(train_set, train_labels, trial):
    """
    Objective function for Optuna hyperparameter tuning of AdaBoostClassifier.
    """
    
    X_train, X_valid, y_train, y_valid = train_test_split(train_set, train_labels, test_size=0.2,random_state=1)

    estimator =  DecisionTreeClassifier(
        max_depth = trial.suggest_int('max_depth', 1, 3),
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10),
        random_state=1
    )


    ada_param={  
        'estimator': estimator,
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
        'random_state': 1
    }

    ada_model = AdaBoostClassifier(**ada_param)
    ada_model.fit(X_train, y_train)

    trial.set_user_attr("trained_params", ada_param)
    trial.set_user_attr("trained_model", ada_model)

    accuracy = ada_model.score(X_valid, y_valid)
    return accuracy



def save_optuna_results(study, trial_history, filename):
    """
    Save the Optuna tuning history and the best trial information to a text file.

    Args:
        study (optuna.study.Study): 
            The Optuna Study object that stores all optimization results.
        trial_history (list[dict]): 
            List of dictionaries recording best trial number and best score at each trial.
        filename (str): 
            The output file name used to save results.

    Returns:
        None
    """
    
    base_dir = os.path.join(project_root, 'best_score')
    os.makedirs(base_dir, exist_ok=True)

    results_path = os.path.join(base_dir, filename)


    history_df = pd.DataFrame(trial_history)
    history_df.to_csv(results_path, sep=' ', index=False, header=True)
    with open(results_path, 'a') as f:
        f.write("\n=== Best Trial ===\n")
        f.write(f"Trial Number: {study.best_trial.number}\n")
        f.write(f"Best Score: {study.best_trial.value:.6f}\n")
        f.write("Best Params:\n")
        for k, v in study.best_trial.params.items():
            f.write(f"  {k}: {v}\n")

    print(f"\nSaved Optuna results to: {results_path}")




def run_optuna(model_name, train_set, train_labels, n_trials):
    """
    Run Optuna hyperparameter tuning for the given model type.

    Args:
        model_name (str):
            Model class name (from model_class.__name__),
            used for naming Optuna study, tuned_models key,
            saved model files, and result CSVs. (e.g., 'LGBMClassifier', 'XGBClassifier').
        train_set (np.ndarray or pd.DataFrame):
            Training feature matrix with shape (n_samples, n_features).
        train_labels (np.ndarray or list[int]):
            Training labels.
        n_trials (int):
            Number of Optuna trials to execute.

    Returns:
        optuna.study.Study:
            The Optuna study object containing all trials, scores,
            and the best hyperparameters.
    """

    # Map model names to corresponding objective functions
    model_objectives = {
        'LGBMClassifier': objective_lightgbm,
        'XGBClassifier': objective_xgboost,
        'AdaBoostClassifier': objective_adaboost
    }
     
    objective_func = model_objectives[model_name]

    trial_history = []
    best_trial_number = 0
    best_score_so_far = float('-inf') # Initialize best score to negative infinity

    def objective(trial):
        """
        Optuna objective function for a single trial.
        """
        nonlocal best_trial_number, best_score_so_far
        trial_score = objective_func(train_set, train_labels, trial)

        if trial_score > best_score_so_far:
            best_trial_number = trial.number
            best_score_so_far = trial_score

        trial_history.append({
            'Trial_Number': int(best_trial_number),
            'Best_Score': best_score_so_far
        })
        return trial_score

    # Create Optuna study and optimize
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=1))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{model_name}_optuna_{timestamp}.txt"

    
    save_optuna_results(study, trial_history, filename=filename)

    return study
