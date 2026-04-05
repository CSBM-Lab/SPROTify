import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from lazypredict.Supervised import LazyClassifier
from metrics import compute_metrics, get_scores
from sklearn.metrics import recall_score, make_scorer


def train_and_evaluate_models(train_set, train_labels, test_set, test_labels, tuned_models, run_baseline=True):
    """
    Train baseline models (LazyPredict) and tuned models, then evaluate them using
    cross-validation and an independent test set.

    Args:
        train_set (np.ndarray or pd.DataFrame): 
            Training feature matrix with shape (n_samples, n_features).
        train_labels (np.ndarray or list[int]): 
            Training labels.
        test_set (np.ndarray or pd.DataFrame): 
            Testing feature matrix with shape (n_samples, n_features).
        test_labels (np.ndarray or list[int]): 
            Testing labels.
        tuned_models (dict[str, object]): 
            A dictionary mapping model names to model instances.
        run_baseline (bool): 
            If True, runs LazyPredict to generate baseline model results.

    Returns:
        pandas.DataFrame: Metrics from training CV and independent testing
        (F1, accuracy, AUC, sensitivity, specificity).

    """

    # Split training data into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_set, train_labels, test_size=0.2, random_state=1
    )

    model_dict = {}

    # LazyPredict baseline
    if run_baseline:
        
        try:
            clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
            models, predictions = clf.fit(X_train, X_val, y_train, y_val)
            print(f"LazyPredict finished: {len(models)} models trained.")
        except Exception as e:
            print(f"LazyPredict failed: {e}")
            models = pd.DataFrame()

        if models.empty:
            print("No valid baseline results generated.")
        else:
            model_dict.update(clf.models)

    # Tuned models
    for name, model in tuned_models.items():
        model_dict[name] = model

    # Evaluation 
    evaluation_results = []
    final_trained_models = {} 

    for model_name, model_wrapper in model_dict.items():
        model = getattr(model_wrapper, 'model', model_wrapper)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        specificity_scorer = make_scorer(recall_score, pos_label=0)

        scoring_metrics = {
            'auc': 'roc_auc',
            'sensitivity': 'recall',      
            'specificity': specificity_scorer,
            'f1': 'f1',
            'acc': 'accuracy'
        }

        cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring_metrics)


        f1_mean   = cv_results['test_f1'].mean()
        acc_mean  = cv_results['test_acc'].mean()
        auc_mean  = cv_results['test_auc'].mean()
        sens_mean = cv_results['test_sensitivity'].mean()
        spec_mean = cv_results['test_specificity'].mean()

        
        final_model = clone(model) 
        final_model.fit(train_set, train_labels)

        # Evaluation on independent test set
        y_prob_test = get_scores(final_model, test_set, model_name=model_name)

        if y_prob_test is not None:
            y_pred_test = (y_prob_test >= 0.5).astype(int)

            f1_test, acc_test, auc_test, sens_test, spec_test = compute_metrics(
                test_labels, y_pred_test, y_prob_test
            )
        else:
            f1_test = acc_test = auc_test = sens_test = spec_test = np.nan

        final_trained_models[model_name] = final_model

        evaluation_results.append({
            'model': model_name,
            'train_f1': f1_mean,
            'train_acc': acc_mean,
            'train_auc': auc_mean,
            'train_sens': sens_mean,
            'train_spec': spec_mean,
            'test_f1': f1_test,
            'test_acc': acc_test,
            'test_auc': auc_test,
            'test_sens': sens_test,
            'test_spec': spec_test,
        })

    if len(evaluation_results) == 0:
        print("No valid model results generated.")
        return pd.DataFrame()

    return pd.DataFrame(evaluation_results), final_trained_models



def save_results_report(results_df, sort_key=None, save_path=None, model_name=None):
    """
    Save a formatted evaluation report of model performance to a text file.

    The report includes metrics for each model, both cross-validated training
    metrics and independent test metrics.

    Args:
        results_df (pd.DataFrame): 
            DataFrame containing model evaluation results.
            Must include the metrics for training and test sets 
            (F1, Accuracy, AUC, Sensitivity, Specificity) and a 'model' column.
        sort_key (str, optional): 
            Column name used to sort the models.
        save_path (str or None, optional): 
            Directory to save the report file.
            If None, defaults to `project_root/results_baseline`.
        model_name (str or None, optional): 
            Model name to include in the filename.
            If None, uses 'all_model_evaluation.txt'.

    Returns:
        None
    """

    if results_df.empty:
        print('No results to save — DataFrame is empty.')
        return

    if save_path is None:
        save_path = os.path.join(project_root, 'results_baseline')
    os.makedirs(save_path, exist_ok=True)


    if sort_key and len(results_df) > 1:
        results_df = results_df.sort_values(by=sort_key, ascending=False)
    

    results_sorted = results_df.to_dict('records')
    
    output_lines = []
    for i, res in enumerate(results_sorted, 1):
        model_class = res['model']

        # Updated model status logic
        if 'optuna' in model_class.lower():
            status = 'Optuna tuned'
        elif len(results_sorted) == 1:
            status = ''
        else:
            status = 'Baseline'

        # Build model display name
        if status:
            model_label = f"{model_class}  ({status})"
        else:
            model_label = f"{model_class}"


        # Determine if model is tuned/fixed or baseline for display
        rank_label = f'#{i}  ' if len(results_sorted) > 1 else ''

        block = [
            f"\n{rank_label}{model_label}",
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


    if model_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f'{model_name}_evaluation_{timestamp}.txt'
    else:
        filename = 'all_model_evaluation.txt'

    text_file = os.path.join(save_path, filename)

    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(output_text)

    print(f"\nSaved evaluation results to: {text_file}")