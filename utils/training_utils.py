import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_predict
from lazypredict.Supervised import LazyClassifier
from metrics import compute_metrics, get_scores


def train_and_evaluate_models(train_set, train_labels, test_set, test_labels, tuned_models, run_baseline=True):
    """
    Train baseline models (LazyPredict) and tuned models, then evaluate them using
    cross-validation and an independent test set.

    Args:
        train_set (np.ndarray or pd.DataFrame): 
            Training feature matrix with shape (n_train, n_features).
        train_labels (np.ndarray or list[int]): 
            Training labels.
        test_set (np.ndarray or pd.DataFrame): 
            Testing feature matrix with shape (n_test, n_features).
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
        model.fit(X_train, y_train)
        model_dict[name] = model

    # Evaluation 
    evaluation_results = []
    for model_class, model_wrapper in model_dict.items():
        # Extract actual model if wrapped
        model = getattr(model_wrapper, 'model', model_wrapper)

        # Cross-validation on training set
        y_pred_train = cross_val_predict(model, X_train, y_train, cv=5)
        y_prob_train = get_scores(model, X_train, y_train, cv=5)
        f1_train, acc_train, auc_train, sens_train, spec_train = compute_metrics(
            y_train, y_pred_train, y_prob_train
        )

        # Evaluation on independent test set
        y_pred_test = model_wrapper.predict(test_set)
        y_prob_test = get_scores(model_wrapper, test_set)
        f1_test, acc_test, auc_test, sens_test, spec_test = compute_metrics(
            test_labels, y_pred_test, y_prob_test
        )

        evaluation_results.append({
            'model': model_class,
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



def save_results_report(results_df, sort_key='test_auc', save_path=None, model_name=None):
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
            Column name used to sort the models. Default is 'test_auc'.
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

    if sort_key not in results_df.columns:
        print(f"sort_key '{sort_key}' not found. Available columns: {results_df.columns.tolist()}")
        sort_key = results_df.columns[0]

    # Sort results by the specified key in descending order
    results_sorted = results_df.sort_values(by=sort_key, ascending=False).to_dict('records')


    output_lines = []
    for i, res in enumerate(results_sorted, 1):
        model_class = res['model']

        # Updated model status logic
        if 'optuna' in model_class.lower():
            status = 'Optuna tuned'
        elif '' in model_class.lower():
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

    print(f"Saved evaluation results to: {text_file}")