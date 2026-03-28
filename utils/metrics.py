import os
import numpy as np
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score, recall_score, confusion_matrix)
from sklearn.model_selection import cross_val_predict


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute classification evaluation metrics：
    F1、Accuracy、AUC、Sensitivity、Specificity

    Args:
        y_true (array-like): 
            Ground truth binary labels (0 or 1).
        y_pred (array-like):
            Predicted binary class labels (0 or 1).
        y_prob (array-like, optional): 
            Predicted probabilities for the positive class.
            If None or all values are NaN, AUC will be set to NaN.
            Default is None.

    Returns:
        A tuple containing (f1, accuracy, auc, sensitivity, specificity).
        All values are floats.
    """

    f1 = f1_score(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)

    if y_prob is not None:

        y_prob = np.asarray(y_prob)

        if np.any(np.isnan(y_prob)) or np.any(np.isinf(y_prob)):
            auc = np.nan
        else:
            try:
                auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                auc = np.nan
    else:
        auc = np.nan

    sensitivity = recall_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = np.nan  # Avoid division by zero
        
    return f1, acc, auc, sensitivity, specificity

def get_scores(model, X, y=None, cv=None):
    """
    Get prediction scores from a model.

    Args:
        model: 
            A fitted model or an unfitted estimator that implements
            'predict_proba' or 'decision_function'.
        X: 
            Training feature matrix with shape (n_train, n_features).
        y: 
            Training labels. Required only when using cross-validation.
        cv:
            Cross-validation number of folds.

    Returns:
        array or None: 
            Prediction scores (higher = more likely positive class),
            or None if the model does not support scoring methods.
    """

    # When using cross-validation
    if cv is not None and y is not None:
        if hasattr(model, 'predict_proba'):
            return cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
        elif hasattr(model, 'decision_function'):
            return cross_val_predict(model, X, y, cv=cv, method='decision_function')
        else:
            return None
    # Without cross-validation: directly predict on the given dataset (e.g., test set)
    else:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            return model.decision_function(X)
        else:
            return None