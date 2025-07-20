"""
Visualization utilities for fraud detection.

This module provides functions to create visualizations for EDA, model evaluation,
and interpretation of fraud detection models.
"""

from .plot_utils import (
    plot_class_distribution,
    plot_numerical_distribution,
    plot_categorical_distribution,
    plot_correlation_matrix,
    plot_tsne,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_shap_summary,
    plot_shap_dependence
)

__all__ = [
    'plot_class_distribution',
    'plot_numerical_distribution',
    'plot_categorical_distribution',
    'plot_correlation_matrix',
    'plot_tsne',
    'plot_feature_importance',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_shap_summary',
    'plot_shap_dependence'
]
