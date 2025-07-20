"""
Plotting utilities for fraud detection.

This module provides functions to create various visualizations for EDA,
model evaluation, and interpretation of fraud detection models.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Union
import logging
from matplotlib.colors import LinearSegmentedColormap

# Configure logging
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Custom color palette
FRAUD_PALETTE = ['#3498db', '#e74c3c']  # Blue for non-fraud, Red for fraud

def plot_class_distribution(
    y: Union[pd.Series, np.ndarray],
    title: str = 'Class Distribution',
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot the distribution of classes (fraud vs non-fraud).
    
    Args:
        y: Target variable (0 for non-fraud, 1 for fraud)
        title: Plot title
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Count plot
    sns.countplot(x=y, ax=ax1, palette=FRAUD_PALETTE)
    ax1.set_title('Count of Transactions')
    ax1.set_xticklabels(['Non-Fraud', 'Fraud'])
    ax1.set_ylabel('Count')
    
    # Pie chart
    class_counts = pd.Series(y).value_counts()
    ax2.pie(
        class_counts,
        labels=['Non-Fraud', 'Fraud'],
        autopct='%1.1f%%',
        colors=FRAUD_PALETTE,
        startangle=90,
        explode=(0.1, 0)
    )
    ax2.set_title('Percentage of Transactions')
    
    plt.suptitle(title, y=1.05, fontsize=14)
    plt.tight_layout()
    
    return fig

def plot_numerical_distribution(
    df: pd.DataFrame,
    column: str,
    target: str = 'class',
    bins: int = 50,
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """
    Plot distribution of a numerical feature with respect to the target.
    
    Args:
        df: DataFrame containing the data
        column: Name of the numerical column to plot
        target: Name of the target column
        bins: Number of bins for histograms
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Distribution plot
    sns.histplot(
        data=df,
        x=column,
        hue=target,
        bins=bins,
        kde=True,
        element='step',
        common_norm=False,
        palette=FRAUD_PALETTE,
        ax=ax1
    )
    ax1.set_title(f'Distribution of {column}')
    ax1.legend(title='Class', labels=['Non-Fraud', 'Fraud'])
    
    # Box plot
    sns.boxplot(
        data=df,
        x=target,
        y=column,
        palette=FRAUD_PALETTE,
        showfliers=False,
        ax=ax2
    )
    ax2.set_title(f'Boxplot of {column} by Class')
    ax2.set_xticklabels(['Non-Fraud', 'Fraud'])
    
    plt.suptitle(f'Analysis of {column} by Fraud Status', y=1.05, fontsize=14)
    plt.tight_layout()
    
    return fig

def plot_categorical_distribution(
    df: pd.DataFrame,
    column: str,
    target: str = 'class',
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """
    Plot distribution of a categorical feature with respect to the target.
    
    Args:
        df: DataFrame containing the data
        column: Name of the categorical column to plot
        target: Name of the target column
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    # Calculate value counts and percentages
    cross_tab = pd.crosstab(df[column], df[target], normalize='index')
    cross_tab = cross_tab.sort_values(1, ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Count plot
    sns.countplot(
        data=df,
        x=column,
        hue=target,
        order=cross_tab.index,
        palette=FRAUD_PALETTE,
        ax=ax1
    )
    ax1.set_title(f'Count of {column} by Class')
    ax1.legend(title='Class', labels=['Non-Fraud', 'Fraud'])
    ax1.tick_params(axis='x', rotation=45)
    
    # Percentage stacked bar plot
    cross_tab.plot(kind='bar', stacked=True, color=FRAUD_PALETTE, ax=ax2)
    ax2.set_title(f'Percentage of Fraud by {column}')
    ax2.set_ylabel('Percentage')
    ax2.legend(title='Class', labels=['Non-Fraud', 'Fraud'])
    ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'Analysis of {column} by Fraud Status', y=1.05, fontsize=14)
    plt.tight_layout()
    
    return fig

def plot_correlation_matrix(
    df: pd.DataFrame,
    method: str = 'pearson',
    figsize: tuple = (14, 12),
    annot: bool = True,
    cmap: str = 'coolwarm',
    vmin: float = -1,
    vmax: float = 1
) -> plt.Figure:
    """
    Plot correlation matrix of numerical features.
    
    Args:
        df: DataFrame containing numerical features
        method: Correlation method ('pearson', 'spearman', 'kendall')
        figsize: Figure size (width, height)
        annot: Whether to annotate the heatmap
        cmap: Color map for the heatmap
        vmin: Minimum value for color scaling
        vmax: Maximum value for color scaling
        
    Returns:
        Matplotlib Figure object
    """
    # Calculate correlation matrix
    corr = df.select_dtypes(include=[np.number]).corr(method=method)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        fmt='.2f',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title(f'Correlation Matrix ({method.capitalize()})', fontsize=14)
    plt.tight_layout()
    
    return plt.gcf()

def plot_tsne(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    perplexity: int = 30,
    n_components: int = 2,
    random_state: int = 42,
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """
    Plot t-SNE visualization of the data.
    
    Args:
        X: Feature matrix
        y: Target labels
        perplexity: t-SNE perplexity parameter
        n_components: Number of dimensions (2 or 3)
        random_state: Random seed
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    from sklearn.manifold import TSNE
    
    # Convert to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    # Apply t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_jobs=-1
    )
    
    logger.info("Fitting t-SNE...")
    X_tsne = tsne.fit_transform(X)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    if n_components == 2:
        scatter = plt.scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            c=y,
            cmap=plt.cm.get_cmap('viridis', 2),
            alpha=0.7,
            edgecolors='w',
            linewidth=0.5
        )
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
    else:  # 3D
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.axes(projection='3d')
        scatter = ax.scatter3D(
            X_tsne[:, 0],
            X_tsne[:, 1],
            X_tsne[:, 2],
            c=y,
            cmap=plt.cm.get_cmap('viridis', 2),
            alpha=0.7,
            edgecolors='w',
            linewidth=0.5
        )
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_zlabel('t-SNE Component 3')
    
    # Add legend
    legend = plt.legend(*scatter.legend_elements(), title='Classes')
    plt.gca().add_artist(legend)
    
    plt.title('t-SNE Visualization of Transaction Data', fontsize=14)
    plt.tight_layout()
    
    return plt.gcf()

def plot_feature_importance(
    importance: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    figsize: tuple = (12, 8)
) -> plt.Figure:
    """
    Plot feature importance scores.
    
    Args:
        importance: Array of feature importance scores
        feature_names: List of feature names
        top_n: Number of top features to display
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    # Create a DataFrame for plotting
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Create plot
    plt.figure(figsize=figsize)
    sns.barplot(
        x='importance',
        y='feature',
        data=feature_importance,
        palette='viridis'
    )
    
    plt.title(f'Top {top_n} Most Important Features', fontsize=14)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    return plt.gcf()

def plot_confusion_matrix(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    classes: List[str] = None,
    normalize: bool = False,
    cmap: Any = plt.cm.Blues,
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """
    Plot a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        normalize: Whether to normalize the confusion matrix
        cmap: Color map for the heatmap
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    from sklearn.metrics import confusion_matrix
    import itertools
    
    if classes is None:
        classes = ['Non-Fraud', 'Fraud']
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
    else:
        title = 'Confusion Matrix'
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Create heatmap
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    return plt.gcf()

def plot_roc_curve(
    y_true: Union[np.ndarray, pd.Series],
    y_score: np.ndarray,
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True binary labels
        y_score: Target scores (probability estimates of the positive class)
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot ROC curve
    plt.plot(
        fpr, tpr, color='darkorange',
        lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'
    )
    
    # Plot random guessing line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True)
    
    return plt.gcf()

def plot_precision_recall_curve(
    y_true: Union[np.ndarray, pd.Series],
    y_score: np.ndarray,
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True binary labels
        y_score: Target scores (probability estimates of the positive class)
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # Compute precision-recall curve and average precision
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot precision-recall curve
    plt.step(
        recall, precision, color='b', alpha=0.2,
        where='post', label=f'AP = {avg_precision:.2f}'
    )
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    
    # Set plot properties
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True)
    
    return plt.gcf()

def plot_shap_summary(
    shap_values: np.ndarray,
    features: Union[np.ndarray, pd.DataFrame],
    feature_names: List[str] = None,
    max_display: int = 20,
    plot_type: str = 'dot',
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """
    Plot SHAP summary plot.
    
    Args:
        shap_values: SHAP values matrix (samples x features)
        features: Feature matrix used to calculate SHAP values
        feature_names: List of feature names
        max_display: Maximum number of features to display
        plot_type: Type of plot ('dot', 'bar', 'violin')
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    import shap
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Generate SHAP summary plot
    shap.summary_plot(
        shap_values,
        features,
        feature_names=feature_names,
        max_display=max_display,
        plot_type=plot_type,
        show=False
    )
    
    plt.title('SHAP Feature Importance', fontsize=14)
    plt.tight_layout()
    
    return plt.gcf()

def plot_shap_dependence(
    shap_values: np.ndarray,
    features: Union[np.ndarray, pd.DataFrame],
    feature_names: List[str],
    target_feature: str,
    interaction_index: str = None,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot SHAP dependence plot for a single feature.
    
    Args:
        shap_values: SHAP values matrix (samples x features)
        features: Feature matrix used to calculate SHAP values
        feature_names: List of feature names
        target_feature: Name of the feature to plot dependence for
        interaction_index: Feature to use for coloring the points
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    import shap
    
    # Get index of target feature
    if isinstance(feature_names, list):
        feature_idx = feature_names.index(target_feature)
    else:
        feature_idx = target_feature
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Generate SHAP dependence plot
    shap.dependence_plot(
        feature_idx,
        shap_values,
        features,
        feature_names=feature_names,
        interaction_index=interaction_index,
        show=False
    )
    
    plt.title(f'SHAP Dependence Plot for {target_feature}', fontsize=14)
    plt.tight_layout()
    
    return plt.gcf()
