"""
Visualization utilities for survival analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Dict

from .config import TARGET_TIME, TARGET_EVENT


def plot_survival_distribution(
    df: pd.DataFrame,
    figsize: tuple = (12, 5),
    bins: int = 40
) -> plt.Figure:
    """
    Plot distribution of survival times, split by event status.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with OS_YEARS and OS_STATUS
    figsize : tuple
        Figure size
    bins : int
        Number of histogram bins
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Overall distribution
    ax = axes[0]
    df[TARGET_TIME].hist(bins=bins, ax=ax, edgecolor='white', alpha=0.7)
    ax.set_title("Distribution of OS_YEARS (all patients)")
    ax.set_xlabel("OS_YEARS")
    ax.set_ylabel("Count")
    
    # By event status
    ax = axes[1]
    df.loc[df[TARGET_EVENT] == 1, TARGET_TIME].hist(
        bins=bins, alpha=0.6, label="Death (event=1)", ax=ax, color='red'
    )
    df.loc[df[TARGET_EVENT] == 0, TARGET_TIME].hist(
        bins=bins, alpha=0.6, label="Censored (event=0)", ax=ax, color='blue'
    )
    ax.set_title("OS_YEARS: Death vs Censored")
    ax.set_xlabel("OS_YEARS")
    ax.set_ylabel("Count")
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_feature_importance(
    importances: pd.Series,
    top_n: int = 15,
    figsize: tuple = (10, 6),
    title: str = "Permutation Importance (IPCW C-index drop)"
) -> plt.Figure:
    """
    Plot feature importance as horizontal bar chart.
    
    Parameters
    ----------
    importances : pd.Series
        Feature importances (index=feature name, value=importance)
    top_n : int
        Number of top features to show
    figsize : tuple
        Figure size
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    top_imp = importances.head(top_n).iloc[::-1]  # Reverse for horizontal bars
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_imp)))
    
    ax.barh(top_imp.index, top_imp.values, color=colors)
    ax.set_xlabel("Score Drop (higher = more important)")
    ax.set_title(title)
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    return fig


def plot_cluster_survival(
    cluster_summary: pd.DataFrame,
    figsize: tuple = (10, 5)
) -> plt.Figure:
    """
    Plot cluster survival characteristics.
    
    Parameters
    ----------
    cluster_summary : pd.DataFrame
        Summary with columns: n, death_rate, os_median, os_mean
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Median survival by cluster
    ax = axes[0]
    clusters = cluster_summary.index.astype(str)
    ax.bar(clusters, cluster_summary["os_median"], color='steelblue', alpha=0.7)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Median OS (years)")
    ax.set_title("Median Survival by Cluster")
    
    # Death rate by cluster
    ax = axes[1]
    colors = plt.cm.Reds(cluster_summary["death_rate"])
    ax.bar(clusters, cluster_summary["death_rate"], color=colors)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Death Rate")
    ax.set_title("Death Rate by Cluster")
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig


def plot_silhouette_scores(
    scores: Dict[int, float],
    best_k: int,
    figsize: tuple = (8, 5)
) -> plt.Figure:
    """
    Plot silhouette scores for different k values.
    
    Parameters
    ----------
    scores : Dict[int, float]
        Silhouette scores by k
    best_k : int
        Best k value
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    k_values = sorted(scores.keys())
    sil_values = [scores[k] for k in k_values]
    
    ax.plot(k_values, sil_values, 'bo-', markersize=8)
    ax.axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("KMeans Clustering: Silhouette Score vs k")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_model_comparison(
    results: Dict[str, float],
    figsize: tuple = (10, 6),
    title: str = "Model Comparison (IPCW C-index)"
) -> plt.Figure:
    """
    Plot comparison of model performances.
    
    Parameters
    ----------
    results : Dict[str, float]
        Model name -> score
    figsize : tuple
        Figure size
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = list(results.keys())
    scores = list(results.values())
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    
    bars = ax.bar(models, scores, color=colors)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{score:.4f}',
            ha='center', va='bottom', fontsize=10
        )
    
    ax.set_ylabel("IPCW C-index")
    ax.set_title(title)
    ax.set_ylim(0, max(scores) * 1.15)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.legend()
    
    # Rotate x labels if many models
    if len(models) > 4:
        plt.xticks(rotation=30, ha='right')
    
    plt.tight_layout()
    return fig


def plot_cv_scores(
    cv_scores: List[float],
    model_name: str = "Model",
    figsize: tuple = (8, 5)
) -> plt.Figure:
    """
    Plot cross-validation score distribution.
    
    Parameters
    ----------
    cv_scores : List[float]
        CV fold scores
    model_name : str
        Model name for title
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.boxplot(cv_scores, vert=True)
    ax.scatter([1] * len(cv_scores), cv_scores, alpha=0.6, s=100)
    
    mean_score = np.mean(cv_scores)
    ax.axhline(y=mean_score, color='red', linestyle='--', 
               label=f'Mean: {mean_score:.4f}')
    
    ax.set_ylabel("IPCW C-index")
    ax.set_title(f"{model_name}: Cross-Validation Scores")
    ax.set_xticklabels([model_name])
    ax.legend()
    
    plt.tight_layout()
    return fig
