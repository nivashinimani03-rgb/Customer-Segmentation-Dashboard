"""
Customer Segmentation Project
Segments customers based on behavior (spending score) and demographics
(age, income) using K-Means clustering.

Usage:
    python customer_segmentation.py mall_customers.csv

Requires: pandas, numpy, scikit-learn, matplotlib
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'AnnualIncome_k', 'SpendingScore'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")
    return df


def find_optimal_k(X_scaled, k_range=range(2, 10)):
    """Elbow method + silhouette score to pick the number of clusters."""
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))
    best_k = list(k_range)[int(np.argmax(sil_scores))]
    return best_k, inertias, sil_scores, list(k_range)


def label_segment(income: float, score: float) -> str:
    """Turn cluster centroids into human-readable segment names."""
    if income >= 75 and score >= 60:
        return 'Premium Spenders'
    if income >= 75 and score < 40:
        return 'High-Income Savers'
    if income < 45 and score >= 60:
        return 'Budget Impulsive Buyers'
    if income < 45 and score < 40:
        return 'Low Engagement'
    return 'Balanced / Average'


def run_segmentation(df: pd.DataFrame, n_clusters: int = 5):
    features = df[['AnnualIncome_k', 'SpendingScore']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    best_k, inertias, sil_scores, k_range = find_optimal_k(X_scaled)
    print(f"Suggested optimal k (by silhouette score): {best_k}")

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df['Cluster'] = model.fit_predict(X_scaled)

    profile = df.groupby('Cluster')[['AnnualIncome_k', 'SpendingScore', 'Age']].mean()
    df['Segment'] = df['Cluster'].map(
        {c: label_segment(row.AnnualIncome_k, row.SpendingScore)
         for c, row in profile.iterrows()}
    )
    return df, profile, (k_range, inertias, sil_scores)


def plot_results(df: pd.DataFrame, k_range, inertias, sil_scores, out_prefix='segmentation'):
    # Elbow + silhouette
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(list(k_range), inertias, marker='o')
    axes[0].set_title('Elbow Method')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Inertia')

    axes[1].plot(list(k_range), sil_scores, marker='o', color='orange')
    axes[1].set_title('Silhouette Score')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Score')
    fig.tight_layout()
    fig.savefig(f'{out_prefix}_k_selection.png', dpi=150)

    # Segment scatter
    fig2, ax = plt.subplots(figsize=(7, 6))
    for seg in df['Segment'].unique():
        sub = df[df['Segment'] == seg]
        ax.scatter(sub['AnnualIncome_k'], sub['SpendingScore'], label=seg, alpha=0.7)
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score')
    ax.set_title('Customer Segments')
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(f'{out_prefix}_clusters.png', dpi=150)
    print(f"Saved: {out_prefix}_k_selection.png, {out_prefix}_clusters.png")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'mall_customers.csv'
    data = load_data(path)
    result_df, profile, (k_range, inertias, sil_scores) = run_segmentation(data)

    print("\nSegment profile (mean values):")
    print(profile.round(1))
    print("\nSegment sizes:")
    print(result_df['Segment'].value_counts())

    result_df.to_csv('mall_customers_clustered.csv', index=False)
    plot_results(result_df, k_range, inertias, sil_scores)
    print("\nDone. Clustered data saved to mall_customers_clustered.csv")
