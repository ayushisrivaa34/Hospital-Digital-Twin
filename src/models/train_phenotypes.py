"""
train_phenotypes.py — Task C: Patient Phenotyping (Clustering)
==============================================================
Uses PyCaret clustering to identify patient phenotypes/archetypes.
Tries multiple cluster counts and selects the best by silhouette score.

Usage:
    python -m src.models.train_phenotypes
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def train_phenotype_model():
    """Run clustering to identify patient phenotypes."""
    from pycaret.clustering import setup, create_model, assign_model, save_model

    # ── Load features ──────────────────────────────────────────────────────
    features_path = os.path.join(PROJECT_ROOT, "data", "processed", "features.csv")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} records for phenotyping")

    # ── Prepare clustering features ────────────────────────────────────────
    # Drop target columns — clustering is unsupervised
    drop_cols = ["length_of_stay_days", "in_hospital_death"]
    df_patients = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── PyCaret Setup ──────────────────────────────────────────────────────
    cluster_setup = setup(
        data=df_patients,
        normalize=True,
        session_id=42,
        pca=True,
        pca_components=0.95,
        verbose=False,
        html=False
    )

    # ── Try multiple cluster counts ────────────────────────────────────────
    cluster_counts = [4, 6, 8]
    best_score = -1
    best_model = None
    best_k = None

    for k in cluster_counts:
        print(f"\n  Trying k={k}...")
        model = create_model("kmeans", num_clusters=k)

        # Get silhouette score from model
        from sklearn.metrics import silhouette_score
        results = assign_model(model)
        cluster_col = [c for c in results.columns if "Cluster" in c][0]
        score = silhouette_score(
            df_patients.select_dtypes(include=[np.number]).fillna(0),
            results[cluster_col]
        )
        print(f"  k={k} -> silhouette score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_k = k

    print(f"\n  Best: k={best_k} (silhouette={best_score:.4f})")

    # ── Assign clusters and save ───────────────────────────────────────────
    results = assign_model(best_model)

    out_path = os.path.join(PROJECT_ROOT, "outputs", "patient_phenotypes.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results.to_csv(out_path, index=False)

    model_path = os.path.join(PROJECT_ROOT, "models", "phenotype_model")
    save_model(best_model, model_path)

    print(f"\n[OK] Phenotype model saved to {model_path}")
    print(f"   Phenotypes exported to {out_path}")

    # ── Print cluster summaries ────────────────────────────────────────────
    cluster_col = [c for c in results.columns if "Cluster" in c][0]
    print(f"\nCluster distribution:")
    print(results[cluster_col].value_counts().sort_index())

    return best_model


if __name__ == "__main__":
    print("=" * 60)
    print("Task C: Patient Phenotyping (Clustering)")
    print("=" * 60)
    train_phenotype_model()
