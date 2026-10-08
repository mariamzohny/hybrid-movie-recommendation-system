"""Hybrid Movie Recommendation System data pipeline.

This script converts the original Colab workflow into a VS Code-friendly Python
pipeline. It downloads the public TMDB 5000 and MovieLens 20M datasets, performs
preprocessing, builds FP-Growth association rules, constructs the PageRank graph,
and creates BERT movie-description embeddings used by the Streamlit GUI.

Run once before launching the GUI:
    python pipeline.py

Generated tabular outputs are saved under data/processed/ and BERT embeddings
under models/.
"""

from __future__ import annotations

import argparse
import ast
import os
import warnings
from pathlib import Path

import kagglehub
import networkx as nx
import numpy as np
import pandas as pd
import torch
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Parameters preserved from the team notebook.
POSITIVE_RATING = 4.0
TOP_N_MOVIES = 300
MAX_TRANSACTIONS = 20_000
MIN_SUPPORT = 0.05
MAX_ITEMSET_LENGTH = 3
MIN_CONFIDENCE = 0.25
PAGERANK_ALPHA = 0.85
BERT_MODEL_NAME = "bert-base-uncased"


def extract_names(text):
    """Parse TMDB list-of-dictionaries strings and return their names."""
    try:
        items = ast.literal_eval(text)
        return [item["name"] for item in items]
    except Exception:
        return []


def download_and_prepare_data():
    print("[1/5] Downloading datasets with kagglehub...")
    tmdb_path = Path(kagglehub.dataset_download("tmdb/tmdb-movie-metadata"))
    movielens_path = Path(kagglehub.dataset_download("grouplens/movielens-20m-dataset"))

    movies_tmdb = pd.read_csv(tmdb_path / "tmdb_5000_movies.csv")
    ratings = pd.read_csv(movielens_path / "rating.csv")
    movies_movielens = pd.read_csv(movielens_path / "movie.csv")
    links = pd.read_csv(movielens_path / "link.csv")

    movies_movielens["genres"] = movies_movielens["genres"].str.split("|")

    tmdb_selected = movies_tmdb[
        [
            "id",
            "title",
            "overview",
            "keywords",
            "genres",
            "vote_average",
            "vote_count",
            "popularity",
            "release_date",
            "runtime",
            "original_language",
        ]
    ].copy()

    tmdb_selected["keywords"] = tmdb_selected["keywords"].apply(extract_names)
    tmdb_selected["runtime"] = tmdb_selected["runtime"].fillna(tmdb_selected["runtime"].median())
    tmdb_selected["release_date"] = pd.to_datetime(tmdb_selected["release_date"], errors="coerce")
    tmdb_selected = tmdb_selected.dropna(subset=["release_date", "overview"]).copy()
    tmdb_selected["release_year"] = tmdb_selected["release_date"].dt.year
    tmdb_selected["release_month"] = tmdb_selected["release_date"].dt.month
    tmdb_selected["overview"] = tmdb_selected["overview"].str.strip()
    tmdb_selected = tmdb_selected.drop(columns=["release_date"])

    # MovieLens link table can contain missing TMDB IDs. Drop before integer cast.
    links = links.dropna(subset=["tmdbId"]).copy()
    links["tmdbId"] = links["tmdbId"].astype(int)
    tmdb_selected["id"] = tmdb_selected["id"].astype(int)

    merged = links.merge(tmdb_selected, left_on="tmdbId", right_on="id")
    merged = merged.merge(movies_movielens[["movieId", "genres"]], on="movieId", suffixes=("_tmdb", "_movielens"))
    merged = merged.drop_duplicates(subset="movieId")
    merged = merged.drop(columns=[c for c in ["id", "imdbId"] if c in merged.columns])

    # Keep a simple genres column for the GUI. Prefer MovieLens genres because it
    # already contains readable labels.
    if "genres_movielens" in merged.columns:
        merged["genres"] = merged["genres_movielens"]
    elif "genres" not in merged.columns and "genres_tmdb" in merged.columns:
        merged["genres"] = merged["genres_tmdb"].apply(extract_names)

    valid_movies = merged["movieId"].unique()
    ratings_filtered = ratings[ratings["movieId"].isin(valid_movies)].copy()
    ratings_filtered = ratings_filtered.drop(columns=["timestamp"], errors="ignore")

    print(f"Prepared {len(merged):,} matched movies and {len(ratings_filtered):,} ratings.")
    return merged, ratings_filtered


def build_fp_growth(merged, ratings_filtered):
    print("[2/5] Building FP-Growth association rules...")
    positive_ratings = ratings_filtered[ratings_filtered["rating"] >= POSITIVE_RATING].copy()
    positive_ratings = positive_ratings.merge(
        merged[["movieId", "title"]], on="movieId", how="left"
    ).dropna(subset=["title"])

    movie_transactions = (
        positive_ratings.groupby("userId")["title"]
        .apply(lambda x: sorted(set(x)))
        .reset_index(name="movies")
    )
    movie_transactions = movie_transactions[movie_transactions["movies"].apply(len) >= 2]

    movie_frequency = positive_ratings["title"].value_counts()
    top_movies = set(movie_frequency.head(TOP_N_MOVIES).index)
    filtered_transactions = movie_transactions["movies"].apply(
        lambda movies: [movie for movie in movies if movie in top_movies]
    )
    filtered_transactions = filtered_transactions[
        filtered_transactions.apply(len) >= 2
    ].tolist()

    if not filtered_transactions:
        raise RuntimeError("No transactions remain after filtering; cannot run FP-Growth.")

    encoder = TransactionEncoder()
    encoded_array = encoder.fit(filtered_transactions).transform(filtered_transactions)
    encoded_movies = pd.DataFrame(encoded_array, columns=encoder.columns_).astype(bool)

    if len(encoded_movies) > MAX_TRANSACTIONS:
        encoded_sample = encoded_movies.sample(MAX_TRANSACTIONS, random_state=42)
    else:
        encoded_sample = encoded_movies

    itemsets = fpgrowth(
        encoded_sample,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=MAX_ITEMSET_LENGTH,
    )
    itemsets["length"] = itemsets["itemsets"].apply(len)
    itemsets = itemsets.sort_values(["support", "length"], ascending=False).reset_index(drop=True)

    combinations = (
        itemsets[itemsets["length"] >= 2]
        .sort_values(["support", "length"], ascending=False)
        .reset_index(drop=True)
    )

    rules_raw = association_rules(
        itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE
    )
    rules = rules_raw[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    rules = rules[rules["lift"] > 1].reset_index(drop=True)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)

    itemsets.to_csv(DATA_DIR / "fpgrowth frequent itemsets.csv", index=False)
    combinations.to_csv(DATA_DIR / "fpgrowth frequent movie combinations.csv", index=False)
    rules.to_csv(DATA_DIR / "fpgrowth frequent association rules.csv", index=False)

    print(f"Saved {len(itemsets):,} itemsets and {len(rules):,} association rules.")
    return itemsets, rules_raw, rules, positive_ratings


def build_pagerank(merged, itemsets, positive_ratings):
    print("[3/5] Building PageRank recommendation graph...")
    raw_graph_rules = association_rules(
        itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE
    )
    raw_graph_rules = raw_graph_rules[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ].copy()
    raw_graph_rules = raw_graph_rules[
        (raw_graph_rules["lift"] > 1) & (raw_graph_rules["confidence"] > 0)
    ].reset_index(drop=True)

    edge_rows = []
    for _, row in raw_graph_rules.iterrows():
        for source_movie in list(row["antecedents"]):
            for target_movie in list(row["consequents"]):
                if source_movie != target_movie:
                    edge_rows.append(
                        {
                            "source": source_movie,
                            "target": target_movie,
                            "support": row["support"],
                            "confidence": row["confidence"],
                            "lift": row["lift"],
                            "weight": row["support"] * row["confidence"] * row["lift"],
                        }
                    )

    edges_df = pd.DataFrame(edge_rows)
    if edges_df.empty:
        raise RuntimeError("Association rules produced no graph edges.")

    edges_df = (
        edges_df.groupby(["source", "target"], as_index=False)
        .agg({"support": "max", "confidence": "max", "lift": "max", "weight": "sum"})
    )

    graph = nx.DiGraph()
    for _, row in edges_df.iterrows():
        graph.add_edge(
            row["source"],
            row["target"],
            weight=float(row["weight"]),
            support=float(row["support"]),
            confidence=float(row["confidence"]),
            lift=float(row["lift"]),
        )

    pagerank_scores = nx.pagerank(graph, alpha=PAGERANK_ALPHA, weight="weight")
    pagerank_df = pd.DataFrame(
        {"movie": list(pagerank_scores.keys()), "pagerank_score": list(pagerank_scores.values())}
    )
    pagerank_df["in_degree"] = pagerank_df["movie"].apply(graph.in_degree)
    pagerank_df["out_degree"] = pagerank_df["movie"].apply(graph.out_degree)
    pagerank_df["total_degree"] = pagerank_df["in_degree"] + pagerank_df["out_degree"]
    pagerank_df["weighted_in_degree"] = pagerank_df["movie"].apply(
        lambda x: sum(d["weight"] for _, _, d in graph.in_edges(x, data=True))
    )
    pagerank_df["weighted_out_degree"] = pagerank_df["movie"].apply(
        lambda x: sum(d["weight"] for _, _, d in graph.out_edges(x, data=True))
    )
    pagerank_df = pagerank_df.sort_values("pagerank_score", ascending=False).reset_index(drop=True)

    movie_rating_stats = (
        positive_ratings.groupby("title")
        .agg(avg_user_rating=("rating", "mean"), liked_by_users=("userId", "nunique"))
        .reset_index()
    )
    metadata_cols = ["title", "popularity", "vote_average", "vote_count", "release_year"]
    metadata_available = [c for c in metadata_cols if c in merged.columns]
    movie_metadata = merged[metadata_available].drop_duplicates(subset="title")

    ranking = (
        pagerank_df.merge(movie_rating_stats, left_on="movie", right_on="title", how="left")
        .merge(movie_metadata, left_on="movie", right_on="title", how="left", suffixes=("", "_metadata"))
    )
    ranking = ranking.drop(columns=[c for c in ["title", "title_metadata"] if c in ranking.columns])
    ranking = ranking.sort_values(
        ["pagerank_score", "weighted_in_degree", "liked_by_users"], ascending=False
    ).reset_index(drop=True)
    top_ranking = ranking.head(20).copy()

    edges_df.to_csv(DATA_DIR / "movie graph edges.csv", index=False)
    ranking.to_csv(DATA_DIR / "pagerank movie ranking.csv", index=False)
    top_ranking.to_csv(DATA_DIR / "top pagerank movies.csv", index=False)
    nx.write_graphml(graph, DATA_DIR / "movie recommendation graph.graphml")

    print(f"Saved graph with {graph.number_of_nodes():,} nodes and {graph.number_of_edges():,} edges.")
    return graph, ranking


def _compute_embeddings(texts, tokenizer, model, device, batch_size=32):
    all_embeddings = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1e-9)
        all_embeddings.append((summed / count).cpu().numpy())
        print(f"  BERT: {min(start + batch_size, total):,}/{total:,}", end="\r")
    print()
    return np.vstack(all_embeddings)


def build_bert(merged, batch_size=32):
    print("[4/5] Building BERT semantic embeddings...")
    columns = ["movieId", "title", "overview"]
    if "genres" in merged.columns:
        columns.append("genres")
    bert_df = merged[columns].dropna(subset=["title", "overview"]).drop_duplicates(subset=["movieId"]).reset_index(drop=True)

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertModel.from_pretrained(BERT_MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print("BERT device:", device)

    embeddings = _compute_embeddings(
        bert_df["overview"].fillna("").tolist(), tokenizer, model, device, batch_size=batch_size
    )
    np.save(MODELS_DIR / "movie_bert_embeddings.npy", embeddings)

    # Store genres as a Python-list string, which the Streamlit GUI can parse.
    bert_df.to_csv(DATA_DIR / "bert_movies.csv", index=False)

    similarity_matrix = cosine_similarity(embeddings)
    records = []
    for idx, row in bert_df.iterrows():
        scores = sorted(enumerate(similarity_matrix[idx]), key=lambda x: x[1], reverse=True)
        top = [(bert_df.iloc[i]["title"], round(float(score), 4)) for i, score in scores if i != idx][:5]
        records.append({"movie": row["title"], "top_similar": top})
    pd.DataFrame(records).to_csv(DATA_DIR / "bert_similarity_results.csv", index=False)

    print(f"Saved BERT embeddings for {len(bert_df):,} movies.")


def verify_outputs():
    print("[5/5] Verifying generated project files...")
    required = [
        DATA_DIR / "fpgrowth frequent association rules.csv",
        DATA_DIR / "fpgrowth frequent itemsets.csv",
        DATA_DIR / "fpgrowth frequent movie combinations.csv",
        DATA_DIR / "pagerank movie ranking.csv",
        DATA_DIR / "top pagerank movies.csv",
        DATA_DIR / "movie graph edges.csv",
        DATA_DIR / "bert_similarity_results.csv",
        DATA_DIR / "bert_movies.csv",
        MODELS_DIR / "movie_bert_embeddings.npy",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("Missing outputs:")
        for p in missing:
            print(" -", p.relative_to(BASE_DIR))
        return False
    print("All GUI data files are ready.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build all movie recommendation system outputs.")
    parser.add_argument("--skip-bert", action="store_true", help="Build FP-Growth and PageRank only.")
    parser.add_argument("--bert-batch-size", type=int, default=32, help="BERT inference batch size.")
    args = parser.parse_args()

    merged, ratings_filtered = download_and_prepare_data()
    itemsets, _, _, positive_ratings = build_fp_growth(merged, ratings_filtered)
    build_pagerank(merged, itemsets, positive_ratings)
    if not args.skip_bert:
        build_bert(merged, batch_size=args.bert_batch_size)
    else:
        print("[4/5] BERT skipped by user request.")
    verify_outputs()
    print("\nDone. Launch the GUI with:\n  streamlit run app.py")


if __name__ == "__main__":
    main()
