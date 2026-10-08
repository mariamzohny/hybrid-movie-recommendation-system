# 🎬 Movie Recommendation System — Data Mining, Web Mining & NLP

> **Hybrid Netflix-style recommendation pipeline using FP-Growth association rules, PageRank graph analysis, and BERT semantic similarity, with an interactive Streamlit dashboard.**

This repository contains the integrated work of a **six-member team project** focused on movie recommendation using complementary techniques from **Data Mining, Web Mining, Graph Analysis, and Natural Language Processing (NLP)**.

The system combines user-behavior patterns, graph-based importance, and semantic similarity to recommend movies from different perspectives and present the results through a complete interactive GUI.

---

## ✨ Project Highlights

- **FP-Growth Association Rule Mining** to discover frequently watched movie combinations.
- **Support, Confidence, and Lift** analysis for meaningful movie relationships.
- **PageRank** over a directed weighted movie graph to identify influential movies.
- **BERT embeddings** for semantic comparison of movie overviews.
- **Cosine Similarity** for content-based movie recommendations.
- **Genre filtering** and movie coverage analysis.
- **Interactive network visualization** for recommendation relationships.
- **Streamlit dashboard** for recommendations, analytics, graph exploration, and data inspection.
- Uses **MovieLens 20M** and **TMDB 5000 Movies** datasets.

---

## 🧠 Recommendation Approaches

### 1. Association Rule Mining — FP-Growth

User rating histories are transformed into movie transactions. The project then applies FP-Growth to identify frequent itemsets and generate movie association rules.

The module evaluates relationships using:

- Support
- Confidence
- Lift
- Frequent itemsets
- Frequent movie combinations

These relationships provide recommendations based on movies that frequently appear together in users' positive-rating histories.

### 2. PageRank & Graph Analysis

Association rules are converted into a directed weighted recommendation graph where movies are nodes and recommendation relationships are edges.

PageRank is then used to estimate the relative importance of movies inside the network. The project also analyzes node connectivity, edge strength, user popularity, and graph structure.

### 3. BERT Semantic Similarity

Movie overviews are encoded with **BERT (`bert-base-uncased`)**. Mean-pooled contextual embeddings are compared using cosine similarity to find movies with semantically related descriptions.

This adds a content-based recommendation layer that does not depend only on user interaction patterns.

---

## 🖥️ Interactive Streamlit Application

The repository includes a complete Streamlit GUI with:

- 🏠 Dashboard
- 🔍 Movie Recommendations
- 🎭 Genre Filtering
- 🧠 BERT Recommendations
- 🌱 FP-Growth Recommendations
- 📡 PageRank Recommendations
- 🎯 Movie / Model Coverage
- 📊 Analytics & Visualizations
- 🕸️ Movie Recommendation Network
- 📋 Data Explorer

---

## 🏗️ System Architecture

```text
                 MovieLens 20M + TMDB 5000
                            │
                            ▼
                    Data Preprocessing
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
     FP-Growth       Recommendation Graph    BERT
 Association Rules         + PageRank      Embeddings
          │                 │                 │
          ▼                 ▼                 ▼
 Frequent Movie       Graph-Based       Semantic Similarity
  Relationships      Recommendations      Recommendations
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                   Streamlit Dashboard
```

---

## 📊 Datasets

### MovieLens 20M

Used mainly for user-movie interactions and rating behavior.

Key fields include:

- User ID
- Movie ID
- Rating
- Movie title
- Genres

### TMDB 5000 Movie Dataset

Used for movie metadata and text-based semantic analysis.

Relevant attributes include:

- Movie title
- Overview
- Genres
- Keywords
- Popularity
- Vote average / count
- Release information

> Raw datasets are **not committed to this repository**. `pipeline.py` downloads them using `kagglehub` when the pipeline is executed.

---

## 🛠️ Tech Stack

**Programming & Data Processing**  
Python · Pandas · NumPy

**Data Mining**  
MLxtend · FP-Growth · Association Rules

**Graph / Web Mining**  
NetworkX · PageRank

**NLP**  
BERT · Hugging Face Transformers · PyTorch · Scikit-learn · Cosine Similarity

**Visualization & Interface**  
Matplotlib · Streamlit

---

## 📁 Repository Structure

```text
hybrid-movie-recommendation-system/
│
├── app.py                      # Interactive Streamlit GUI
├── pipeline.py                 # End-to-end data/model pipeline
├── requirements.txt            # Python dependencies
├── .gitignore                  # Files excluded from Git
│
├── setup.bat                   # Create venv + install dependencies (Windows)
├── build_data.bat              # Run the full processing pipeline
├── run_app.bat                 # Launch the Streamlit GUI
├── start.bat                   # Quick launcher / first-time helper
│
├── notebooks/
│   └── team_project_original.ipynb
│
├── data/
│   └── processed/              # Generated CSV / GraphML outputs
│
├── models/
│   └── movie_bert_embeddings.npy   # Generated by pipeline.py
│
└── assets/
    └── screenshots/
```

---

## 🚀 Run Locally in VS Code

### Option A — Easy Windows Setup

Open the repository folder in **VS Code** and run:

```text
setup.bat
```

Then generate the data/model outputs:

```text
build_data.bat
```

Finally launch the application:

```text
run_app.bat
```

### Option B — VS Code Terminal

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the recommendation outputs:

```bash
python pipeline.py
```

Launch the GUI:

```bash
streamlit run app.py
```

### Build Without BERT

BERT embedding generation is the most computationally expensive step. To build only FP-Growth and PageRank:

```bash
python pipeline.py --skip-bert
```

---

## 📦 Generated Outputs

After running the complete pipeline, the project generates:

```text
data/processed/fpgrowth frequent itemsets.csv
data/processed/fpgrowth frequent movie combinations.csv
data/processed/fpgrowth frequent association rules.csv
data/processed/movie graph edges.csv
data/processed/pagerank movie ranking.csv
data/processed/top pagerank movies.csv
data/processed/movie recommendation graph.graphml
data/processed/bert_movies.csv
data/processed/bert_similarity_results.csv
models/movie_bert_embeddings.npy
```

These generated files are consumed by the Streamlit application.

---

## 👥 Team Project & Contributions

This project was developed collaboratively by a **six-member team**. The repository is published from a single account for easier academic and portfolio presentation; this does **not** imply that one member completed the full project alone.

Before publishing, replace the GitHub placeholders with the correct usernames and adjust any role wording if needed.

| # | Team Member | Main Responsibility | GitHub |
|---:|---|---|---|
| 1 | **[Member 1 Name]** | Data preparation / assigned Module 1 + GUI integration | `@[username]` |
| 2 | **[mariamzohny]** | Association Rule Mining — FP-Growth, frequent movie combinations, support, confidence & lift | `@[mariamzohny]` |
| 3 | **Ahmed Abdelrahim** | PageRank / Graph Analysis | `@[username]` |
| 4 | **Amr Fekar** | Data Visualization & Analytical Charts | `@[username]` |
| 5 | **Mohamed Ahmed** | BERT Semantic Similarity / NLP | `@[username]` |
| 6 | **Mazen** | **[Add exact assigned role]** | `@[username]` |

### Member 2 — Association Rule Mining Contribution

The Association Rule Mining module includes:

- Filtering positive user ratings.
- Converting user histories into movie transactions.
- Selecting frequent movies for efficient mining.
- One-hot encoding transactions with `TransactionEncoder`.
- Mining frequent itemsets with FP-Growth.
- Extracting frequent movie combinations.
- Generating association rules.
- Evaluating relationships with support, confidence, and lift.
- Filtering relationships with `lift > 1`.
- Ranking association-based recommendations.

---

## 📓 Original Team Notebook

The original integrated notebook is retained under:

```text
notebooks/team_project_original.ipynb
```

It preserves the original team workflow for academic documentation. The standalone `pipeline.py` reorganizes the same overall workflow into a VS Code-friendly executable pipeline for easier reuse and demonstration.

---

## 🔐 Files That Should Not Be Committed

The included `.gitignore` excludes common local and sensitive files such as:

```text
.venv/
venv/
__pycache__/
.env
kaggle.json
.streamlit/secrets.toml
.DS_Store
Thumbs.db
```

Never commit passwords, API keys, access tokens, or private credentials.

---

## 🔮 Possible Future Improvements

- Unified hybrid score combining FP-Growth, PageRank, and BERT.
- Personalized user profiles and collaborative filtering.
- Precision@K / Recall@K / NDCG recommendation evaluation.
- Movie posters and richer metadata integration.
- More interactive graph visualization.
- Public cloud deployment.
- Caching / vector indexing for faster semantic recommendations.

---

## 📌 Project Description

**Movie Recommendation System - Data Mining, Web Mining & NLP**

Created a hybrid Netflix-style recommendation pipeline using **Apriori/FP-Growth, PageRank popularity, and BERT semantic similarity**, with genre-distribution and frequent-pair visualizations for actionable insights.

---

### Academic / Portfolio Project

This repository represents the integrated output of a collaborative team project. Individual contributions should remain clearly credited whenever the project is published, presented, or reused.
