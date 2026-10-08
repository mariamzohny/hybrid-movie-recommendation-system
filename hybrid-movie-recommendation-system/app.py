import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import warnings
import os
import ast
import re

warnings.filterwarnings("ignore")

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def project_file(filename):
    """Return the organized project path, with root fallback for legacy files."""
    organized = os.path.join(DATA_DIR, filename)
    if os.path.exists(organized):
        return organized
    return os.path.join(BASE_DIR, filename)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    border-right: 1px solid #2a2a4a;
}
[data-testid="stSidebar"] * {
    color: #c9c9e8 !important;
}

/* Main background */
.stApp {
    background: #0d0d1a;
    color: #e0e0f0;
}

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #1a0533 0%, #0d1b4a 50%, #001a2e 100%);
    border: 1px solid #3d2a6e;
    border-radius: 16px;
    padding: 40px 50px;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(138,43,226,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #bb86fc, #03dac6, #00f260);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    color: #8888aa;
    font-size: 1.1rem;
    margin-top: 10px;
    font-weight: 300;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 20px 28px;
    flex: 1;
    min-width: 160px;
    text-align: center;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #03dac6;
}
.metric-label {
    font-size: 0.8rem;
    color: #7777aa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Section cards */
.section-card {
    background: #12122a;
    border: 1px solid #252545;
    border-radius: 14px;
    padding: 28px;
    margin-bottom: 20px;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #bb86fc;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Table styling */
.dataframe {
    background: #1a1a2e !important;
    color: #e0e0f0 !important;
    border-radius: 10px;
    font-size: 0.85rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #bb86fc, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    transition: all 0.3s;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(187,134,252,0.4);
}

/* Selectbox */
.stSelectbox > div > div {
    background: #1a1a2e;
    border: 1px solid #3a3a6a;
    color: #e0e0f0;
    border-radius: 8px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #12122a;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8888aa;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #bb86fc22, #03dac622);
    color: #bb86fc !important;
    border: 1px solid #bb86fc44;
}

/* Slider */
.stSlider > div { color: #e0e0f0; }

/* Info/success boxes */
.stAlert {
    border-radius: 10px;
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
}

/* Expander */
.streamlit-expanderHeader {
    background: #1a1a2e !important;
    color: #bb86fc !important;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
}

/* Custom badge */
.badge {
    display: inline-block;
    background: linear-gradient(90deg, #bb86fc22, #03dac622);
    border: 1px solid #bb86fc55;
    color: #bb86fc;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* Recommendation result card */
.rec-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-left: 4px solid #03dac6;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.rec-title { 
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    color: #e0e0f0;
    font-size: 1rem;
}
.rec-score {
    font-size: 0.8rem;
    color: #03dac6;
    margin-top: 4px;
}

hr { border-color: #2a2a4a !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
DARK_BG   = '#0d0d1a'
CARD_BG   = '#12122a'
TEXT_CLR  = '#c9c9e8'
ACCENT1   = '#bb86fc'
ACCENT2   = '#03dac6'
ACCENT3   = '#00f260'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   CARD_BG,
    'axes.edgecolor':   '#2a2a4a',
    'axes.labelcolor':  TEXT_CLR,
    'xtick.color':      TEXT_CLR,
    'ytick.color':      TEXT_CLR,
    'text.color':       TEXT_CLR,
    'grid.color':       '#2a2a4a',
    'grid.alpha':       0.4,
})

# ─────────────────────────────────────────────
# DATA LOADING  (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    files = {
        "fpgrowth_rules":        "fpgrowth frequent association rules.csv",
        "fpgrowth_itemsets":     "fpgrowth frequent itemsets.csv",
        "fpgrowth_combinations": "fpgrowth frequent movie combinations.csv",
        "pagerank_ranking":      "pagerank movie ranking.csv",
        "top_pagerank":          "top pagerank movies.csv",
        "graph_edges":           "movie graph edges.csv",
        "bert_similarity":       "bert_similarity_results.csv",
        "bert_movies":           "bert_movies.csv",
    }
    data = {}
    for key, fname in files.items():
        path = project_file(fname)
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            data[key] = None
    return data

@st.cache_data(show_spinner=False)
def load_graph(edges_df):
    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        G.add_edge(row['source'], row['target'],
                   weight=float(row['weight']),
                   confidence=float(row['confidence']),
                   lift=float(row['lift']))
    return G

@st.cache_data(show_spinner=False)
def build_graph_from_rules(rules_df):
    """Build graph from FP-Growth rules when edges CSV is missing."""
    G = nx.DiGraph()
    for _, row in rules_df.iterrows():
        src = row['antecedents']
        tgt = row['consequents']
        w = float(row['support']) * float(row['confidence']) * float(row['lift'])
        G.add_edge(src, tgt,
                   weight=w,
                   confidence=float(row['confidence']),
                   lift=float(row['lift']))
    return G

@st.cache_data(show_spinner=False)
def load_bert_embeddings():
    embedding_path = os.path.join(MODELS_DIR, "movie_bert_embeddings.npy")
    if not os.path.exists(embedding_path):
        embedding_path = os.path.join(BASE_DIR, "movie_bert_embeddings.npy")
    if os.path.exists(embedding_path):
        return np.load(embedding_path)
    return None

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def recommend_fp(movie_title, rules_df, top_n=10):
    recs = rules_df[
        rules_df['antecedents'].str.contains(movie_title, case=False, regex=False)
    ].copy()
    recs = recs.sort_values(['lift', 'confidence', 'support'], ascending=False)
    recs = recs.drop_duplicates(subset=['consequents'])
    return recs[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(top_n)


def recommend_pagerank(movie_title, graph, pagerank_df, top_n=10):
    if movie_title not in graph.nodes:
        return None
    edges = []
    for _, tgt, data in graph.out_edges(movie_title, data=True):
        edges.append({'recommended_movie': tgt,
                      'edge_weight': data.get('weight', 0),
                      'confidence':  data.get('confidence', 0),
                      'lift':        data.get('lift', 0)})
    if not edges:
        return None
    rdf = pd.DataFrame(edges)
    rdf = rdf.merge(pagerank_df[['movie','pagerank_score']],
                    left_on='recommended_movie', right_on='movie', how='left').drop(columns='movie')
    return rdf.sort_values(['edge_weight','pagerank_score'], ascending=False).head(top_n)


def recommend_bert(movie_title, bert_df, sim_matrix, top_n=10):
    matches = bert_df[bert_df['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        return None
    idx = matches.index[0]
    scores = sorted(enumerate(sim_matrix[idx]), key=lambda x: x[1], reverse=True)[1:top_n+1]
    return pd.DataFrame([{
        'recommended_movie': bert_df.iloc[i]['title'],
        'similarity_score':  round(s, 4)
    } for i, s in scores])


def parse_itemset(x):
    """Parse itemset from CSV - handles frozenset(...) format and quoted strings."""
    if not isinstance(x, str):
        return []
    # frozenset({'Movie A', 'Movie B'}) format
    matches = re.findall(r"'([^']*)'", x)
    if matches:
        return matches
    # fallback: try literal_eval
    try:
        result = ast.literal_eval(x)
        return list(result)
    except:
        return [x]


def make_hbar(titles, values, title, color_main, color_rest, xlabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    clrs = [color_main if i == len(values)-1 else color_rest for i in range(len(values))]
    ax.barh(titles, values, color=clrs, edgecolor='none', height=0.6)
    ax.set_title(title, fontsize=13, fontweight='bold', color=ACCENT3, pad=12)
    ax.set_xlabel(xlabel, color=TEXT_CLR)
    ax.grid(axis='x', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800;
                    background:linear-gradient(90deg,#bb86fc,#03dac6);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            🎬 Movie Recommender
        </div>
        <div style="color:#555577; font-size:0.8rem; letter-spacing:2px; margin-top:4px;">
            DATA MINING · GRAPH ANALYSIS · NLP
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Dashboard",
        "🔍 Get Recommendations",
        "🎯 Movie Coverage",
        "📊 Visualizations",
        "🕸️ Network Graph",
        "📋 Data Explorer",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#444466; line-height:1.8; padding:10px;">
        <b style="color:#bb86fc;">Models Used</b><br>
        🧠 BERT Semantic Similarity<br>
        🌱 FP-Growth Association Rules<br>
        📡 PageRank Graph Scoring<br><br>
        <b style="color:#bb86fc;">Datasets</b><br>
        📦 TMDB 5000 Movies<br>
        🎯 MovieLens 20M Ratings
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Loading data..."):
    data = load_data()

missing_outputs = [key for key, value in data.items() if value is None]
if missing_outputs:
    st.sidebar.warning(
        "Some generated data files are missing. Run `python pipeline.py` once to build the project outputs."
    )

graph = None
if data["graph_edges"] is not None:
    graph = load_graph(data["graph_edges"])
elif data["fpgrowth_rules"] is not None:
    graph = build_graph_from_rules(data["fpgrowth_rules"])

bert_embeddings = load_bert_embeddings()
sim_matrix = None
if bert_embeddings is not None:
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(bert_embeddings)

# Build pagerank from graph if pagerank_ranking CSV is missing
if data["pagerank_ranking"] is None and graph is not None:
    pr = nx.pagerank(graph, alpha=0.85, weight='weight')
    data["pagerank_ranking"] = pd.DataFrame({
        'movie': list(pr.keys()),
        'pagerank_score': list(pr.values())
    }).sort_values('pagerank_score', ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════
if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">Movie Recommendation<br>System</div>
        <div class="hero-sub">Data Mining · Web Mining · Graph Analysis · NLP</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics ──
    n_rules   = len(data["fpgrowth_rules"])  if data["fpgrowth_rules"]  is not None else 0
    n_movies  = len(data["pagerank_ranking"]) if data["pagerank_ranking"] is not None else 0
    n_nodes   = graph.number_of_nodes()      if graph else 0
    n_edges   = graph.number_of_edges()      if graph else 0
    bert_size = len(data["bert_movies"])     if data["bert_movies"] is not None else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in [
        (c1, f"{n_rules:,}",  "Association Rules"),
        (c2, f"{n_movies:,}", "Ranked Movies"),
        (c3, f"{n_nodes:,}",  "Graph Nodes"),
        (c4, f"{n_edges:,}",  "Graph Edges"),
        (c5, f"{bert_size:,}","BERT Movies"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top PageRank ──
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">🏆 Top 10 Movies by PageRank</div>', unsafe_allow_html=True)
        if data["pagerank_ranking"] is not None:
            top10 = data["pagerank_ranking"].head(10)[['movie','pagerank_score','liked_by_users']].copy()
            top10['pagerank_score'] = top10['pagerank_score'].round(6)
            st.dataframe(top10, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown('<div class="section-title">🌱 Top Association Rules (by Lift)</div>', unsafe_allow_html=True)
        if data["fpgrowth_rules"] is not None:
            top_rules = data["fpgrowth_rules"].head(10)[
                ['antecedents','consequents','confidence','lift']].copy()
            top_rules['confidence'] = top_rules['confidence'].round(3)
            top_rules['lift'] = top_rules['lift'].round(3)
            st.dataframe(top_rules, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PageRank bar chart (quick) ──
    if data["pagerank_ranking"] is not None:
        st.markdown('<div class="section-title">📈 PageRank Score — Top 15</div>', unsafe_allow_html=True)
        top15 = data["pagerank_ranking"].head(15)
        fig = make_hbar(
            top15['movie'][::-1].tolist(),
            top15['pagerank_score'][::-1].tolist(),
            "Top 15 Movies by PageRank Score",
            ACCENT1, ACCENT2, "PageRank Score"
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════
# PAGE: GET RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════
elif page == "🔍 Get Recommendations":

    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#bb86fc; margin-bottom:6px;">
        🔍 Movie Recommender
    </div>
    <div style="color:#7777aa; margin-bottom:24px;">
        Find similar movies using BERT, FP-Growth, or PageRank
    </div>
    """, unsafe_allow_html=True)

    # ── Build genre list from bert_movies ──
    all_genres = ["🎬 All Genres"]
    genre_map  = {}   # movie -> list of genres

    if data["bert_movies"] is not None and 'genres' in data["bert_movies"].columns:
        import ast as _ast
        def _parse_genres(g):
            if isinstance(g, list): return g
            if not isinstance(g, str): return []
            try:
                parsed = _ast.literal_eval(g)
                if isinstance(parsed, list):
                    return [i['name'] if isinstance(i, dict) else str(i) for i in parsed]
            except:
                pass
            return [x.strip() for x in g.replace('[','').replace(']','').replace("'",'').split(',') if x.strip()]

        bm = data["bert_movies"].copy()
        bm['_genres'] = bm['genres'].apply(_parse_genres)
        for _, row in bm.iterrows():
            genre_map[row['title']] = row['_genres']
        genre_set = sorted(set(g for gs in genre_map.values() for g in gs if g))
        all_genres += genre_set

    # ── Movie selector ──
    all_movies = []
    if data["bert_movies"] is not None:
        all_movies = sorted(data["bert_movies"]['title'].dropna().unique().tolist())
    elif data["pagerank_ranking"] is not None:
        all_movies = sorted(data["pagerank_ranking"]['movie'].dropna().unique().tolist())

    # ── Controls row ──
    col_genre, col_sel, col_n, col_btn = st.columns([2, 3, 1, 1])

    with col_genre:
        selected_genre = st.selectbox("🎭 Filter by Genre", all_genres)

    # Filter movies by genre
    if selected_genre != "🎬 All Genres" and genre_map:
        filtered_movies = sorted([m for m, gs in genre_map.items() if selected_genre in gs])
        if not filtered_movies:
            filtered_movies = all_movies
    else:
        filtered_movies = all_movies

    with col_sel:
        selected_movie = st.selectbox("🎬 Select a Movie", filtered_movies if filtered_movies else ["No data loaded"])
    with col_n:
        top_n = st.slider("Top N", 5, 20, 10)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("🚀 Recommend")

    # ── Show selected movie genres ──
    if selected_movie and genre_map:
        movie_genres = genre_map.get(selected_movie, [])
        if movie_genres:
            badges = " ".join([f'<span class="badge">{g}</span>' for g in movie_genres])
            st.markdown(f"**Genres:** {badges}", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    def filter_by_genre(recs_df, title_col, genre_filter, gmap):
        """Filter recommendations dataframe by genre."""
        if genre_filter == "🎬 All Genres" or not gmap:
            return recs_df
        return recs_df[recs_df[title_col].apply(
            lambda t: genre_filter in gmap.get(t, [])
        )].reset_index(drop=True)

    if run and selected_movie:
        tab1, tab2, tab3 = st.tabs(["🧠 BERT", "🌱 FP-Growth", "📡 PageRank"])

        # ── BERT ──
        with tab1:
            st.markdown(f'<div class="section-title">🧠 BERT — Top {top_n} Similar to "{selected_movie}"</div>',
                        unsafe_allow_html=True)
            if sim_matrix is not None and data["bert_movies"] is not None:
                recs = recommend_bert(selected_movie, data["bert_movies"], sim_matrix, top_n * 3)
                if recs is not None:
                    recs = filter_by_genre(recs, 'recommended_movie', selected_genre, genre_map).head(top_n)
                    if recs.empty:
                        st.info(f"No BERT results found for genre: {selected_genre}")
                    else:
                        for _, row in recs.iterrows():
                            bar_w = int(row['similarity_score'] * 100)
                            mg = genre_map.get(row['recommended_movie'], [])
                            genre_tags = " ".join([f'<span class="badge" style="font-size:0.65rem;">{g}</span>' for g in mg[:3]])
                            st.markdown(f"""
                            <div class="rec-card">
                                <div class="rec-title">🎬 {row['recommended_movie']}</div>
                                <div style="margin:4px 0;">{genre_tags}</div>
                                <div class="rec-score">Similarity: {row['similarity_score']:.4f}</div>
                                <div style="background:#2a2a4a;border-radius:4px;height:6px;margin-top:8px;">
                                    <div style="background:linear-gradient(90deg,#bb86fc,#03dac6);
                                                width:{bar_w}%;height:6px;border-radius:4px;"></div>
                                </div>
                            </div>""", unsafe_allow_html=True)

                        fig, ax = plt.subplots(figsize=(9, 4))
                        colors = plt.cm.cool(np.linspace(0.3, 0.9, len(recs)))
                        ax.barh(recs['recommended_movie'], recs['similarity_score'],
                                color=colors, edgecolor='none', height=0.6)
                        genre_label = f" [{selected_genre}]" if selected_genre != "🎬 All Genres" else ""
                        ax.set_title(f'BERT Similarity to "{selected_movie}"{genre_label}',
                                     color=ACCENT3, fontsize=12, fontweight='bold')
                        ax.set_xlabel('Cosine Similarity')
                        for spine in ax.spines.values(): spine.set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                else:
                    st.warning("Movie not found in BERT index.")
            else:
                st.warning("⚠️ BERT embeddings not found.")

        # ── FP-Growth ──
        with tab2:
            st.markdown(f'<div class="section-title">🌱 FP-Growth — Rules for "{selected_movie}"</div>',
                        unsafe_allow_html=True)
            if data["fpgrowth_rules"] is not None:
                recs = recommend_fp(selected_movie, data["fpgrowth_rules"], top_n * 3)
                if not recs.empty:
                    recs = filter_by_genre(recs, 'consequents', selected_genre, genre_map).head(top_n)
                    if recs.empty:
                        st.info(f"No FP-Growth results found for genre: {selected_genre}")
                    else:
                        for _, row in recs.iterrows():
                            mg = genre_map.get(row['consequents'], [])
                            genre_tags = " ".join([f'<span class="badge" style="font-size:0.65rem;">{g}</span>' for g in mg[:3]])
                            st.markdown(f"""
                            <div class="rec-card" style="border-left-color:#00f260;">
                                <div class="rec-title">🎬 {row['consequents']}</div>
                                <div style="margin:4px 0;">{genre_tags}</div>
                                <div class="rec-score">
                                    Confidence: {row['confidence']:.3f} &nbsp;|&nbsp;
                                    Lift: {row['lift']:.3f} &nbsp;|&nbsp;
                                    Support: {row['support']:.4f}
                                </div>
                            </div>""", unsafe_allow_html=True)
                        st.dataframe(recs, use_container_width=True, hide_index=True)
                else:
                    st.info("No FP-Growth rules found for this movie.")
            else:
                st.warning("⚠️ FP-Growth rules CSV not found.")

        # ── PageRank ──
        with tab3:
            st.markdown(f'<div class="section-title">📡 PageRank — Graph Links for "{selected_movie}"</div>',
                        unsafe_allow_html=True)
            if graph is not None and data["pagerank_ranking"] is not None:
                recs = recommend_pagerank(selected_movie, graph, data["pagerank_ranking"], top_n * 3)
                if recs is not None:
                    recs = filter_by_genre(recs, 'recommended_movie', selected_genre, genre_map).head(top_n)
                    if recs.empty:
                        st.info(f"No PageRank results found for genre: {selected_genre}")
                    else:
                        for _, row in recs.iterrows():
                            mg = genre_map.get(row['recommended_movie'], [])
                            genre_tags = " ".join([f'<span class="badge" style="font-size:0.65rem;">{g}</span>' for g in mg[:3]])
                            st.markdown(f"""
                            <div class="rec-card" style="border-left-color:#bb86fc;">
                                <div class="rec-title">🎬 {row['recommended_movie']}</div>
                                <div style="margin:4px 0;">{genre_tags}</div>
                                <div class="rec-score">
                                    PageRank: {row['pagerank_score']:.6f} &nbsp;|&nbsp;
                                    Weight: {row['edge_weight']:.4f}
                                </div>
                            </div>""", unsafe_allow_html=True)
                        st.dataframe(recs, use_container_width=True, hide_index=True)
                else:
                    st.info(f'"{selected_movie}" not in recommendation graph.')
            else:
                st.warning("⚠️ Graph data not found.")

    elif not run:
        st.info("👆 Select a movie and click **Recommend** to get started.")


# ═══════════════════════════════════════════════════════════
# PAGE: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════
elif page == "📊 Visualizations":

    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#bb86fc; margin-bottom:20px;">
        📊 Analytics & Visualizations
    </div>
    """, unsafe_allow_html=True)

    viz_choice = st.selectbox("Select Chart", [
        "Top 15 Movies by PageRank",
        "Most Frequent Movies (Support)",
        "Top Movie Combinations",
        "Most Liked Movies (Users)",
        "Top Rules by Lift",
        "BERT Similarity Distribution",
    ])

    # ── Chart 1 ──
    if viz_choice == "Top 15 Movies by PageRank":
        if data["pagerank_ranking"] is not None:
            df = data["pagerank_ranking"].head(15)
            fig = make_hbar(df['movie'][::-1].tolist(),
                            df['pagerank_score'][::-1].tolist(),
                            "Top 15 Movies by PageRank Score",
                            ACCENT1, ACCENT2, "PageRank Score")
            st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ── Chart 2 ──
    elif viz_choice == "Most Frequent Movies (Support)":
        if data["fpgrowth_itemsets"] is not None:
            df = data["fpgrowth_itemsets"].copy()
            df['parsed'] = df['itemsets'].apply(parse_itemset)
            df['length'] = df['parsed'].apply(len)
            singles = df[df['length'] == 1].sort_values('support', ascending=False).head(15)
            singles['movie'] = singles['parsed'].apply(lambda x: x[0] if x else '')
            fig = make_hbar(singles['movie'][::-1].tolist(),
                            singles['support'][::-1].tolist(),
                            "Top 15 Most Frequent Movies",
                            '#ff4c4c', ACCENT2, "Support")
            st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ── Chart 3 ──
    elif viz_choice == "Top Movie Combinations":
        if data["fpgrowth_combinations"] is not None:
            df = data["fpgrowth_combinations"].head(15).copy()
            df['parsed'] = df['itemsets'].apply(parse_itemset)
            df['pair'] = df['parsed'].apply(lambda x: ' + '.join(x) if x else '')
            fig = make_hbar(df['pair'][::-1].tolist(),
                            df['support'][::-1].tolist(),
                            "Top Frequent Movie Combinations",
                            '#ff4c4c', ACCENT2, "Support")
            fig.set_size_inches(12, 7)
            st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ── Chart 4 ──
    elif viz_choice == "Most Liked Movies (Users)":
        if data["pagerank_ranking"] is not None:
            df = data["pagerank_ranking"].dropna(subset=['liked_by_users']) \
                .sort_values('liked_by_users', ascending=False).head(15)
            fig = make_hbar(df['movie'][::-1].tolist(),
                            df['liked_by_users'][::-1].tolist(),
                            "Most Liked Movies (by User Count)",
                            '#ff4c4c', ACCENT2, "Number of Users")
            st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ── Chart 5 ──
    elif viz_choice == "Top Rules by Lift":
        if data["fpgrowth_rules"] is not None:
            df = data["fpgrowth_rules"].head(15)
            fig, ax = plt.subplots(figsize=(10, 5))
            sc = ax.scatter(df['confidence'], df['lift'],
                            c=df['support'], cmap='cool',
                            s=120, alpha=0.85, edgecolors='none')
            plt.colorbar(sc, ax=ax, label='Support')
            ax.set_xlabel('Confidence'); ax.set_ylabel('Lift')
            ax.set_title("Top Rules — Confidence vs Lift (color = Support)",
                         color=ACCENT3, fontsize=12, fontweight='bold')
            for spine in ax.spines.values(): spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ── Chart 6 ──
    elif viz_choice == "BERT Similarity Distribution":
        if sim_matrix is not None:
            all_scores = []
            for i in range(min(len(sim_matrix), 500)):
                s = sorted(sim_matrix[i], reverse=True)[1:6]
                all_scores.extend(s)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(all_scores, bins=40, color=ACCENT1, edgecolor='none', alpha=0.85)
            mean_s = np.mean(all_scores)
            ax.axvline(mean_s, color=ACCENT3, linewidth=2, linestyle='--',
                       label=f'Mean: {mean_s:.4f}')
            ax.set_title("BERT — Distribution of Top-5 Similarity Scores",
                         color=ACCENT3, fontsize=12, fontweight='bold')
            ax.set_xlabel('Similarity Score'); ax.set_ylabel('Frequency')
            ax.legend(facecolor=CARD_BG, labelcolor=TEXT_CLR)
            for spine in ax.spines.values(): spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        else:
            st.warning("⚠️ BERT embeddings not found.")


# ═══════════════════════════════════════════════════════════
# PAGE: NETWORK GRAPH
# ═══════════════════════════════════════════════════════════
elif page == "🕸️ Network Graph":

    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#bb86fc; margin-bottom:20px;">
        🕸️ Movie Recommendation Network
    </div>
    """, unsafe_allow_html=True)

    if graph is None:
        st.warning("⚠️ Graph data not loaded. Make sure `movie graph edges.csv` exists.")
    else:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            top_k = st.slider("Top N nodes by PageRank", 5, 30, 10)
            seed  = st.slider("Layout seed", 0, 100, 42)
            show_labels = st.checkbox("Show labels", True)

        with col_b:
            if data["pagerank_ranking"] is not None:
                top_nodes = data["pagerank_ranking"].head(top_k)['movie'].tolist()
                subg = graph.subgraph(top_nodes)

                fig, ax = plt.subplots(figsize=(12, 9))
                pos = nx.spring_layout(subg, seed=seed, k=1.5)

                pr_scores = nx.pagerank(subg, alpha=0.85) if len(subg.nodes) > 0 else {}
                sizes = [pr_scores.get(n, 0.001) * 60000 for n in subg.nodes()]
                
                nx.draw_networkx_nodes(subg, pos, node_size=sizes,
                                       node_color=ACCENT1, alpha=0.85, ax=ax)
                nx.draw_networkx_edges(subg, pos, alpha=0.35, arrows=True,
                                       edge_color=ACCENT2, ax=ax,
                                       arrowsize=15, width=1.2)
                if show_labels:
                    labels = {n: n[:14]+"…" if len(n)>14 else n for n in subg.nodes()}
                    nx.draw_networkx_labels(subg, pos, labels, font_size=8,
                                            font_color=TEXT_CLR, ax=ax)
                ax.set_title(f"Top {top_k} Movies — Recommendation Graph",
                             color=ACCENT3, fontsize=13, fontweight='bold')
                ax.axis('off')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        st.markdown("---")
        st.markdown('<div class="section-title">🔗 Graph Edge Data (top 50)</div>', unsafe_allow_html=True)
        if data["graph_edges"] is not None:
            st.dataframe(data["graph_edges"].head(50), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ═══════════════════════════════════════════════════════════
elif page == "📋 Data Explorer":

    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#bb86fc; margin-bottom:20px;">
        📋 Data Explorer
    </div>
    """, unsafe_allow_html=True)

    dataset_map = {
        "Association Rules (FP-Growth)":    "fpgrowth_rules",
        "Frequent Itemsets":                "fpgrowth_itemsets",
        "Frequent Combinations":            "fpgrowth_combinations",
        "PageRank Ranking":                 "pagerank_ranking",
        "Top PageRank Movies":              "top_pagerank",
        "Graph Edges":                      "graph_edges",
        "BERT Movies":                      "bert_movies",
        "BERT Similarity Results":          "bert_similarity",
    }

    chosen = st.selectbox("📦 Select Dataset", list(dataset_map.keys()))
    key = dataset_map[chosen]
    df  = data[key]

    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

        with st.expander("🔎 Column Info"):
            info = pd.DataFrame({'dtype': df.dtypes, 'nulls': df.isnull().sum(),
                                  'unique': df.nunique()})
            st.dataframe(info, use_container_width=True)

        search = st.text_input("🔍 Filter rows (any column contains...)")
        display_df = df.copy()
        if search:
            mask = display_df.apply(
                lambda col: col.astype(str).str.contains(search, case=False, na=False)).any(axis=1)
            display_df = display_df[mask]

        n_rows = st.slider("Rows to display", 10, 200, 50)
        st.dataframe(display_df.head(n_rows), use_container_width=True, hide_index=True)

        csv = display_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download as CSV", csv,
                           file_name=f"{chosen.replace(' ','_')}.csv",
                           mime="text/csv")
    else:
        st.warning(f"⚠️ Dataset `{chosen}` not found. Make sure the corresponding CSV file is in the same directory as this app.")


# ═══════════════════════════════════════════════════════════
# PAGE: MOVIE COVERAGE
# ═══════════════════════════════════════════════════════════
elif page == "🎯 Movie Coverage":

    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#bb86fc; margin-bottom:6px;">
        🎯 Movie Coverage
    </div>
    <div style="color:#7777aa; margin-bottom:24px;">
        شوف الأفلام المتاحة لكل model
    </div>
    """, unsafe_allow_html=True)

    # ── Collect available movies per model ──
    bert_movies_set = set()
    fp_movies_set   = set()
    pr_movies_set   = set()

    if data["bert_movies"] is not None:
        bert_movies_set = set(data["bert_movies"]['title'].dropna().tolist())

    if data["fpgrowth_rules"] is not None:
        for _, row in data["fpgrowth_rules"].iterrows():
            fp_movies_set.add(row['antecedents'])
            fp_movies_set.add(row['consequents'])

    if graph is not None:
        pr_movies_set = set(graph.nodes())

    all_movies_set = bert_movies_set | fp_movies_set | pr_movies_set

    # ── Coverage Summary Cards ──
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl, color in [
        (c1, len(all_movies_set),  "Total Movies",     "#bb86fc"),
        (c2, len(bert_movies_set), "BERT Movies",      "#03dac6"),
        (c3, len(fp_movies_set),   "FP-Growth Movies", "#00f260"),
        (c4, len(pr_movies_set),   "PageRank Movies",  "#ff9800"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color};">{val:,}</div>
            <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Coverage Bar Chart ──
    fig, ax = plt.subplots(figsize=(8, 3))
    models = ['BERT', 'FP-Growth', 'PageRank']
    counts = [len(bert_movies_set), len(fp_movies_set), len(pr_movies_set)]
    colors = ['#03dac6', '#00f260', '#bb86fc']
    bars = ax.barh(models, counts, color=colors, edgecolor='none', height=0.5)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f'{val:,}', va='center', color=TEXT_CLR, fontsize=11, fontweight='bold')
    ax.set_title("Movies Available per Model", color=ACCENT3, fontsize=12, fontweight='bold')
    ax.set_xlabel("Number of Movies", color=TEXT_CLR)
    for spine in ax.spines.values(): spine.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # ── Coverage Legend ──
    st.markdown("""
    <div style="display:flex; gap:20px; margin-bottom:20px; flex-wrap:wrap;">
        <div><span style="background:#03dac6;padding:3px 10px;border-radius:4px;font-size:0.8rem;">🧠 BERT</span></div>
        <div><span style="background:#00f260;padding:3px 10px;border-radius:4px;font-size:0.8rem;color:#000;">🌱 FP-Growth</span></div>
        <div><span style="background:#bb86fc;padding:3px 10px;border-radius:4px;font-size:0.8rem;">📡 PageRank</span></div>
        <div><span style="background:#ff9800;padding:3px 10px;border-radius:4px;font-size:0.8rem;color:#000;">✅ All 3</span></div>
        <div><span style="background:#444466;padding:3px 10px;border-radius:4px;font-size:0.8rem;">❌ None</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Search ──
    search = st.text_input("🔍 ابحث عن فيلم", placeholder="اكتب اسم الفيلم...")

    # ── Build coverage table ──
    rows = []
    for movie in sorted(all_movies_set):
        in_bert = movie in bert_movies_set
        in_fp   = movie in fp_movies_set
        in_pr   = movie in pr_movies_set
        count   = sum([in_bert, in_fp, in_pr])
        rows.append({
            'Movie':     movie,
            'BERT 🧠':   '✅' if in_bert else '❌',
            'FP-Growth 🌱': '✅' if in_fp else '❌',
            'PageRank 📡':  '✅' if in_pr else '❌',
            'Coverage':  f'{count}/3',
            '_count':    count,
            '_bert':     in_bert,
            '_fp':       in_fp,
            '_pr':       in_pr,
        })

    coverage_df = pd.DataFrame(rows)

    # ── Filter tabs ──
    tab_all, tab_full, tab_bert, tab_fp, tab_pr, tab_none = st.tabs([
        "📋 الكل",
        "⭐ الـ 3 مع بعض",
        "🧠 BERT فقط",
        "🌱 FP-Growth فقط",
        "📡 PageRank فقط",
        "❌ مفيش coverage",
    ])

    def show_table(df, search_term):
        display = df[['Movie', 'BERT 🧠', 'FP-Growth 🌱', 'PageRank 📡', 'Coverage']].copy()
        if search_term:
            display = display[display['Movie'].str.contains(search_term, case=False, na=False)]
        st.markdown(f"**{len(display):,} فيلم**")
        st.dataframe(display, use_container_width=True, hide_index=True, height=400)
        csv = display.to_csv(index=False).encode()
        st.download_button("⬇️ Download", csv,
                           file_name="coverage.csv", mime="text/csv",
                           key=f"dl_{search_term}_{len(display)}")

    with tab_all:
        show_table(coverage_df.sort_values('_count', ascending=False), search)

    with tab_full:
        show_table(coverage_df[coverage_df['_count'] == 3].sort_values('Movie'), search)

    with tab_bert:
        show_table(coverage_df[coverage_df['_bert']].sort_values('Movie'), search)

    with tab_fp:
        show_table(coverage_df[coverage_df['_fp']].sort_values('Movie'), search)

    with tab_pr:
        show_table(coverage_df[coverage_df['_pr']].sort_values('Movie'), search)

    with tab_none:
        show_table(coverage_df[coverage_df['_count'] == 0].sort_values('Movie'), search)
