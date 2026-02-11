import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="News Topic Discovery Dashboard",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🟣 News Topic Discovery Dashboard")
st.write(
    "This system uses **Hierarchical Clustering** to automatically group "
    "similar news articles based on textual similarity."
)

# ---------------- LOAD DATASET ----------------
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "dataset.csv")
    return pd.read_csv(file_path)

df = load_data()
st.success("Dataset loaded from repository")

# ---------------- TEXT COLUMN ----------------
st.sidebar.header("📝 Text Column Selection")
text_column = st.sidebar.selectbox(
    "Select the column containing news text",
    df.columns
)

texts = df[text_column].astype(str)

# ---------------- TF-IDF CONTROLS ----------------
st.sidebar.header("📝 Text Vectorization")

max_features = st.sidebar.slider(
    "Maximum TF-IDF Features",
    100, 2000, 1000
)

use_stopwords = st.sidebar.checkbox(
    "Remove English Stopwords",
    value=True
)

ngram_choice = st.sidebar.selectbox(
    "N-gram Range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

if ngram_choice == "Unigrams":
    ngram_range = (1, 1)
elif ngram_choice == "Bigrams":
    ngram_range = (2, 2)
else:
    ngram_range = (1, 2)

vectorizer = TfidfVectorizer(
    max_features=max_features,
    stop_words="english" if use_stopwords else None,
    ngram_range=ngram_range
)

X_tfidf = vectorizer.fit_transform(texts)

# ---------------- HIERARCHICAL CONTROLS ----------------
st.sidebar.header("🌳 Hierarchical Clustering")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

subset_size = st.sidebar.slider(
    "Articles for Dendrogram",
    20, 200, 50
)

# ---------------- DENDROGRAM ----------------
if st.sidebar.button("🟦 Generate Dendrogram"):
    st.subheader("🌳 Hierarchical Dendrogram (Subset)")

    X_subset = X_tfidf[:subset_size].toarray()
    Z = linkage(X_subset, method=linkage_method)

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, ax=ax)
    ax.set_ylabel("Distance")
    ax.set_xlabel("Article Index")

    st.pyplot(fig)

# ---------------- APPLY CLUSTERING ----------------
st.sidebar.header("🟩 Apply Clustering")

num_clusters = st.sidebar.slider(
    "Number of Clusters",
    2, 10, 3
)

model = AgglomerativeClustering(
    n_clusters=num_clusters,
    linkage=linkage_method
)

clusters = model.fit_predict(X_tfidf.toarray())
df["Cluster"] = clusters

# ---------------- PCA VISUALIZATION ----------------
st.subheader("📉 Cluster Visualization (PCA Projection)")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_tfidf.toarray())

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="tab10",
    alpha=0.7
)
ax2.set_xlabel("PCA Component 1")
ax2.set_ylabel("PCA Component 2")

st.pyplot(fig2)

# ---------------- CLUSTER SUMMARY ----------------
st.subheader("📊 Cluster Summary")

feature_names = np.array(vectorizer.get_feature_names_out())
summary = []

for c in range(num_clusters):
    idx = np.where(clusters == c)[0]
    cluster_tfidf = X_tfidf[idx]

    mean_tfidf = cluster_tfidf.mean(axis=0).A1
    top_terms = feature_names[np.argsort(mean_tfidf)[-10:]][::-1]

    summary.append({
        "Cluster ID": c,
        "Number of Articles": len(idx),
        "Top Keywords": ", ".join(top_terms)
    })

summary_df = pd.DataFrame(summary)
st.dataframe(summary_df)

# ---------------- SILHOUETTE SCORE ----------------
st.subheader("📊 Clustering Validation")

sil_score = silhouette_score(X_tfidf, clusters)
st.metric("Silhouette Score", round(sil_score, 3))

st.caption(
    "Close to 1 → well-separated clusters | "
    "Close to 0 → overlapping clusters | "
    "Negative → poor clustering"
)

# ---------------- BUSINESS INTERPRETATION ----------------
st.subheader("🧠 Business Interpretation")

for _, row in summary_df.iterrows():
    st.write(
        f"🟣 Cluster {row['Cluster ID']}: "
        f"Articles mainly relate to **{row['Top Keywords'].split(',')[0]}**."
    )

# ---------------- USER GUIDANCE ----------------
st.info(
    "Articles grouped into the same cluster share similar vocabulary and themes. "
    "These clusters can support automatic tagging, content recommendation, "
    "and editorial organization."
)
