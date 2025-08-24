# 🎵 Vibe-based Song Recommender

A comprehensive explanation of the code and components in `untitled3.py` for the Song Recommender deployed [here](https://huggingface.co/spaces/nalinkt23/SongRecommender).

---

## Table of Contents

1. [Dataset Loading](#dataset-loading)
2. [Data Preprocessing](#data-preprocessing)
3. [Text Embedding](#text-embedding)
4. [Image Embedding (Optional)](#image-embedding-optional)
5. [Training Ridge Regression](#training-ridge-regression)
6. [Recommendation Functions](#recommendation-functions)
    - [Cosine Similarity Recommendation](#cosine-similarity-recommendation)
    - [MMR Diversity Recommendation](#mmr-diversity-recommendation)
    - [Explaining Recommendations](#explaining-recommendations)
7. [2D Visualization of Song Vibes](#2d-visualization-of-song-vibes)
8. [Gradio UI Deployment](#gradio-ui-deployment)
9. [Exporting Models and Data](#exporting-models-and-data)

---

## 1. Dataset Loading

```python
import kagglehub
path = kagglehub.dataset_download("yamaerenay/spotify-dataset-1921-2020-160k-tracks")
```
**Why?**  
Downloads a large Spotify dataset from Kaggle using `kagglehub`. This gives us access to 160k tracks with features.

```python
import pandas as pd
import numpy as np
df = pd.read_csv("/kaggle/input/spotify-dataset-1921-2020-160k-tracks/data.csv")
```
**Why?**  
Loads the dataset into a Pandas DataFrame for easy manipulation and analysis.

---

## 2. Data Preprocessing

```python
feature_cols = [...]
df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
df = df.dropna(subset=feature_cols).reset_index(drop=True)
```
**Why?**  
Removes duplicate songs and rows missing essential audio features to ensure clean, usable data.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_songs = scaler.fit_transform(df[feature_cols])
```
**Why?**  
Standardizes audio features for better model training and similarity calculations.

---

## 3. Text Embedding

```python
!pip install -q sentence-transformers
from sentence_transformers import SentenceTransformer
text_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
```
**Why?**  
Loads a powerful HuggingFace transformer model to convert text queries and song metadata into dense vector embeddings.

```python
def get_text_embedding(text: str):
    emb = text_model.encode(text, normalize_embeddings=True)
    return emb
```
**Why?**  
Defines a reusable function to convert any text into an embedding.

---

## 4. Image Embedding (Optional)

```python
!pip install -q transformers ftfy regex tqdm
!pip install -q git+https://github.com/openai/CLIP.git
import clip
clip_model, preprocess = clip.load("ViT-B/32", device=device)
```
**Why?**  
(Optional) Loads OpenAI CLIP for potential use in embedding images, e.g., album art, to allow multimodal recommendations.

---

## 5. Training Ridge Regression

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    text_embeddings, X_songs, test_size=0.2, random_state=42
)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, Y_train)
```
**Why?**  
Trains a linear regression model (`Ridge`) to map from text embedding space (song metadata/query) to audio feature space. This enables us to predict the "vibe" (audio features) from a text query.

---

## 6. Recommendation Functions

### Cosine Similarity Recommendation

```python
def recommend_from_text(user_text, top_k=10):
    e_text = get_text_embedding(user_text).reshape(1, -1)
    vibe_vector = ridge.predict(e_text)
    sims = cosine_similarity(vibe_vector, X_songs)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = songs_df.iloc[top_idx].copy()
    results["similarity"] = sims[top_idx]
    return results.reset_index(drop=True)
```
**Why?**  
Given a user's text description, this function predicts the corresponding vibe vector and finds the most similar songs using cosine similarity.

### MMR Diversity Recommendation

```python
def mmr(query_vec, doc_vecs, k=10, lambda_param=0.7):
    # Implements Maximal Marginal Relevance to select diverse recommendations
```
**Why?**  
Provides more diverse song recommendations by balancing relevance and novelty.

### Explaining Recommendations

```python
def explain_recommendation(query_vec, song_vec, top_n=2):
    # Compares query and song vectors to explain why a song was recommended
```
**Why?**  
Generates human-readable explanations for recommendations, increasing user trust and satisfaction.

---

## 7. 2D Visualization of Song Vibes

```python
import umap
umap_model = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
X_2d = umap_model.fit_transform(X_songs)
songs_df["x"] = X_2d[:,0]
songs_df["y"] = X_2d[:,1]
```
**Why?**  
Reduces high-dimensional audio features to 2D for visualization, making it possible to plot and visually explore song "vibes".

```python
plt.scatter(...)
```
**Why?**  
Plots the UMAP-reduced song features, color-coded by vibe label, aiding in understanding the distribution of moods.

---

## 8. Gradio UI Deployment

```python
import gradio as gr

def recommend_ui(query, top_k=5):
    # Uses recommend_with_explanations to return recommendations in readable format

demo = gr.Interface(
    fn=recommend_ui,
    inputs=[
        gr.Textbox(label="Enter a vibe (text query)", placeholder="e.g. chill rainy evening"),
        gr.Slider(1, 15, value=5, step=1, label="Number of Recommendations")
    ],
    outputs=gr.Textbox(label="Recommendations"),
    title="🎵 Vibe-based Song Recommender",
    description="Type a vibe ... and get music recommendations with explanations."
)
demo.launch(share=True)
```
**Why?**  
Provides a web-based interface for users to enter queries and get recommendations, powered by Gradio and deployed on Hugging Face Spaces.

---

## 9. Exporting Models and Data

```python
import os, joblib, numpy as np
export_dir = "/content/exports"
os.makedirs(export_dir, exist_ok=True)

joblib.dump(ridge, f"{export_dir}/reg_text.pkl")
np.save(f"{export_dir}/X_audio.npy", X_songs)
songs_df[subset_cols + extra_cols].to_parquet(f"{export_dir}/songs_meta.parquet")
if "x" in songs_df.columns and "y" in songs_df.columns:
    X_2d = songs_df[["x","y"]].to_numpy()
    np.save(f"{export_dir}/X_2d.npy", X_2d)
```
**Why?**  
Saves trained models, processed features, and song metadata for future use or deployment. This ensures reproducibility and portability.

---

## Summary

This project uses state-of-the-art NLP and machine learning methods to map user vibes (text descriptions) to song recommendations, visualizes song moods, and explains recommendations in plain language.  
It is deployed with Gradio for easy access at [this Hugging Face Space](https://huggingface.co/spaces/nalinkt23/SongRecommender).

Feel free to explore, modify, and extend this code for your own vibe-based music applications!
