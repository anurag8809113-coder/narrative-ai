import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CACHE_DIR = ".cache"
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
os.makedirs(CACHE_DIR, exist_ok=True)

_model = None
def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMB_MODEL_NAME)
    return _model

def _cache_path(key: str):
    return os.path.join(CACHE_DIR, f"{key}.pkl")

def _load_cache(key: str):
    p = _cache_path(key)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

def _save_cache(key: str, obj):
    with open(_cache_path(key), "wb") as f:
        pickle.dump(obj, f)

def _get_embeddings(chunks, cache_key="chunks_emb"):
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached
    model = _get_model()
    vecs = model.encode(chunks, show_progress_bar=False)
    _save_cache(cache_key, vecs)
    return vecs

def _get_tfidf(chunks, cache_key="chunks_tfidf"):
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(chunks)
    _save_cache(cache_key, (vectorizer, X))
    return vectorizer, X

def retrieve(chunks, query, k=5, alpha=0.65):
    emb_chunks = _get_embeddings(chunks, cache_key="chunks_emb")
    model = _get_model()
    q_emb = model.encode([query])
    sim_emb = cosine_similarity(q_emb, emb_chunks)[0]

    vectorizer, X = _get_tfidf(chunks, cache_key="chunks_tfidf")
    q_tfidf = vectorizer.transform([query])
    sim_tfidf = cosine_similarity(q_tfidf, X)[0]

    def _norm(x):
        if x.max() - x.min() < 1e-9:
            return x
        return (x - x.min()) / (x.max() - x.min())

    sim = alpha * _norm(sim_emb) + (1 - alpha) * _norm(sim_tfidf)
    top_idx = np.argsort(sim)[::-1][:k]
    return [chunks[i] for i in top_idx]

