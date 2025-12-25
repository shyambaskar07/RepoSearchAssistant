import numpy as np
import faiss

# Lazy-loaded globals - initialized on first use (once per worker)
EMBED_MODEL = None
EMBED_DIM = None

def _init_model():
    """Initialize model on first use - called once per worker"""
    global EMBED_MODEL, EMBED_DIM
    if EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        EMBED_DIM = EMBED_MODEL.get_sentence_embedding_dimension()

def embed_text(text):
    """Returns normalized float32 vector for cosine similarity"""
    _init_model()
    vec = EMBED_MODEL.encode([text], convert_to_numpy=True)[0].astype("float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def vectors_from_db(uploaded_files):
    """Extract vectors from UploadedFile queryset/list"""
    _init_model()
    vecs = []
    ids = []
    for f in uploaded_files:
        if f.vector:
            arr = np.frombuffer(f.vector, dtype="float32")
            vecs.append(arr)
            ids.append(f.id)
    if vecs:
        return np.vstack(vecs).astype("float32"), ids
    return np.empty((0, EMBED_DIM), dtype="float32"), []

def build_faiss_index(vectors):
    """Build FAISS index from normalized vectors"""
    _init_model()
    if vectors.shape[0] == 0:
        return None
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    return index

def search_index(index, query_vec, top_k=5):
    """Search FAISS index, returns (distances, indices)"""
    if index is None:
        return [], []
    D, I = index.search(np.array([query_vec], dtype="float32"), top_k)
    return D[0].tolist(), I[0].tolist()
