# ragapp/rag_engine.py
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# load model once
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = EMBED_MODEL.get_sentence_embedding_dimension()  # usually 384

def embed_text(text):
    # returns a normalized float32 vector (for cosine via inner product)
    vec = EMBED_MODEL.encode([text], convert_to_numpy=True)[0].astype("float32")
    # normalize to unit length for cosine similarity via inner product
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def vectors_from_db(uploaded_files):
    # uploaded_files: queryset/list of UploadedFile objects
    vecs = []
    ids = []
    for i, f in enumerate(uploaded_files):
        if f.vector:
            arr = np.frombuffer(f.vector, dtype="float32")
            vecs.append(arr)
            ids.append(f.id)
    if vecs:
        return np.vstack(vecs).astype("float32"), ids
    else:
        return np.empty((0, EMBED_DIM), dtype="float32"), []
    
def build_faiss_index(vectors):
    # using IndexFlatIP on normalized vectors (inner product ~ cosine)
    if vectors.shape[0] == 0:
        return None
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    return index

def search_index(index, query_vec, top_k=5):
    if index is None:
        return [], []
    # query_vec should be normalized float32
    D, I = index.search(np.array([query_vec], dtype="float32"), top_k)
    return D[0].tolist(), I[0].tolist()
