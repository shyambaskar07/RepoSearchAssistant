import numpy as np
import faiss
from collections import Counter

EMBED_MODEL = None
EMBED_DIM = None

def _init_model():
    global EMBED_MODEL, EMBED_DIM
    if EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        if EMBED_MODEL.device.type == 'cuda':
            EMBED_MODEL.half()
        EMBED_DIM = EMBED_MODEL.get_sentence_embedding_dimension()

def embed_text(text):
    _init_model()
    vec = EMBED_MODEL.encode([text], convert_to_numpy=True)[0].astype("float32")
    norm = np.linalg.norm(vec)
    if norm > 0: vec = vec / norm
    return vec

def vectors_from_db(uploaded_files):
    _init_model()
    vecs, ids = [], []
    for f in uploaded_files:
        if f.vector:
            vecs.append(np.frombuffer(f.vector, dtype="float32"))
            ids.append(f.id)
    if vecs: return np.vstack(vecs).astype("float32"), ids
    return np.empty((0, EMBED_DIM), dtype="float32"), []

def build_faiss_index(vectors=None):
    _init_model()
    if vectors is None:
        from .models import UploadedFile
        all_files = list(UploadedFile.objects.all())
        vectors, ids = vectors_from_db(all_files)
        if vectors.shape[0] == 0: return None, []
    else:
        ids = []
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    return index if not ids else (index, ids)

def search_similar_file(file_content, top_k=3):
    _init_model()
    chunks = [file_content[i:i+500] for i in range(0, len(file_content), 500)]
    if not chunks: return []
    query_vectors = np.array([embed_text(c) for c in chunks], dtype="float32")
    from .models import UploadedFile
    files = list(UploadedFile.objects.all())
    vectors, ids = vectors_from_db(files)
    if vectors.shape[0] == 0: return []
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    D, I = index.search(query_vectors, k=1)
    votes = [next((x.name for x in files if x.id == ids[idx_arr[0]]), None) for idx_arr in I if 0 <= idx_arr[0] < len(ids)]
    vote_counts = Counter(filter(None, votes))
    return [{'source': n, 'confidence': f"{round((c/len(chunks))*100,1)}%", 'matches': c} for n, c in vote_counts.most_common(top_k)]