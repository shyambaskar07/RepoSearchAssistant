from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.conf import settings
from django.core.cache import cache

from .models import UploadedFile
from .utils import extract_text_from_file
from .rag_engine import embed_text, vectors_from_db, build_faiss_index, search_index

import numpy as np


def upload_file_view(request):
    if request.method == "POST":
        f = request.FILES.get("file")
        if not f:
            return JsonResponse({"error": "no file"}, status=400)

        obj = UploadedFile.objects.create(file=f, name=f.name)
        obj.refresh_from_db()  # ensure file path

        txt = extract_text_from_file(obj.file.path)
        obj.text = txt

        vec = embed_text(txt)
        obj.vector = vec.tobytes()
        obj.save()

        return redirect("upload_success")

    return render(request, "ragapp/upload.html")


def upload_success(request):
    return render(request, "ragapp/upload_success.html")


def search_view(request):
    q = request.GET.get("q", "")
    results = []

    if q.strip():
        files = list(UploadedFile.objects.all())
        vectors, ids = vectors_from_db(files)

        if vectors.shape[0] > 0:
            index = build_faiss_index(vectors)
            qvec = embed_text(q)
            scores, indices = search_index(index, qvec, top_k=5)

            for s, idx in zip(scores, indices):
                if idx < 0 or idx >= len(ids):
                    continue
                fid = ids[idx]
                fobj = next((x for x in files if x.id == fid), None)
                if fobj and s > 0.2:
                    results.append({"file": fobj, "score": float(s)})

    return render(request, "ragapp/results.html", {"query": q, "results": results})


@require_http_methods(["GET", "POST"])
def github_similarity_search(request):
    """
    Semantic GitHub repository search.
    Input: query text
    Output: JSON with repositories: title, github_url, description, similarity, language, stars, reason.
    """
    from github import Github
    from sentence_transformers import SentenceTransformer, util

    query = request.POST.get("query") or request.GET.get("q", "")
    if not query.strip():
        return JsonResponse({"error": "Query required"}, status=400)

    # ---- simple local cache (LocMemCache) ----
    cache_key = "github_search:" + query.strip().lower()
    cached = cache.get(cache_key)
    if cached is not None:
        return JsonResponse(cached)

    # GitHub client (token optional but recommended)
    g = Github(getattr(settings, "GITHUB_TOKEN", "") or "")

    # Search: description + Python language
    search_query = f'{query} language:python in:description'
    repos = g.search_repositories(query=search_query, sort="stars", order="desc")

    repo_texts = []
    repo_info = []

    for repo in list(repos)[:20]:
        desc = (repo.description or "").strip()
        if not desc:
            continue
        repo_texts.append(desc)
        repo_info.append({
            "full_name": repo.full_name,
            "url": repo.html_url,
            "description": desc,
            "stars": repo.stargazers_count,
            "language": repo.language or "Unknown",
        })

    if not repo_texts:
        response_data = {
            "query": query,
            "repositories": [],
            "message": "No relevant repositories found. Try simpler or broader keywords."
        }
        cache.set(cache_key, response_data, timeout=300)
        return JsonResponse(response_data)

    # Embeddings + cosine similarity
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = model.encode([query])
    repo_embs = model.encode(repo_texts)
    similarities = util.cos_sim(query_emb, repo_embs)[0]

    top_k = min(10, len(similarities))
    top_indices = similarities.argsort(descending=True)[:top_k].tolist()

    repositories = []
    for idx in top_indices:
        score = similarities[idx].item()
        info = repo_info[idx]

        short_desc = info["description"]
        if len(short_desc) > 180:
            short_desc = short_desc[:180].rstrip() + "..."

        reason = (
            f"This repository is semantically similar to your query "
            f"and focuses on {info['language']} code with {info['stars']} stars."
        )

        repositories.append({
            "title": info["full_name"],
            "github_url": info["url"],
            "description": short_desc,
            "similarity": round(float(score), 3),
            "language": info["language"],
            "stars": info["stars"],
            "reason": reason,
        })

    response_data = {
        "query": query,
        "repositories": repositories,
    }
    cache.set(cache_key, response_data, timeout=300)
    return JsonResponse(response_data)
