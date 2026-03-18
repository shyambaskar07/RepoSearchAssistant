from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from .models import UploadedFile
from .utils import extract_text_from_file
from . import repo_search_assistant_engine as engine

def upload_file_view(request):
    if request.method == "POST":
        f = request.FILES.get("file")
        if not f: return JsonResponse({"error": "no file"}, status=400)
        obj = UploadedFile.objects.create(file=f, name=f.name)
        obj.text = extract_text_from_file(obj.file.path)
        obj.vector = engine.embed_text(obj.text).tobytes()
        obj.save()
        return redirect("upload_success")
    return render(request, "ragapp/upload.html")

def github_similarity_search(request):
    query = request.POST.get("query") or request.GET.get("q", "")
    if not query.strip(): return JsonResponse({"error": "Query required"}, status=400)
    try:
        from github import Github
        from sentence_transformers import util
        g = Github(getattr(settings, "GITHUB_TOKEN", ""))
        repos = g.search_repositories(query=f'{query} in:description,name', sort="stars", order="desc")
        repo_texts, repo_info = [], []
        for repo in list(repos)[:10]:
            desc = (repo.description or "").strip()
            if desc:
                repo_texts.append(desc)
                repo_info.append({"full_name": repo.full_name, "url": repo.html_url})
        if not repo_texts: return JsonResponse({"repositories": []})
        query_emb = engine.embed_text(query)
        repo_embs = engine.EMBED_MODEL.encode(repo_texts)
        sims = util.cos_sim(query_emb, repo_embs)[0]
        repositories = []
        for i in sims.argsort(descending=True)[:5]:
            idx = i.item()
            repositories.append({"title": repo_info[idx]["full_name"], "url": repo_info[idx]["url"], "similarity": round(float(sims[idx]), 3)})
        return JsonResponse({"query": query, "repositories": repositories})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def search_by_file(request):
    if request.method != 'POST': return render(request, 'ragapp/search_upload.html')
    if 'search_file' in request.FILES:
        try:
            content = request.FILES['search_file'].read().decode('utf-8', errors='ignore')
            results = engine.search_similar_file(content)
            return render(request, 'ragapp/results.html', {'query': request.FILES['search_file'].name, 'results': results})
        except Exception as e: return render(request, 'ragapp/search_upload.html', {'error': str(e)})
    return render(request, 'ragapp/search_upload.html')

def upload_success(request): return render(request, "ragapp/upload_success.html")