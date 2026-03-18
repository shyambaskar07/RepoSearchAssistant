from django.urls import path
from . import views

urlpatterns = [
    path("upload/", views.upload_file_view, name="upload_file"),
    path("upload-success/", views.upload_success, name="upload_success"),
    path("search-github/", views.github_similarity_search, name="github_search"),
    path("search-file/", views.search_by_file, name="search_by_file"),
]