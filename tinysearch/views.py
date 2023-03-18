# Import necessary modules
from django.shortcuts import render,get_object_or_404
from .models import  Book, Booktype
from django.db.models import Q
from build_index import load_doc_file_list
import json
from search import retrieve
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

def main(request):
    return render(request, 'main.html', {})

file_list = load_doc_file_list()
file_dict = {}
for file_path in file_list:
    with open(file_path, "r") as f:
        data = json.load(f)
        file_dict[data["url"]] = file_path

def detail(request):
    target_url = request.GET.get('url')
    
    for (url, file_path) in file_dict.items():
        if url == target_url:
            target_file_path = file_path
    else:
        for (url, file_path) in file_dict.items():
            if url.startswith(target_url):
                target_file_path = file_path

    with open(target_file_path, "r") as f:
        data = json.load(f)
        content = data["content"]

    return render(request, 'detail.html', {'content': content})

def search(request):
    results = []

    if request.method == "GET":
        query = request.GET.get('search')
        page_num = request.GET.get('page', 1)

        if query == '':
            query = 'None'

        results = retrieve(query, 10)

        paginator = Paginator(results, 10)

        try:
            page_obj = paginator.page(page_num)
        except PageNotAnInteger:
            # if page is not an integer, deliver the first page
            page_obj = paginator.page(1)
        except EmptyPage:
            # if the page is out of range, deliver the last page
            page_obj = paginator.page(paginator.num_pages)



    return render(request, 'search.html', {'query': query, 'results': results, "len": len(results), 'page_obj': page_obj})