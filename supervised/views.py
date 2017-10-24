from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from supervised.tasks import add_test, mnist_task

# Create your views here.
def index(request):
    # add_test.apply_async([23, 43])
    mnist_task.apply_async()
    return JsonResponse({"Status": 200, "message": "Training Starting"})