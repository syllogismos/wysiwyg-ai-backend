from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from supervised.tasks import add_test, mnist_task, launch_exp_task
from eschernode.settings import mongoClient
from django.views.decorators.csrf import csrf_exempt
import json
import time

# Create your views here.
def index(request):
    # add_test.apply_async([23, 43])
    mnist_task.apply_async()
    return JsonResponse({"Status": 200, "message": "Training Starting"})

@csrf_exempt
def launchExperiment(request):
    """
    launch experiment of a given experiment id
    """
    if request.method == 'POST':
        # print(json.loads(request.body.decode('utf-8')), "POST")
        json_body = json.loads(request.body.decode('utf-8'))
        # time.sleep(2)
        # launch_exp_task.apply_async([json_body['exp_id']])
        return JsonResponse({"status": 200, "message": "Launching Experiment"})
