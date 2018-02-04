from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from supervised.tasks import add_test, launch_sup_exp_task, launch_rl_exp_task
from eschernode.settings import mongoClient
from django.views.decorators.csrf import csrf_exempt
from core.config import USER_DATA
import json
import time

# Create your views here.
def index(request):
    add_test(23, 43)
    # mnist_task.apply_async()
    a = USER_DATA
    return JsonResponse({"Status": 200, "message": "Training Starting"})

@csrf_exempt
def launchSupervised(request):
    """
    launch experiment of a given experiment id
    """
    if request.method == 'POST':
        json_body = json.loads(request.body.decode('utf-8'))
        print(json_body['exp_id'])
        launch_sup_exp_task.apply_async([json_body['exp_id']])
        # launch_sup_exp_task(json_body['exp_id'])
        return JsonResponse({"status": 200, "message": "Launching Supervised Experiment"})

@csrf_exempt
def launchRL(request):
    """
    launch rl experiment of a given experiment id
    """
    if request.method == 'POST':
        json_body = json.loads(request.body.decode('utf-8'))
        print(json_body['exp_id'])
        launch_rl_exp_task.apply_async([json_body['exp_id']])
        # launch_rl_exp_task(json_body['exp_id'])
        return JsonResponse({"status": 200, "message": "Launching RL experiment"})
