from celery.decorators import task
from celery.utils.log import get_task_logger
import structlog, os
from core.mongo_queries import getExperimentById, getUserById
from core.mnist import experiment
from core.experiment import launch_exp

logger = structlog.get_logger('train_logs')

@task(name='simple add test', queue='add_test')
def add_test(a, b):
    log = logger.new(request_id='aniasdfkjasfdk')
    # log.info("adding a and b")
    return adding(a, b)


def adding(a, b):
    logger.info('adding a and b')
    return a + b


@task(name='launch_sup_exp_task', queue='launch_sup_exp')
def launch_sup_exp_task(exp_id):
    exp = getExperimentById(exp_id)
    # print(exp)
    # print(os.environ['DJANGO_RUNSERVER'])
    # log = logger.new(user=exp['user'], exp=str(exp['_id']))
    launch_exp(exp)


@task(name='launch_rl_exp_task', queue='launch_rl_exp')
def launch_rl_exp_task(exp_id):
    exp = getExperimentById(exp_id)
    print(exp)
    launch_exp(exp)
    