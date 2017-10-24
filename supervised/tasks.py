from celery.decorators import task
from celery.utils.log import get_task_logger
import structlog
from core.mongo_queries import getExperimentById, getUserById, getNetworkById
from core.mnist import experiment

logger = structlog.get_logger('train_logs')

@task(name="simple add test", queue='add_test')
def add_test(a, b):
    log = logger.new(request_id="aniasdfkjasfdk")
    # log.info("adding a and b")
    return adding(a, b)


def adding(a, b):
    logger.info('adding a and b')
    return a + b


@task(name="train_nn", queue="escher_train_nn")
def train_neural_netowrk(userId, expId):
    experiment = getExperimentById(expId)
    user = getUserById(userId)
    networkId = experiment['network_json']
    network = getNetworkById(networkId)

@task(name='mnist_test', queue="mnist_test")
def mnist_task():
    log = logger.new(user="anil")
    experiment(log)
