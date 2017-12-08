
from core.mongo_queries import getExperimentById
from core.experiment import supervised_exp_single_variant
import argparse, logging, structlog
# import pdb


def start_exp_variant(exp_id, variant):
    print(variant)
    print(exp_id)
    exp = getExperimentById(exp_id)
    logger = structlog.get_logger('train_logs')
    log = logger.new(user=exp['user'], exp=exp_id, variant=variant)
    log.info('exp_timeline', timeline={
        'message': 'Training started on variant %s' %variant,
        'variant': variant,
        'level': 'info'
    })
    supervised_exp_single_variant(exp, int(variant), log)
    log.info('exp_timeline', timeline={
        "message": "Training done in Variant %s, and machine terminating" %variant,
        "level": "info",
        "variant": variant
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--expId', help='Experiment id')
    parser.add_argument('--variantId', help='Variant of the experiment to run')
    args = parser.parse_args()
    # logger = structlog.get_logger('train_logs')
    # log = logger.new(exp=args.expId, variant=args.variantId)
    # log.info('test_from_main', data={'message': 'from sup launcher', 'level': 'info'})
    start_exp_variant(args.expId, args.variantId)

if __name__ == '__main___':
    start_exp_variant('5a29873bd53e4dc3af967d00', 0)
