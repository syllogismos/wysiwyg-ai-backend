
from core.mongo_queries import getExperimentById
from core.experiment import supervised_exp_single_variant
import argparse, logging, structlog
# import pdb


def start_exp_variant(exp_id, variant, log):
    exp = getExperimentById(exp_id)
    supervised_exp_single_variant(exp, int(variant), log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', help='Experiment id')
    parser.add_argument('--variant', help='Variant of the experiment to run')
    args = parser.parse_args()
    logger = structlog.get_logger('train_logs')
    log = logger.new(exp=args.exp_id, variant=args.variant)
    log.info('test_from_main', data={'message': 'from sup launcher', 'level': 'info'})
    start_exp_variant(args.exp_id, args.variant, log)

if __name__ == '__main___':
    logger = structlog.get_logger('train_logs')
    log = logger.new(exp='5a271c3bd53e4dc3af967cfc', variant=0)
    log.info("test_from_main_dot", data={'message': "from sup launcher", 'level': 'info'})
    # pdb.set_trace()
    start_exp_variant('5a271c3bd53e4dc3af967cfc', 0, log)
