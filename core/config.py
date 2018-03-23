import os


RLLAB_AMI = "ami-f4d50589"
SUPERVISED_AMI = "ami-2c0d7656"

if 'ESCHERNODE_ENV' not in os.environ:
    os.environ['ESCHERNODE_ENV'] = 'dev'

if os.environ['ESCHERNODE_ENV'] == 'prod':
    HOME_DIR = '/home/ubuntu/dashboard_backend'
elif os.environ['ESCHERNODE_ENV'] == 'dev':
    HOME_DIR = '/Users/anil/Code/escher/eschernode'

if os.environ['ESCHERNODE_ENV'] == 'prod':
    USER_DATA = open('/home/ubuntu/dashboard_backend/core/rl_user_data_script.sh', 'rb').read()
else:
    USER_DATA = open('/Users/anil/Code/escher/eschernode/core/rl_user_data_script.sh', 'rb').read()

RL_USER_DATA = open(os.path.join(HOME_DIR, 'core/rl_user_data_script.sh'), 'rb').read()
SUP_USER_DATA = open(os.path.join(HOME_DIR, 'core/sup_user_data_script.sh'), 'rb').read()
