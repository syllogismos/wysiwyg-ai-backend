import os


RLLAB_AMI = "ami-c6f294bc"

if os.environ['ESCHERNODE_ENV'] == 'prod':
    USER_DATA = open('/home/ubuntu/dashboard_backend/core/rl_user_data_script.sh', 'rb').read()
else:
    USER_DATA = open('/Users/anil/Code/escher/eschernode/core/rl_user_data_script.sh', 'rb').read()
