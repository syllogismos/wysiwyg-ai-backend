#!/bin/bash
{
# make sure aws is installed and configured in the root


# anaconda right environment
# make sure anaconda is accessible from the root
# modify /root/.bashrc to include anaconda in $PATH
# also source the conda env in this file
# modifying bash script is not working so i change the PATH and PYTHONPATH in this script itself


# Get experimentid and variant index from tags
INSTANCE_ID="`wget -qO- http://instance-data/latest/meta-data/instance-id`"
REGION="`wget -qO- http://instance-data/latest/meta-data/placement/availability-zone | sed -e 's:\([0-9][0-9]*\)[a-z]*\$:\\1:'`"
EXPERIMENT="`aws ec2 describe-tags --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=ExperimentId" --region $REGION --output=text | cut -f5`"
VARIANT="`aws ec2 describe-tags --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=VariantIndex" --region $REGION --output=text | cut -f5`"

export AWS_DEFAULT_REGION='us-east-1'

# wait until filebeat is started
until pids=$(pidof filebeat)
do   
    sleep 5
done

# source activate rllabpp
export PATH=/home/ubuntu/anaconda2/envs/rllabpp/bin:$PATH
export PYTHONPATH=/home/ubuntu/anaconda2/envs/rllabpp/bin:/home/ubuntu/rllabpp:/home/ubuntu:/home/ubuntu/anaconda2/envs/rllabpp/lib/python36.zip:/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6:/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/lib-dynload:/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/site-packages:/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/site-packages/Mako-1.0.7-py3.6.egg:/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/site-packages/cycler-0.10.0-py3.6.egg:/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/site-packages/IPython/extensions:$PYTHONPATH

cd /home/ubuntu/rllabpp

# start s3 sync periodically
aws s3 sync --quiet /home/ubuntu/rllabpp/data/local/ s3://karaka_test/$EXPERIMENT/$VARIANT/

while /bin/true; do
aws s3 sync  --quiet /home/ubuntu/rllabpp/data/local/ s3://karaka_test/$EXPERIMENT/$VARIANT/
sleep 200
done & echo sync initiated
while /bin/true; do
if [ -z $(curl -Is http://169.254.169.254/latest/meta-data/spot/termination-time | head -1 | grep 404 | cut -d \  -f 2) ]
then
logger "Running shutdown hook."
aws s3 cp --recursive --quiet /home/ubuntu/rllabpp/data/local s3://karaka_test/$EXPERIMENT/$VARIANT/
# aws s3 cp  --quiet /home/ubuntu/rllabpp/data/local/user_data.log s3://karaka_test/$EXPERIMENT/$VARIANT/user_data.log
break
else
# Spot instance not yet marked for termination.
sleep 5
fi
done & echo log sync initiated


# start the experiment
python -u sandbox/rocky/tf/launchers/algo_gym_stub.py --expId $EXPERIMENT

# Copy the checkpoint logs and user data logs
aws s3 cp --recursive --quiet /home/ubuntu/rllabpp/data/local s3://karaka_test/$EXPERIMENT/$VARIANT/
aws s3 cp --quiet /home/ubuntu/rllabpp/data/local/user_data.log s3://karaka_test/$EXPERIMENT/$VARIANT/user_data.log
} >> /home/ubuntu/rllabpp/data/local/user_data.log 2>&1