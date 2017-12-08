#!/bin/bash
{
# Get experimentid and variant index from tags
INSTANCE_ID="`wget -qO- http://instance-data/latest/meta-data/instance-id`"
REGION="`wget -qO- http://instance-data/latest/meta-data/placement/availability-zone | sed -e 's:\([0-9][0-9]*\)[a-z]*\$:\\1:'`"

# EXPERIMENT="notyetset"
# VARIANT="notyetset"
# https://serverfault.com/questions/7503/how-to-determine-if-a-bash-variable-is-empty
until [ -n "$EXPERIMENT" ] # start querying for ExperimentId tag and be in loop till it is set
do
    EXPERIMENT="`aws ec2 describe-tags --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=ExperimentId" --region $REGION --output=text | cut -f5`"
    echo "querying ExperimentId tag"
    echo $EXPERIMENT
    sleep 5
done

until [ -n "$VARIANT" ]
do
    VARIANT="`aws ec2 describe-tags --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=VariantIndex" --region $REGION --output=text | cut -f5`"
    echo "querying VariantIndex tag"
    echo $VARIANT
    sleep 5
done

export AWS_DEFAULT_REGION='us-east-1'

# wait until filebeat is started
until pids=$(pidof filebeat)
do   
    echo "Waiting for filebeat to start"
    sleep 5
done

export PATH=/home/ubuntu/anaconda3/bin:$PATH
export PYTHONPATH=/home/ubuntu/anaconda3/bin:/home/ubuntu/anaconda3/lib/python36.zip:/home/ubuntu/anaconda3/lib/python3.6:/home/ubuntu/anaconda3/lib/python3.6/lib-dynload:/home/ubuntu/anaconda3/lib/python3.6/site-packages:/home/ubuntu/anaconda3/lib/python3.6/site-packages/torchvision-0.1.9-py3.6.egg:/home/ubuntu/anaconda3/lib/python3.6/site-packages/IPython/extensions:/home/ubuntu/dashboard_backend:$PYTHONPATH
export ESCHERNODE_ENV=prod

cd /home/ubuntu/dashboard_backend

git fetch --all
git reset --hard origin/master

aws s3 sync --quiet /home/ubuntu/dashboard_backend/results/ s3://karaka_test/$EXPERIMENT/$VARIANT/

while /bin/true; do
    aws s3 sync  --quiet /home/ubuntu/dashboard_backend/results/ s3://karaka_test/$EXPERIMENT/$VARIANT/
    sleep 300
done & echo "sync initiated"
while /bin/true; do
    if [ -z $(curl -Is http://169.254.169.254/latest/meta-data/spot/termination-time | head -1 | grep 404 | cut -d \  -f 2) ]
    then
        logger "Running shutdown hook."
        # aws s3 cp --recursive --quiet /home/ubuntu/rllabpp/data/local s3://karaka_test/$EXPERIMENT/$VARIANT/
        aws s3 sync --quiet /home/ubuntu/dashboard_backend/results/ s3://karaka_test/$EXPERIMENT/$VARIANT/
        break
    else
        # Spot instance not yet marked for termination.
        sleep 5
    fi
done & echo "log sync initiated"

# start the experiment
python -u core/sup_exp_launcher.py --expId $EXPERIMENT --variantId $VARIANT

# final sync before terminating
aws s3 sync --quiet /home/ubuntu/dashboard_backend/results/ s3://karaka_test/$EXPERIMENT/$VARIANT/

sleep 60
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
} >> /home/ubuntu/dashboard_backend/results/user_data.log 2>&1