# shell script to terminate the ec2 instance
# sh terminate.sh
# import subprocess
# subprocess.check_call(['sh', 'path to ./terminate.sh'])


instanceId=$(curl http://169.254.169.254/latest/meta-data/instance-id/)
REGION="`wget -qO- http://instance-data/latest/meta-data/placement/availability-zone | sed -e 's:\([0-9][0-9]*\)[a-z]*\$:\\1:'`"
EXPERIMENT="`aws ec2 describe-tags --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=ExperimentId" --region $REGION --output=text | cut -f5`"
VARIANT="`aws ec2 describe-tags --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=VariantIndex" --region $REGION --output=text | cut -f5`"

if [ -n "$EXPERIMENT" ]
then
    cd /home/ubuntu/rllabpp
    aws s3 sync --quiet /home/ubuntu/rllabpp/data/local/ s3://karaka_test/$EXPERIMENT/$VARIANT/
fi


aws ec2 terminate-instances --instance-ids $instanceId --region $REGION