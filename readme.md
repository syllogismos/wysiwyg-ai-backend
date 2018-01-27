
## Development

### Start server for development
```
python manage.py runserver
```
### Start a celery worker
```
celery -A eschernode worker -l warning -Q mnist_test
```

## Notes
### ulimit settings
http://posidev.com/blog/2009/06/04/set-ulimit-parameters-on-ubuntu/

### systemctl cheatsheet
https://fedoraproject.org/wiki/SysVinit_to_Systemd_Cheatsheet

### install nginx, nvm, pm2, gunicorn

gunicorn, pm2, nginx, django server
http://technopy.com/deploying-a-flask-application-in-production-nginx-gunicorn-and-pm2-html-2/


supervisord for celery worker
pm2 for the gunicorn server


# Start server on Production

pm2 start server.sh
pm2 save

git pull
pm2 restart server


# Elasticsearch
## Start from AMI
Stop the elasticsearch instance started with start of the machine

`sudo systemctl stop elasticsearch`

edit `/etc/elasticsearch/elasticsearch.yml` to modify cluster name, node name and host in the end

restart elasticsearch instance with

`sudo systemctl start elasticsearch`

## Cerebro
Start cerebro using the conf file in `elasticsearch/cerebro.conf` from this repository, put it in `conf/application.conf`


you start it using the command 

`bin/cerebro  -Dconfig.file=/some/other/dir/alternate.conf`

or using the shell script `cerebro.sh` using `sh cerebro.sh`

## Elasticsearch stuff

https://www.elastic.co/guide/en/elasticsearch/reference/current/deb.html
https://www.elastic.co/guide/en/elasticsearch/reference/master/setting-system-settings.html#jvm-options
https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-get-on-ubuntu-16-04


# AMI's
GPU instances, installing pytorch in gpu machines


## Install nvidia and cuda drivers
https://github.com/kevinzakka/blog-code/blob/master/aws-pytorch/install.sh
```
# drivers
wget http://us.download.nvidia.com/tesla/375.51/nvidia-driver-local-repo-ubuntu1604_375.51-1_amd64.deb
sudo dpkg -i nvidia-driver-local-repo-ubuntu1604_375.51-1_amd64.deb
sudo apt-get update
sudo apt-get -y install cuda-drivers
sudo apt-get update && sudo apt-get -y upgrade
```

```
nvidia-smi # to check the status of the gpus
```

## Checking if cuda is working with pytorch
```
import torch
torch.randn(2,3).cuda()
torch.cuda.is_available()
torch.cuda.device_count()
```

Userful links:
1. https://discuss.pytorch.org/t/request-tutorial-for-deploying-on-cloud-based-virtual-machine/28/3
2. https://github.com/kevinzakka/blog-code/blob/master/aws-pytorch/install.sh
3. https://blog.waya.ai/quick-start-pyt-rch-on-an-aws-ec2-gpu-enabled-compute-instance-5eed12fbd168
4. http://pytorch.org/


## S3 karaka_test Bucket ACL
To allow all the mp4 files to be read from dashboard url
```
{
    "Version": "2012-10-17",
    "Id": "Policy1514905660784",
    "Statement": [
        {
            "Sid": "Stmt1514905654924",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::karaka_test/*.mp4"
        }
    ]
}
```


