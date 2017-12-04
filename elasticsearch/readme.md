# Start from AMI
Stop the elasticsearch instance started with start of the machine

`sudo systemctl stop elasticsearch`

edit `/etc/elasticsearch/elasticsearch.yml` to modify cluster name, node name and host in the end

restart elasticsearch instance with

`sudo systemctl start elasticsearch`

# Cerebro
Start cerebro using the conf file in `elasticsearch/cerebro.conf` from this repository, put it in `conf/application.conf`


you start it using the command 

`bin/cerebro  -Dconfig.file=/some/other/dir/alternate.conf`

or using the shell script `cerebro.sh` using `sh cerebro.sh`