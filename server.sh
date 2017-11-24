#!/bin/bash

# wait until filebeat is started
until pids=$(pidof filebeat)
do   
    sleep 5
    echo "waiting for filebeat to start"
done

export ESCHERNODE_ENV=prod
gunicorn -w 3 eschernode.wsgi