
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
### Elasticsearch stuff

https://www.elastic.co/guide/en/elasticsearch/reference/current/deb.html
https://www.elastic.co/guide/en/elasticsearch/reference/master/setting-system-settings.html#jvm-options
https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-get-on-ubuntu-16-04


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

