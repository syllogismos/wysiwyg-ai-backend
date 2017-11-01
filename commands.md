
## Start server for development
```
python manage.py runserver
```



## Start a celery worker
```
celery -A eschernode worker -l warning -Q mnist_test
```