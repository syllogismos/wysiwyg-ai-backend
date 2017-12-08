"""
Django settings for eschernode project.

Generated by 'django-admin startproject' using Django 1.11.3.

For more information on this file, see
https://docs.djangoproject.com/en/1.11/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.11/ref/settings/
"""

import os
import logging, structlog
from logging.handlers import WatchedFileHandler
from structlog.threadlocal import wrap_dict
from pymongo import MongoClient
from core.config import HOME_DIR


# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.11/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '+yh97%jk(goz&gcm(op9$h6j7ce$c9wliy#x_t0qva-=v8te@^'

# SECURITY WARNING: don't run with debug turned on in production!



# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'eschernode.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'eschernode.wsgi.application'


# Database
# https://docs.djangoproject.com/en/1.11/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}


# Password validation
# https://docs.djangoproject.com/en/1.11/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/1.11/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.11/howto/static-files/

STATIC_URL = '/static/'

if 'ESCHERNODE_ENV' not in os.environ:
    os.environ['ESCHERNODE_ENV'] = 'dev'

if not os.path.isdir(os.path.join(HOME_DIR, 'results')):
    os.makedirs(os.path.join(HOME_DIR, 'results'))

if os.environ['ESCHERNODE_ENV'] == 'dev':
    DEBUG = True
    MONGO_HOST = '52.2.113.244'
    MONGO_PORT = 27017
    MONGO_DB = 'eschernode'
    FILEBEAT_LOGFILE = os.path.join(HOME_DIR, 'results', 'filebeat.log')
    ALLOWED_HOSTS = ['localhost']

elif os.environ['ESCHERNODE_ENV'] == 'prod':
    DEBUG = False
    MONGO_HOST = '172.30.0.169'
    MONGO_PORT = 27017
    MONGO_DB = 'eschernode'
    FILEBEAT_LOGFILE = os.path.join(HOME_DIR, 'results', 'filebeat.log')
    ALLOWED_HOSTS = ['172.30.0.251']
    

# Structlog config
structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=wrap_dict(dict),
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = logging.getLogger('train_logs')
logger.setLevel(logging.INFO)

handler = WatchedFileHandler(FILEBEAT_LOGFILE)
logger.addHandler(handler)



mongoClient = MongoClient(MONGO_HOST, MONGO_PORT, maxPoolSize=200, connect=False)

# CELERY STUFF
# CELERY_BROKER_URL = 'redis://localhost:6379'
# CELERY_RESULT_BACKEND = 'redis://localhost:6379'
CELERY_BROKER_URL = 'sqs://AKIAISU4PGJH6TNJ7D2Q:DWMbGw12am06X1qSQuUd0F2ww6iSRvBcU3Na6+dz@'
CELERY_BROKER_TRANSPORT_OPTIONS = {
    'region': 'us-east-1'
}
CELERY_ACCEPT_CONTENT = ['application/json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_IGNORE_RESULT = True
