#!/bin/bash
gunicorn -w 3 eschernode.wsgi