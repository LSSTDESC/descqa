#!/bin/bash

python -m py_compile utils/*.py

chmod o+r   config
chmod o+rx  *.cgi
chmod o+rx  style utils viewer descqa descqa/templates
chmod o+rwx cache
chmod o+r   viewer/*
chmod o+rx  viewer/*.cgi
chmod o+r   style/*
chmod o+r   utils/*
chmod o+r   descqa/*
chmod o+rx  descqa/*.py
chmod o+r   descqa/templates/*
