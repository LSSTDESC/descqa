#!/bin/bash

python -m py_compile utils/*.py descqa/*.py

chmod o+r    config
chmod o+rx   *.cgi
chmod o+rx   style utils viewer descqa descqa/templates
chmod o+rwx  cache
chmod o+r    viewer/*
chmod o+rx   viewer/*.cgi
chmod o+r    style/*
chmod o+r    utils/*
chmod -R o+r descqa/*
chmod o+r    ../config_*.py
ln -sf ../config_*.py ./
