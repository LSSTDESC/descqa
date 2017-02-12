#!/bin/bash

python -m py_compile utils/*.py

chmod o+r   config
chmod o+rx  home.py
chmod o+rx  style utils viewer
chmod o+rwx cache
chmod o+r   viewer/*
chmod o+rx  viewer/*.py
chmod o+r   style/*
chmod o+r   utils/*
