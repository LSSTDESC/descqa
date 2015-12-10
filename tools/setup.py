#!/usr/bin/env python

# Setup script for the BCC QA tools. This is mainly for use by flashTest but
# can also be invoked directly by users.

# At the moment this doesn't do anything. None of the current tools requires
# anything to be done.

import sys, os

def usage(is_from_main):
    if is_from_main:
        print "tools setup arguments:"
    else:
        print "usage: ./setup.py"
    print
    sys.exit(1)

def parse_args():
    args = sys.argv[1:]
    if len(args) == 0:
        usage(0)
    else:
        usage(args[0] == "from_main")


# Main program

parse_args()
