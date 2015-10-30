#!/usr/bin/env python

# Script to put "runtime parameter files" where they need to be for flashTest.

import sys, os, os.path

def usage(is_from_main):
    if is_from_main:
        print "catalog parfile setup arguments: parfile-name run-dir"
    else:
        print "usage: ./setup_parfile.py parfile-name run-dir"
    print
    sys.exit(1)

def parse_args():
    args = sys.argv[1:]
    if len(args) == 0:
        usage(0)
    else:
        if args[0] == "from_main":
            is_from_main = 1
            args = args[1:]
        else:
            is_from_main = 0
        if len(args) == 2:
            parfile = args[0]
            rundir = args[1]
        else:
            usage(is_from_main)
    return parfile, rundir


# Main program

parfile, rundir = parse_args()

if not os.path.isdir(rundir):
    print "catalog parfile setup: directory %s not accessible" % rundir
    sys.exit(1)

os.chdir(rundir)

if not os.path.isfile(parfile):
    print "catalog parfile setup: file %s not accessible" % parfile
    sys.exit(1)

os.rename(parfile, "descqaTestConfig.py")
