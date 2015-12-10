#!/usr/bin/env python

# Setup script for the BCC QA tests. This is mainly for use by flashTest but
# can also be invoked directly by users. Some of the BCC QA tools are purely
# IDL, while others involve some compiled code. This script wraps the
# appropriate actions needed to build the code if needed and to move the
# necessary scripts and executables to the place they need to go.

import sys, os, os.path

def get_avail_tasks():
    dirs = os.listdir(".")
    dirs.remove("setup.py")
    dirs.remove("setup_parfile.py")
    return dirs

def usage():
    print "usage: ./setup.py test-type [test-options]"
    print
    outstr = ""
    tasks = get_avail_tasks()
    for task in tasks:
        outstr = outstr + " " + task
    print "    test-type: one of:%s" % outstr
    print "    test-options: use './setup.py test-type help' to see"
    sys.exit(1)

def parse_args():
    args = sys.argv[1:]
    if len(args) == 0:
        usage()
    avail_tasks = get_avail_tasks()
    if args[0] not in avail_tasks:
        usage()
    task_name = args[0]
    task_args = ["./setup.py", "from_main"]
    if len(args) > 1:
        task_args.extend(args[1:])
    return task_name, task_args

def exec_task(task, opts):
    os.chdir(task)
    os.execv("./setup.py", opts)


# Main program

setupdir = os.path.dirname(sys.argv[0])
if setupdir not in ["", "."]:
    os.chdir(setupdir)
task, opts = parse_args()
exec_task(task, opts)
