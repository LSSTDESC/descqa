#!/usr/bin/env python

# Setup script for the BCC QA catalog tests. This is mainly for use by flashTest
# but can also be invoked directly by users.

# At the moment this copies the requested IDL script to a run directory and then
# creates an executable shell script there that runs IDL on it.

import sys, os, os.path, shutil, stat

def usage(is_from_main):
    if is_from_main:
        print "catalog setup arguments: script-name build-dir simdata-dir mockdata-dir realdata-dir"
    else:
        print "usage: ./setup.py script-name build-dir simdata-dir mockdata-dir realdata-dir"
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
#            rootdir = "."
            rootdir = os.path.join(os.getcwd(), "../")
        else:
            is_from_main = 0
            rootdir = os.path.dirname(sys.argv[0])
        if len(args) == 5:
            script   = args[0]
            builddir = args[1]
            simdir   = args[2]
            mockdir  = args[3]
            realdir  = args[4]
        else:
            usage(is_from_main)
    return rootdir, script, builddir, simdir, mockdir, realdir


# Main program

rootdir, script, builddir, simdir, mockdir, realdir = parse_args()

script_fullpath      = os.path.join(rootdir, "catalog", "scripts", script)
script_targetpath    = os.path.join(builddir, "bccqaScript.idl")
globalpar_fullpath   = os.path.join(rootdir, "catalog", "bccqaGlobalConfig.idl")
globalpar_targetpath = os.path.join(builddir, "bccqaGlobalConfig.idl")

if not os.path.isfile(globalpar_fullpath):
    print "catalog setup: file %s not accessible" % globalpar_fullpath
    sys.exit(1)
if not os.path.isfile(script_fullpath):
    print "catalog setup: file %s not accessible" % script_fullpath
    sys.exit(1)
if not os.path.isdir(builddir):
    print "catalog setup: directory %s not accessible" % builddir
    sys.exit(1)

shutil.copy(script_fullpath, script_targetpath)
shutil.copy(globalpar_fullpath, globalpar_targetpath)

os.chdir(builddir)
f = open("bccqa.sh", "w")
f.write("#!/bin/sh\n")
f.write("export BCCQA_ROOT_DIR=%s\n" % rootdir)
f.write("export BCCQA_BUILD_DIR=%s\n" % builddir)
f.write("export BCCQA_SIM_DATA_DIR=%s\n" % simdir)
f.write("export BCCQA_MOCK_DATA_DIR=%s\n" % mockdir)
f.write("export BCCQA_REAL_DATA_DIR=%s\n" % realdir)

# Kludge to get IDL working on orange.
f.write("if [ `hostname` == \"orange\" ]; then\n")
f.write("  source /afs/slac/g/ki/software/idl/idl82/bin/idl_setup.bash\n")
f.write("fi\n")
# End kludge

f.write("idl %s %s" % (globalpar_targetpath, script_targetpath))
f.close()
os.chmod("bccqa.sh", stat.S_IWUSR | stat.S_IREAD | stat.S_IEXEC)
