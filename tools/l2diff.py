#!/usr/bin/env python

# Diff columns in two ASCII table files and produce the L2 norm of the
# difference as output. Two columns from each file are used as (x,y) 
# values. The first file is treated as the "theory" file; its y-values
# are interpolated to the x-values from the second ("data") file. SUCCESS
# or FAILURE is triggered depending on whether the result exceeds a given
# threshold. A third column from the "data" file can be used to give errors.
# In this case a reduced chi^2 value and significance are also output if
# the number of parameters used in the theory curve is given.

# 10/24/12 P. M. Ricker

import numpy as np
from scipy.stats import chi2
import os.path, sys, argparse

#------------------------------------------------------------------------------

def getvalid(file):
    infinity = float("inf")
    f = open(file)
    line = f.readline()
    if "# valid" in line:
        line = filter(None, line.split(" "))
        v = [float(line[2]), float(line[3])]
    else:
        line = f.readline()
        if "# valid" in line:
            line = filter(None, line.split(" "))
            v = [float(line[2]), float(line[3])]
        else:
            v = [-infinity, infinity]
    f.close()
    return v

#------------------------------------------------------------------------------

# Parse command line

parser = argparse.ArgumentParser(prog='l2diff', description='Diff columns in two ASCII table files and produce the L2 norm of the difference as output.')

parser.add_argument('theory_file', \
                    help='path of file to treat as theory/fit curve')
parser.add_argument('data_file', \
                    help='path of file to treat as measured data')
parser.add_argument('-tx', '--theory-xcolumn', type=int,   default=1, \
                    metavar='#', \
                    help='column containing theory x-values')
parser.add_argument('-ty', '--theory-ycolumn', type=int,   default=2, \
                    metavar='#', \
                    help='column containing theory y-values')
parser.add_argument('-dx', '--data-xcolumn',   type=int,   default=1, \
                    metavar='#', \
                    help='column containing data x-values')
parser.add_argument('-dy', '--data-ycolumn',   type=int,   default=2, \
                    metavar='#', \
                    help='column containing data y-values')
parser.add_argument('-t',  '--threshold',      type=float, default=0., \
                    metavar='#', \
                    help='threshold for grading SUCCESS or FAILURE')
parser.add_argument('-e',  '--data-ecolumn',   type=int,   default=None, \
                    metavar='#', \
                    help='column containing data errors')
parser.add_argument('-n',  '--num-params',     type=int,   default=None, \
                    metavar='#', \
                    help='number of fit parameters for computing significance')

result = vars(parser.parse_args())

file1     = result['theory_file']
xcol1     = result['theory_xcolumn']
ycol1     = result['theory_ycolumn']
file2     = result['data_file']
xcol2     = result['data_xcolumn']
ycol2     = result['data_ycolumn']
threshold = result['threshold']
ecol2     = result['data_ecolumn']
npar      = result['num_params']

# Read the files

if not os.path.isfile(file1):
    print "file %s not found or not accessible." % file1
    sys.exit(1)
if not os.path.isfile(file2):
    print "file %s not found or not accessible." % file2
    sys.exit(1)

try:
    x1, y1 = np.loadtxt(file1, usecols=(xcol1-1, ycol1-1), unpack=True)
    v1 = getvalid(file1)
except:
    print "error while trying to read file %s." % file1
    sys.exit(1)

try:
    if ecol2:
        x2, y2, e2 = np.loadtxt(file2, usecols=(xcol2-1, ycol2-1, ecol2-1), unpack=True)
    else:
        x2, y2 = np.loadtxt(file2, usecols=(xcol2-1, ycol2-1), unpack=True)
    v2 = getvalid(file2)
except:
    print "error while trying to read file %s." % file2
    sys.exit(1)

# Limit comparison to "valid" range specified in files

ok1 = np.where((x1 >= v1[0]) & (x1 <= v1[1]))
x1  = x1[ok1]
y1  = y1[ok1]

ok2 = np.where((x2 >= v2[0]) & (x2 <= v2[1]))
x2  = x2[ok2]
y2  = y2[ok2]
if ecol2:
    e2 = e2[ok2]

# Interpolate theory curve to data x-points and compute L2 norm and significance

y1int = np.interp(x2, x1, y1)

if ecol2:
    L2 = (np.sum( (y2 - y1int)**2 / e2**2 ))**0.5
else:
    L2 = (np.sum( (y2 - y1int)**2 ))**0.5

if npar:
    chi2red = L2**2 / (len(y2)-npar)
    p = 1. - chi2.cdf(L2**2, len(y2)-npar)
    print "L2 = %G  chi2red = %G  p = %G" % (L2, chi2red, p)
else:
    print "L2 = %G" % L2

# Issue verdict

if (L2 > threshold) or (np.isnan(L2)):
    print "FAILURE"
else:
    print "SUCCESS"
