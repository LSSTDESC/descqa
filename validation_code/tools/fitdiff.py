#!/usr/bin/env python

# Perform a fit to data in an ASCII table file and compare the best-fit
# coefficients to values supplied in another ASCII table file. Two columns
# from the first ("data") file are used as (x,y) values. The fit coefficient
# file is interpreted in the format
#
#     c0 e0 c0-min c0-max
#     c1 e1 c1-min c1-max
#     ...
#
# where each row describes a term in the fitting function
#
#     y = c0*x^e0 + c1*x^e1 + ...
#
# and ci-min and ci-max represent min and max values for each fit coefficient.
# If any of the computed coefficients of the fit to the data fall outside these
# ranges, or if the fit is not statistically significant, FAILURE is triggered;
# otherwise, SUCCESS is triggered.

# TODO: optionally use data errors in fit, determine goodness of fit

# 3/7/13 P. M. Ricker

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
import os.path, sys

#------------------------------------------------------------------------------

def usage():
    print "usage: %s data-file x-col y-col fit-file" % sys.argv[0]
    sys.exit(1)

def validate_numstr(arg):
    try:
        result = int(arg)
    except:
        try:
            result = float(arg)
        except:
            usage()
    return result

def parse_args():
    if len(sys.argv) not in [5]:
        usage()
    file1 = sys.argv[1]
    if not os.path.isfile(file1):
        print "file %s not found or not accessible." % file1
        sys.exit(1)
    xcol1 = validate_numstr(sys.argv[2])
    ycol1 = validate_numstr(sys.argv[3])
    file2 = sys.argv[4]
    if not os.path.isfile(file2):
        print "file %s not found or not accessible." % file2
        sys.exit(1)
    return file1, xcol1, ycol1, file2

def fitfunc(x, *params):
    result = 0.
    for i, e in enumerate(expons):
        result = result + params[i]*x**e
    return result

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

dfile, xcol, ycol, ffile = parse_args()

# Read the files

try:
    x, y = np.loadtxt(dfile, usecols=(xcol-1, ycol-1), unpack=True)
    vd = getvalid(dfile)
except:
    print "error while trying to read file %s." % dfile
    sys.exit(1)

try:
    coeffs, expons, coeff_min, coeff_max = \
                           np.loadtxt(ffile, usecols=(0, 1, 2, 3), unpack=True)
except:
    print "error while trying to read file %s." % ffile
    sys.exit(1)

# Report expectations

print "read fit information from file %s:" % ffile
print "coefficient  exponent     coeff_min    coeff_max"
for c, e, cmin, cmax in zip(coeffs, expons, coeff_min, coeff_max):
    print "%11e  %11e  %11e  %11e" % (c, e, cmin, cmax)
print "testing range: x in [", vd[0], " ... ", vd[1], "]"

# Perform the fit

ok = where((x >= vd[0]) & (x <= vd[1]))
x  = x[ok]
y  = y[o]
copt, ccov = curve_fit(fitfunc, x, y, p0=coeffs)


# Compare fit results to the given ranges and report result

print ""
print "found the following best-fit coefficients:"
print "coefficient  status"

result = "SUCCESS"
for cmeas, cmin, cmax in zip(copt, coeff_min, coeff_max):
    if (np.isnan(cmeas)) or (cmeas < cmin) or (cmeas > cmax):
        result = "FAILURE"
        fit_stat = "BAD"
    else:
        fit_stat = "OK"
    print "%11e  %s" % (cmeas, fit_stat)

print ""
print result

