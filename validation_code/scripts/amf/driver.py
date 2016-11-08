import os
import subprocess
import numpy as np

# Planck LCDM
om = 0.3089
ob = 0.0486
h  = 0.6774
s8 = 0.816
ns = 0.9667

# Example call
FNULL = open(os.devnull, 'w')
p = subprocess.call(["./amf.exe", "-omega_0", str(om), "-omega_bar", str(ob), "-h", str(h),
                     "-sigma_8", str(s8), "-n_s", str(ns), "-tf", "EH"], stdout=FNULL, stderr=FNULL)
MassFunc = np.loadtxt("analytic.dat").T
