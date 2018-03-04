from __future__ import division
from builtins import range
import numpy as np
from scipy.stats import chi2


def get_subvolume_indices(x, y, z, box_size, n_side):
    side_size = box_size/n_side
    return np.ravel_multi_index(np.floor(np.vstack((x, y, z))/side_size).astype(int), (n_side,)*3, 'wrap')


def jackknife(data, jack_indices, n_jack, func, full_args=(), full_kwargs={}, jack_args=(), jack_kwargs={}):
    if len(data) != len(jack_indices):
        raise ValueError('`data` and `jack_indices` must have the same length')
    if not np.in1d(jack_indices, np.arange(n_jack)).all():
        raise ValueError('`jack_indices` must be an array of int between 0 to n_jack-1')

    full = np.array(func(data, *full_args, **full_kwargs), dtype=np.float)

    jack = []
    for i in range(n_jack):
        jack.append(func(data[jack_indices != i], *jack_args, **jack_kwargs))
    jack = np.array(jack, dtype=np.float)

    bias = (jack.mean(axis=0) - full)*(n_jack-1)
    return full-bias, bias, np.cov(jack, rowvar=False, bias=True)*float(n_jack-1)


def chisq(difference, covariance, dof):
    d = np.asarray(difference)
    cov = np.asarray(covariance)
    if cov.ndim == 1:
        cov = np.diag(cov)
    chisq_value = np.dot(d, np.dot(np.linalg.inv(cov), d))
    return chisq_value, chi2.cdf(chisq_value, dof)


def Lp_norm(difference, p=2.0):
    d = np.asarray(difference)
    d **= p
    return d.sum() ** (1.0/p)


def AD_statistic(n1, n2, y1, y2, threshold):
    '''
    Calculate the two-sample Anderson-Darling statistic from two CDFs;
    n1, n2: number of objects in the two samples;
    y1, y2: CDF y-values of the two distribution, and they should have
    the same x-axis.
    '''
    n = n1+n2
    h = (n1*y1+n2*y2)/n
    # compute Anderson-Darling statistic
    inv_weight = (h*(1-h))[:-1]
    # remove infinities in the weight function
    mask = (inv_weight<1e-5)
    inv_weight[mask] = 1
    ads = n1*n2/n * np.sum(((y2 - y1)[:-1])**2*(h[1:]-h[:-1])/inv_weight)

    if ads<threshold:
        success = True
    else:
        success = False

    return ads, success


# def CvM_statistic(n1, n2, y1, y2, threshold):
#     '''
#     Calculate the two-sample Cramer-von Mises statistic from two CDFs;
#     n1, n2: number of objects in the two samples;
#     y1, y2: CDF y-values of the two distribution, and they should have
#     the same x-axis.
#     '''
#     n = n1+n2
#     h = (n1*y1+n2*y2)/n
#     cvm_omega = np.sqrt(np.trapz((y2-y1)**2, h))

#     if cvm_omega<threshold:
#         success = True
#     else:
#         success = False

#     return cvm_omega, success


def CvM_statistic(n1, n2, x1, y1, x2, y2):
    '''
    Calculate the two-sample Cramer-von Mises statistic from two CDFs;
    n1, n2: number of objects in the two samples;
    y1, y2: CDF y-values of the two distribution, and they should have
    the same x-axis.
    '''
    n = n1+n2
    x_interp = np.linspace(-2, 5, 10000)
    y1_interp = np.interp(x_interp, x1, y1)
    y2_interp = np.interp(x_interp, x2, y2)
    h = (n1*y1_interp+n2*y2_interp)/n

    cvm_omega = np.sqrt(np.trapz((y2_interp-y1_interp)**2, x=h))

    return cvm_omega
