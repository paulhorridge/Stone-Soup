import numpy as np
import copy
from scipy.stats import multivariate_normal

from stonesoup.types.array import CovarianceMatrix
from stonesoup.types.state import GaussianState
from stonesoup.types.numeric import Probability

def lonlat2metres(lat_deg):

    earth_radius_metres = 6371.e3
    lat2m = np.pi * earth_radius_metres / 180
    lon2m = lat2m * np.cos(np.deg2rad(lat_deg))
    return lon2m, lat2m


def kalman_update(prior, H, R, y):

    ybar = H @ prior.mean
    S = H @ prior.covar @ H.transpose() + R
    K = prior.covar @ H.transpose()
    innov = (y - ybar)
    KinvS = K @ np.linalg.inv(S)

    xhat = prior.mean + KinvS @ innov
    P = prior.covar - KinvS @ K.transpose()
    post = GaussianState(xhat, (P + P.transpose())/2.0)
    likelihood = Probability(multivariate_normal.logpdf(innov.ravel(), cov=S), log_value=True)

    return post, likelihood


def merge_two_gaussians(weight1, gaussian1, weight2, gaussian2):

    new_weight = weight1 + weight2
    new_mean = float(weight1 / new_weight) * gaussian1.mean + float(weight2 / new_weight) * gaussian2.mean
    dx1 = gaussian1.mean - new_mean
    dx2 = gaussian2.mean - new_mean

    C1 = gaussian1.covar + dx1 * dx1.transpose()
    C2 = gaussian2.covar + dx2 * dx2.transpose()
    new_covar = float(weight1 / new_weight) * C1 + float(weight2 / new_weight) * C2

    return new_weight, GaussianState(new_mean, new_covar)


def merge_gaussians(weighted_components):

    new_weight = weighted_components[0][0]
    new_component = copy.copy(weighted_components[0][1])

    for (w, c) in weighted_components[1:]:
        new_weight, new_component = merge_two_gaussians(new_weight, new_component, w, c)

    return new_weight, new_component


def get_mahalanobis_squared(measurement, xbar, M, A, Q, H, R):

    dx = H @ (A @ xbar) - measurement
    P = A @ M @ A.T + Q
    S = H @ P @ H.T + R
    return (dx.T @ np.linalg.inv(S) @ dx)[0]
