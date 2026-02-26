import numpy as np
from scipy.stats import multivariate_normal
from datetime import datetime, timedelta

from stonesoup.types.numeric import Probability
from stonesoup.types.state import StateVector
from stonesoup.types.state import GaussianState

from functions import lonlat2metres


def fit_normal_to_uniform(min_vals, max_vals):

    mu = (min_vals + max_vals) / 2.
    var = (max_vals - min_vals)**2. / 12.
    C = np.diag(var)
    return GaussianState(mu, C)


class PriorELINT:

    def __init__(self, pos_min, pos_max, speed_sd_mps, colours, vis_given_exist, mmsi_prior_pdf,
                 expected_number_unconfirmed_targets):

        self.pos_min = pos_min
        self.pos_max = pos_max
        self.pos_distribution = fit_normal_to_uniform(pos_min, pos_max)
        self.pos_volume = np.prod(self.pos_max - self.pos_min)
        self.prior_pdf = Probability(1 / self.pos_volume)
        self.speed_sd_mps = speed_sd_mps
        self.colours = colours
        self.visibility_given_existence = vis_given_exist
        self.mmsi_prior_pdf = mmsi_prior_pdf
        self.expected_number_unconfirmed_targets = expected_number_unconfirmed_targets

    def get_initial_state_prior(self, lonlat):

        pos_mean_deg = self.pos_distribution.mean.ravel()
        vel_mean_deg = np.zeros(2)

        pos_var_deg = np.diag(self.pos_distribution.covar)
        lon2m, lat2m = lonlat2metres(lonlat[1])
        vel_var_deg = np.array([(self.speed_sd_mps / lon2m) ** 2, (self.speed_sd_mps / lat2m) ** 2])

        prior_mean = np.array([pos_mean_deg[0], vel_mean_deg[0], pos_mean_deg[1], vel_mean_deg[1]])
        prior_cov = np.diag([pos_var_deg[0], vel_var_deg[0], pos_var_deg[1], vel_var_deg[1]])

        return GaussianState(prior_mean, prior_cov)

    def get_null_weight(self, measurement):

        """
        Get likelihood of measurement under the null hypothesis.
        """
        weight = self.prior_pdf * Probability(self.expected_number_unconfirmed_targets)
        for colour_name in measurement.colours.keys():
            weight *= self.colours[colour_name].prior_pdf

        if measurement.mmsi:
            weight *= self.mmsi_prior_pdf[measurement.type]

        return weight
