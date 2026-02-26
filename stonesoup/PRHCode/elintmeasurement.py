import numpy as np
from scipy.linalg import expm
from scipy.stats import multivariate_normal

from datetime import datetime, timedelta

from stonesoup.types.numeric import Probability
from stonesoup.types.state import StateVector
from stonesoup.types.state import GaussianState

from functions import lonlat2metres


def smaj_smin_orient2cov_m(semimajor_sd, semiminor_sd, orient_deg):

    orient_rad = np.deg2rad(orient_deg)
    rot = np.array([[np.cos(orient_rad), np.sin(orient_rad)],  [-np.sin(orient_rad), np.cos(orient_rad)]])
    R_norm = np.diag([semiminor_sd**2.0, semimajor_sd**2])
    return rot @ R_norm @ rot.transpose()


def get_lon_lat_R(lonlat, semimajor_sd, semiminor_sd, orient_deg):

    R_metres = smaj_smin_orient2cov_m(semimajor_sd, semiminor_sd, orient_deg)
    lon2m, lat2m = lonlat2metres(lonlat[1])
    H = np.diag([1./lon2m, 1./lat2m])
    R_lonlat = H @ R_metres @ H.transpose()

    return R_lonlat


class ELINTMeasurement: # might be AIS so maybe come up with a better name...

    def __init__(self, in_data, sensor, type, colours):

        self.type = type # remove type later and use sensor.type?
        self.sensor = sensor
        self.timestamp = datetime.strptime(in_data["times"], "%d-%b-%Y %H:%M:%S")
        self.coords = StateVector(in_data["coords"])
        self.mmsi = in_data.get("mmsi", None)

        if "semimajorSD" in in_data:
            self.R = get_lon_lat_R(self.coords, in_data["semimajorSD"], in_data["semiminorSD"], in_data["orientation"])
        else:
            self.R = 1.e-4*np.eye(2) # TODO: Pass in?

        self.colours = {}
        for colour_name in colours.keys():
            colour_data = in_data.get(colour_name, None)
            if colour_data is not None:
                self.colours[colour_name] = np.atleast_1d(colour_data)


class ELINTColour:

    def __init__(self, in_data):

        self.name = in_data["name"]
        self.range = in_data["range"]
        self.prior = GaussianState([np.array(in_data["mean"])], np.array([[in_data["cov"]]]))
        self.prior_pdf = Probability(1/(self.range[1] - self.range[0]))
        self.R = np.array([[in_data["measCov"]]])
        self.q = in_data["q"]
        self.is_switch = in_data["isSwitch"]
        if self.is_switch:
            self.prior_switch_prob = Probability(in_data["priorSwitchProb"])
            self.switch_rate_0to1 = in_data["switchRate0to1"]
            self.switch_rate_1to0 = in_data["switchRate1to0"]
        self.is_harmonic = in_data["isHarmonic"]
        if self.is_harmonic:
            self.harmonic_probs = [Probability(log_p, log_value=True) for log_p in in_data["harmonicLogProbs"]]

    def get_mahalanobis_squared(self, measurement, distribution, dt):

        pred_comp_mean = distribution.mean[0]
        pred_comp_cov = distribution.covar[0][0] + self.q * dt.total_seconds()

        if self.is_harmonic:
            mahal_sq = np.inf
            for i in range(len(self.harmonic_probs)):
                this_meas_mn = (i+1) * pred_comp_mean
                this_meas_cv = (i+1)**2 * pred_comp_cov + self.R[0][0]
                this_mahal_sq = (measurement[0] - this_meas_mn)**2 / this_meas_cv
                mahal_sq = min(mahal_sq, this_mahal_sq)
        else:
            meas_mn = pred_comp_mean
            meas_cv = pred_comp_cov + self.R[0][0]
            mahal_sq = (measurement[0] - meas_mn)**2 / meas_cv

        return mahal_sq

    def get_likelihood(self, measurement, distribution):

        if self.is_harmonic:

            likelihood = Probability(0)
            for i, prob in enumerate(self.harmonic_probs):
                this_meas_mn = (i+1) * distribution.mean
                innov = measurement - this_meas_mn
                S = (i + 1) ** 2 * distribution.covar + self.R
                this_likelihood = Probability(multivariate_normal.logpdf(innov.ravel(), cov=S), log_value=True)
                likelihood += prob * this_likelihood

            return likelihood

        else:

            meas_mn = distribution.mean
            innov = measurement - meas_mn
            S = distribution.covar + self.R

            return Probability(multivariate_normal.logpdf(innov.ravel(), cov=S), log_value=True)

    def get_switch_probs(self, dt):

        if self.is_switch:
            # [-rate12 rate12; rate21 - rate21];
            rate_matrix = np.array([[-self.switch_rate_0to1,  self.switch_rate_0to1],
                                    [ self.switch_rate_1to0, -self.switch_rate_1to0]])
            prob_matrix_m = expm(rate_matrix*dt.total_seconds())
            v_constructor = np.vectorize(Probability)
            prob_matrix = v_constructor(prob_matrix_m)

        else:

            prob_matrix = np.matrix([[Probability(1.0)]])

        return prob_matrix

    def predict(self, old_distribution, dt, model_index):

        if model_index == 0:

            # non-switch model
            Q = np.array([[self.q * dt.total_seconds()]])
            new_distribution = GaussianState(old_distribution.mean, old_distribution.covar + Q)

        else:

            # switch model
            new_distribution = GaussianState(self.prior.mean, self.prior.covar)

        return new_distribution

class SensorELINT:

    def __init__(self, sensor_type, measurement_model, mean_meas_time, mean_reveal_time, mean_hide_time):

        self.type = sensor_type
        self.measurement_model = measurement_model
        self.mean_meas_time = mean_meas_time
        self.mean_reveal_time = mean_reveal_time
        self.mean_hide_time = mean_hide_time


    def get_visibility_detection_transition(self, dt):

        meas_c = dt / self.mean_meas_time
        reveal_c = dt / self.mean_reveal_time
        hide_c = dt / self.mean_hide_time

        A = np.matrix([
            [-reveal_c, reveal_c,           0,         0],
            [hide_c,    -(hide_c + meas_c), 0,         meas_c],
            [0,         0,                  -reveal_c, reveal_c],
            [0,         0,                  hide_c,    -hide_c]])
        prob_matrix = expm(A)

        # Transition matrix if target is originally not visible
        vis_det_0 = np.matrix([[Probability(prob_matrix[0, 0]), Probability(prob_matrix[0, 2])],
                               [Probability(prob_matrix[0, 1]), Probability(prob_matrix[0, 3])]])
        # Transition matrix if target is originally visible
        vis_det_1 = np.matrix([[Probability(prob_matrix[1, 0]), Probability(prob_matrix[1, 2])],
                               [Probability(prob_matrix[1, 1]), Probability(prob_matrix[1, 3])]])

        vis_det_trans = [vis_det_0, vis_det_1]

        return vis_det_trans
