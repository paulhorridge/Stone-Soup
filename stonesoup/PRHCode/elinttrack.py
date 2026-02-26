import numpy as np
from functools import reduce
import operator

from scipy.stats import multivariate_normal
from datetime import datetime, timedelta

from stonesoup.types.state import GaussianState
from stonesoup.types.numeric import Probability

from elintmeasurement import ELINTMeasurement, ELINTColour

from functions import kalman_update, merge_gaussians,merge_two_gaussians, get_mahalanobis_squared


class TrackLikelihood:

    """
    Likelihood of measurement from a track
    """
    def __init__(self, likelihood, association_weight, component_likelihoods):

        # Prob(measurement | track, detected)
        self.likelihood  = likelihood
        # Weight of the measurement association hypothesis (Likelihood * detection probability term)
        self.association_weight = association_weight
        # Likelihoods for each track component
        self.component_likelihoods = component_likelihoods

class TrackComponentLikelihood:

    """
    Likelihood of measurement from a track component
    """
    def __init__(self, likelihood, association_weight, colour_likelihood):

        # P(measurement | component, detected)
        self.likelihood = likelihood
        # Weight of the measurement association hypothesis (Likelihood * detection probability term)
        self.association_weight = association_weight
        # Likelihood for the colour distribution
        self.colour_likelihood = colour_likelihood


class ColourDistributionLikelihood:

    """
    Likelihood of the colour measurements from a colour distribution
    """
    def __init__(self, likelihood, single_colour_likelihoods):

        #p(colour measurements | colour distribution)
        self.likelihood = likelihood
        self.single_colour_likelihoods = single_colour_likelihoods

class SingleColourLikelihood:

    """
    Likelihood of a single colour measurement from a colour distribution
    """
    def __init__(self, likelihood, component_likelihoods):

        self.likelihood = likelihood
        self.component_likelihoods = component_likelihoods


class VisibilityDetectionPrediction:

    """
    Represent the visibility and detection of a track component to allow visibility and existence probability update
    """
    def __init__(self, visibility_given_existence, vis_det_trans):

        self.visibility_and_not_detected_given_existence = {}
        for sensor_type, this_visibility_given_existence in visibility_given_existence.items():

            # Probability of transition from hidden to hidden (and not detected)
            ptrans_hidden_to_hidden = vis_det_trans[sensor_type][0][0, 0]
            # Probability of transition from visible to hidden (and not detected)
            ptrans_visible_to_hidden = vis_det_trans[sensor_type][1][0, 0]
            # Probability of transition from hidden to visible but not detected
            ptrans_hidden_to_visible = vis_det_trans[sensor_type][0][1, 0]
            # Probability of transition from visible to visible but not detected for each sensor
            ptrans_visible_to_visible_not_detected = vis_det_trans[sensor_type][1][1, 0]

            this_not_vis_and_not_det = this_visibility_given_existence * ptrans_visible_to_hidden + \
                (1 - this_visibility_given_existence) * ptrans_hidden_to_hidden
            this_vis_and_not_det = this_visibility_given_existence * ptrans_visible_to_visible_not_detected + \
                (1 - this_visibility_given_existence) * ptrans_hidden_to_visible

            self.visibility_and_not_detected_given_existence[sensor_type] = (
                this_not_vis_and_not_det, this_vis_and_not_det)

    def probability_not_detected_given_exist(self):

        return {sensor_type: sum(ps) for sensor_type, ps in self.visibility_and_not_detected_given_existence.items()}

    def probability_visible_given_exist_and_not_detected(self):

        return {sensor_type: ps[1] / sum(ps) for sensor_type, ps in self.visibility_and_not_detected_given_existence.items()}

    def probability_not_detected_any_given_exist(self):
        """
        Get probability of not being detected by any sensor, conditional on existence
        """
        pds = self.probability_not_detected_given_exist().values()
        return reduce(operator.mul, [p for p in pds], Probability(1.0))


class ColourComponent:

    def __init__(self, distribution : GaussianState, weight : Probability = Probability(1.0)):

        self.weight = weight
        self.distribution = distribution

    def duplicate(self):

        return ColourComponent(self.distribution, self.weight)

    def to_string(self):

        out = '{"log_weight": ' + str(self.weight.log_value)
        out += ', "mean": ' + str(self.distribution.mean[0])
        out += ', "var": ' + str(self.distribution.covar[0, 0]) + "}"
        return out

    def get_mahalanobis_squared(self, measurement : ELINTMeasurement, colour : ELINTColour, dt : timedelta) -> float:

        return colour.get_mahalanobis_squared(measurement, self.distribution, dt)

    def get_likelihood(self, colour_measurement, colour : ELINTColour):

        return colour.get_likelihood(colour_measurement, self.distribution)

    def update(self, colour_measurement, colour : ELINTColour):

        new_distribution = None
        new_weight = Probability(0.0)
        if colour.is_harmonic:
            for i, harmonic_prob in enumerate(colour.harmonic_probs):
                H = np.array([[float(i + 1)]])
                this_distribution, this_likelihood = kalman_update(self.distribution, H, colour.R, colour_measurement)
                this_weight = this_likelihood * harmonic_prob
                if new_distribution is None:
                    new_distribution = this_distribution
                    new_weight = this_weight
                else:
                    new_weight, new_distribution = merge_two_gaussians(
                        new_weight, new_distribution, this_weight, this_distribution)
            self.distribution = new_distribution
        else:
            self.distribution, _ = kalman_update(self.distribution, np.array([[1.0]]), colour.R, colour_measurement)


class SingleColourDistribution:
    """
    Represents the probability distribution of a single colour. This could have multiple weighted components due to
    model switching
    """
    def __init__(self, colour_components : dict[int, ColourComponent] , colour):

        self.components = colour_components
        self.colour = colour
        self.normalise()

    def duplicate(self):
        return SingleColourDistribution(
            {model_index: component.duplicate() for model_index, component in self.components.items()},
            self.colour)

    def to_string(self):

        out = '[\n' + ',\n'.join(['{"model_index": ' + str(model_index) +
                                  ', "component": ' + component.to_string() + '}' for
                                  model_index, component in self.components.items()]) + "]"
        return out

    def normalise(self):

        tot_weight = Probability(0.0)
        for c in self.components.values():
            tot_weight += c.weight

        for model_index in self.components:
            self.components[model_index].weight /= tot_weight

    def get_mahalanobis_squared(self, measurement, dt : timedelta):

        #if self.colour.is_switch:
        #    return 0.0

        # Get minimum across components
        mahal_sq = np.inf
        for c in self.components.values():
            this_mahal_sq = c.get_mahalanobis_squared(measurement, self.colour, dt)
            if this_mahal_sq < mahal_sq:
                mahal_sq = this_mahal_sq

        return mahal_sq

    def predict(self, dt : timedelta):

        if self.colour.is_switch:

            switch_matrix = self.colour.get_switch_probs(dt)
            nmodels = len(self.components)
            new_components = {}
            for old_m in range(nmodels):
                for new_m in range(nmodels):
                    old_component = self.components[old_m]
                    this_weight = switch_matrix[old_m, new_m] * old_component.weight
                    this_gaussian = self.colour.predict(old_component.distribution, dt, new_m)
                    if new_m in new_components:
                        new_c = new_components[new_m]
                        new_components[new_m] = merge_two_gaussians(
                            new_c[0], new_c[1], this_weight, this_gaussian)
                    else:
                        new_components[new_m] = (this_weight, this_gaussian)

            self.components = {model_index: ColourComponent(c[1], c[0]) for
                               model_index, c in new_components.items()}
            self.normalise()

        else:

            for model_index, c in self.components.items():
                c.distribution = self.colour.predict(c.distribution, dt, model_index)

    def get_likelihood(self, colour_measurement):

        component_likelihoods = {}
        likelihood = Probability(0)
        for model_index, c in self.components.items():
            this_likelihood = c.get_likelihood(colour_measurement, self.colour)
            component_likelihoods[model_index] = this_likelihood
            likelihood += c.weight * this_likelihood

        return SingleColourLikelihood(likelihood, component_likelihoods)

    def update(self, colour_measurement, single_colour_likelihood):

        tot_weight = Probability(0.0)
        for model_index, likelihood in single_colour_likelihood.component_likelihoods.items():#'.component_likelihoods().items():
            self.components[model_index].weight *= likelihood
            self.components[model_index].update(colour_measurement, self.colour)
            tot_weight += self.components[model_index].weight

        self.normalise()

    @staticmethod
    def merge(weighted_distributions): # list[tuple[Probability, SingleColourDistribution]]

        new_distribution = weighted_distributions[0][1].duplicate()
        for distribution in weighted_distributions[1:]:
            assert(new_distribution.colour == distribution[1].colour)

        # Cluster components according to model index
        clusters = {}
        for this_weight, distribution in weighted_distributions:
            for model_index, component in distribution.components.items():
                this_weighted_component = this_weight * component.weight, component.distribution
                if model_index in clusters.keys():
                    clusters[model_index].append(this_weighted_component)
                else:
                    clusters[model_index] = [this_weighted_component]

        new_components = {}
        for model_index, components in clusters.items():
            this_weight, this_distribution = merge_gaussians(components)
            new_components[model_index] = ColourComponent(distribution=this_distribution, weight=this_weight)

        new_distribution.components = new_components
        new_distribution.normalise()

        return new_distribution


class ColourDistribution:
    """
    Represents a probability distribution of multiple colours.
    """
    def __init__(self, single_colour_distributions: dict[str, SingleColourDistribution]):

        self.single_colour_distributions = {sensor_type: colour_distribution.duplicate()
                                            for sensor_type, colour_distribution in single_colour_distributions.items()}

    def duplicate(self):

        return ColourDistribution(self.single_colour_distributions)

    def to_string(self):
        return '{\n' + ',\n'.join(['"' + name + '":' + distribution.to_string()
                                   for name, distribution in self.single_colour_distributions.items()]) + "\n}"

    def get_mahalanobis_squared(self, measurement : ELINTMeasurement, dt : timedelta):

        mahal_sq = 0.0
        for colour_name, colour_measurement in measurement.colours.items():

            mahal_sq += self.single_colour_distributions[colour_name].get_mahalanobis_squared(
                colour_measurement, dt)

        return mahal_sq

    def get_likelihood(self, measurement):

        single_colour_likelihoods = {}
        likelihood = Probability(1)

        for colour_type, colour_measurement in measurement.colours.items():
            this_colour_likelihood = self.single_colour_distributions[colour_type].get_likelihood(colour_measurement)
            single_colour_likelihoods[colour_type] = this_colour_likelihood
            likelihood *= this_colour_likelihood.likelihood

        return ColourDistributionLikelihood(likelihood, single_colour_likelihoods)

    def predict(self, dt : timedelta):

        for colour_name, colour_distribution in self.single_colour_distributions.items():
            self.single_colour_distributions[colour_name].predict(dt)

    def update(self, measurement, likelihood):

        for colour_type, colour_measurement in measurement.colours.items():
            self.single_colour_distributions[colour_type].update(colour_measurement,
                                                                 likelihood.single_colour_likelihoods[colour_type])

    @staticmethod
    def merge(weighted_colour_distributions): # list[tuple[Probability, ColourDistribution]]

        single_colour_distributions = {}
        for colour_type in weighted_colour_distributions[0][1].single_colour_distributions.keys():
            these_colour_distributions = [
                (w[0], w[1].single_colour_distributions[colour_type]) for w in weighted_colour_distributions]
            single_colour_distributions[colour_type] = SingleColourDistribution.merge(these_colour_distributions)

        return ColourDistribution(single_colour_distributions)


"""
class ELINTTrackComponentBase:
    
    def __init__(self,
                 state_distribution: GaussianState,
                 colour_distribution: ColourDistribution,
                 measurement_history: list[ELINTMeasurement | None],
                 mmsi: str,
                 exist_probability: Probability,
                 weight: Probability):
        
        self.weight = weight
        self.state_distribution = state_distribution
        self.colour_distribution = colour_distribution.duplicate() #{sensor_type: c.duplicate() for sensor_type, c in colour_distribution.items()}
        self.mmsi = mmsi
        self.exist_probability = exist_probability
        self.measurement_history = [m for m in measurement_history]

    def duplicate(self):

        return ELINTTrackComponentBase(self.state_distribution, self.colour_distribution, self.measurement_history,
                                       self.mmsi, self.exist_probability, self.weight)
"""

class ELINTTrackComponent:#(ELINTTrackComponentBase):
    """
    Represent a single measurement hypothesis component of a track, consisting of a weight, Gaussian state distribution,
    measurement history, colour distribution for each colour, mmsi, and existence and visibility information
    """
    def __init__(self,
                 state_distribution : GaussianState,
                 colour_distribution : ColourDistribution,
                 measurement_history : list[ELINTMeasurement | None],
                 mmsi : str,
                 exist_probability : Probability,
                 visibility_given_existence : dict[str, Probability],
                 weight: Probability):

        #super().__init__(state_distribution, colour_distribution, measurement_history, mmsi,
        #         exist_probability, weight)
        self.weight = weight
        self.state_distribution = state_distribution
        self.colour_distribution = colour_distribution.duplicate() #{sensor_type: c.duplicate() for sensor_type, c in colour_distribution.items()}
        self.mmsi = mmsi
        self.exist_probability = exist_probability
        self.visibility_given_existence = {sensor_type: prob for
                                           sensor_type, prob in visibility_given_existence.items()}
        self.measurement_history = [m for m in measurement_history]
        self.visibility_detection_prediction = None

    def duplicate(self):

        return ELINTTrackComponent(
            self.state_distribution,
            self.colour_distribution,
            self.measurement_history,
            self.mmsi,
            self.exist_probability,
            self.visibility_given_existence,
            self.weight)

    def to_string(self):

        # Weight
        out = '{\n"log_weight": ' + str(self.weight.log_value) + ',\n'
        # State mean and covariance
        out += '"state_distribution":\n{\n'
        out += '"mean": [' + ','.join([str(x) for x in self.state_distribution.mean.ravel()]) + '],\n'
        out += '"covar": [' + ','.join([str(x) for x in self.state_distribution.covar.ravel()]) + ']\n'
        out += '},\n'
        out += '"colour_distribution": ' + self.colour_distribution.to_string() + ',\n'
        # MMSI
        out += '"mmsi": ' + ('""' if self.mmsi is None else '"' + self.mmsi + '"') + ',\n'
        # Existence probability
        out += '"log_exist_prob": ' + str(self.exist_probability.log_value) + ',\n'
        out += '"log_vis_given_existence": {'
        out += ', '.join(['"' + sensor_type + '": ' + str(p.log_value) for sensor_type, p
                in self.visibility_given_existence.items()]) + "},\n"
        out += '"measurement_history": [' + ', '.join(['""' if x is None else '"' +x.type + '"' for x in self.measurement_history]) + ']\n'

        out += "}"

        return out

    def predict_existence_and_visibility(self, survival_prob, vis_det_trans):

        self.exist_probability *= survival_prob
        self.visibility_detection_prediction = VisibilityDetectionPrediction(
            self.visibility_given_existence, vis_det_trans)

        for sensor_type, vis_prob in self.visibility_given_existence.items():
            hide_prob = Probability(np.sum(vis_det_trans[sensor_type][1][0, :]))
            reveal_prob = Probability(np.sum(vis_det_trans[sensor_type][0][1, :]))
            new_vis_prob = (1 - hide_prob) * vis_prob + reveal_prob * (1 - vis_prob)
            self.visibility_given_existence[sensor_type] = new_vis_prob

    def _get_state_likelihood(self,
                       measurement : ELINTMeasurement):

        H = measurement.sensor.measurement_model.matrix()
        R = measurement.R
        zbar = H @ self.state_distribution.mean
        innov = zbar - measurement.coords
        S = H @ self.state_distribution.covar @ H.transpose() + R

        return Probability(multivariate_normal.logpdf(innov.ravel(), cov=S), log_value=True)

    def _get_mmsi_likelihood(self, measurement):

        mmsi_prior_pdf = Probability(1.e-5) # TODO: Remove these being hacked in
        mmsi_prob_match = Probability(1.0)

        if measurement.mmsi is None:
            return Probability(1.0)
        else:
            if self.mmsi is None:
                return mmsi_prior_pdf
            else:
                if self.mmsi == measurement.mmsi:
                    return mmsi_prob_match
                else:
                    return (1 - mmsi_prob_match)*mmsi_prior_pdf

    def get_likelihood(self, measurement, dt):

        total_likelihood = self._get_state_likelihood(measurement) * self._get_mmsi_likelihood(measurement)
        colour_likelihood = self.colour_distribution.get_likelihood(measurement)
        total_likelihood *= colour_likelihood.likelihood

        # Get association probability (likelihood multiplied by prob of detection stuff)
        # Probability of being detected by this sensor
        this_visibility_probability = self.visibility_given_existence[measurement.type] * self.exist_probability

        # Probability of not detected given existing for each sensor
        prob_not_detected_given_exists = self.visibility_detection_prediction.probability_not_detected_given_exist()

        pd_others = [p for (sensor_type, p) in prob_not_detected_given_exists.items() if sensor_type != measurement.type]
        prob_not_detected_others_given_exists = reduce(operator.mul, pd_others, Probability(1.0))
        # sensor rate probability (come up with better name!)
        prob_sensor = Probability(-dt / measurement.sensor.mean_meas_time, log_value=True)

        # Calculate association hypothesis weight
        association_weight = (total_likelihood * this_visibility_probability * prob_sensor *
                              prob_not_detected_others_given_exists)

        return TrackComponentLikelihood(total_likelihood, association_weight, colour_likelihood)

    def get_null_hypothesis_weight(self):

        prob_not_detected_given_exists = self.visibility_detection_prediction.probability_not_detected_given_exist()
        prob_not_detected_all_given_exists = reduce(
            operator.mul,[p for p in prob_not_detected_given_exists.values()],Probability(1.0))
        prob_not_detected_all_and_exists = prob_not_detected_all_given_exists * self.exist_probability
        null_weight = prob_not_detected_all_and_exists + (1 - self.exist_probability)

        return null_weight

    def get_updated_missed_detection(self, assign_probability):

        updated = self.duplicate()

        null_weight = self.get_null_hypothesis_weight()
        updated.weight = (1-assign_probability) * null_weight * self.weight
        updated.measurement_history = self.measurement_history + [None]

        prob_not_detected_given_exists = self.visibility_detection_prediction.probability_not_detected_any_given_exist()
        prob_not_detected_and_exists = self.exist_probability * prob_not_detected_given_exists
        prob_not_detected = prob_not_detected_and_exists + (1 - self.exist_probability)
        exists_given_not_detected = prob_not_detected_and_exists / prob_not_detected

        updated.exist_probability = exists_given_not_detected
        updated.visibility_given_existence =(
            self.visibility_detection_prediction.probability_visible_given_exist_and_not_detected())

        return updated

    def get_updated_measurement(self, assign_probability, likelihood, measurement):

        # Update state distribution with measurement
        updated = self.duplicate()
        updated.weight = assign_probability * likelihood.association_weight * self.weight
        updated.measurement_history = self.measurement_history + [measurement]
        # Kalman update on state
        H = measurement.sensor.measurement_model.matrix()
        R = measurement.R
        y = measurement.coords
        updated.state_distribution, _ = kalman_update(updated.state_distribution, H, R, y)

        # Update colour distributions with measurement
        updated.colour_distribution.update(measurement, likelihood.colour_likelihood)

        # Set MMSI if measurement has one, otherwise we inherit the MMSI of the parent component
        if measurement.mmsi is not None:
            updated.mmsi = measurement.mmsi

        # Update existence probability
        updated.exist_probability = Probability(1.0)  # detected, so must exist

        # Update visibility probabilities
        prob_visible_given_exists = self.visibility_detection_prediction.probability_visible_given_exist_and_not_detected()
        prob_visible_given_exists[measurement.sensor.type] = Probability(1.0) # must be visible to sensor that detected it
        updated.visibility_given_existence = prob_visible_given_exists

        # Update measurement history
        updated.measurement_history = self.measurement_history + [measurement]

        return updated

    def to_merge(self, another, measurement_history_length : int):

        # Return true if we want to merge two components
        if self.mmsi != another.mmsi:
            return False # Keep different MMSIs separate
        if measurement_history_length == 0:
            return True
        else:
            return (self.measurement_history[-measurement_history_length:] ==
                another.measurement_history[-measurement_history_length:])

    @staticmethod
    def merge_components(components, measurement_history_length : int):

        new_component = components[0].duplicate()
        if measurement_history_length == 0:
            new_component.measurement_history = []
        else:
            new_component.measurement_history = new_component.measurement_history[-measurement_history_length:]

        # Merge weight and state distribution
        new_weight, new_state_distribution = merge_gaussians(
            list(zip([c.weight for c in components], [c.state_distribution for c in components])))
        new_component.weight = new_weight
        new_component.state_distribution = new_state_distribution

        # Set most likely MMSI
        mmsi_weights = {}
        for component in components:
            mmsi_weights[component.mmsi] = mmsi_weights.get(component.mmsi, Probability(0.0)) + \
                component.weight
        new_component.mmsi = max(mmsi_weights, key=lambda k: mmsi_weights[k])

        # Merge colour components
        new_component.colour_distribution = ColourDistribution.merge(
            [(component.weight, component.colour_distribution) for component in components])

        # Merge existence probability
        weight_sum = sum([component.weight for component in components])
        ew_sum = sum([component.weight * component.exist_probability for component in components])
        new_component.exist_probability = ew_sum / weight_sum

        # Merge visibility probability
        new_visibility_probability = {}
        for sensor_type in new_component.visibility_given_existence.keys():
            vw_sum = Probability(np.sum([component.weight * component.visibility_given_existence[sensor_type]
                             for component in components]))
            new_visibility_probability[sensor_type] = vw_sum / weight_sum
        new_component.visibility_given_existence = new_visibility_probability

        return new_component


class ELINTTrack:

    def __init__(self, track_id, state_distribution, colour_distribution, measurement,
                 mmsi, exist_prob, vis_given_exist, timestamp):

        self.id = track_id
        self.components = [ELINTTrackComponent(state_distribution, colour_distribution, [measurement],
                                               mmsi, exist_prob, vis_given_exist, Probability(1.0))]
        self.timestamp = timestamp

    def to_string(self):
        out = '{\n'
        out += '"id": ' + str(self.id) + ',\n'#,\n''
        out += '"components": [\n' + ',\n'.join([c.to_string() for c in self.components]) + '\n],\n'
        out += '"timestamp": "' + str(self.timestamp) + '"\n'
        out += '}'
        return out

    def mean_state(self):

        return sum([c.weight * c.state_distribution.mean for c in self.components])

    def get_existence_probability(self):

        return sum([c.weight * c.exist_probability for c in self.components])

    def mean_coords(self):

        mean_state = self.mean_state()
        return [mean_state[0], mean_state[2]]

    def get_mahalanobis_squared(self, measurement, transition_model, colours):

        mean_lonlat = self.mean_coords()
        dt = measurement.timestamp - self.timestamp
        A = transition_model.matrix(lonlat=mean_lonlat, time_interval=dt)
        Q = transition_model.covar(lonlat=mean_lonlat, time_interval=dt)
        H = measurement.sensor.measurement_model.matrix()
        R = measurement.R

        mahal_sq = np.inf

        for c in self.components:

            # Don't gate if the MMSI doesn't match
            if measurement.mmsi is None or c.mmsi is None or measurement.mmsi == c.mmsi:

                # Get squared mahalanobis distance according to position
                mahal_sq_state = get_mahalanobis_squared(measurement.coords, c.state_distribution.mean,
                                                         c.state_distribution.covar, A, Q, H, R)
                mahal_sq_colour = c.colour_distribution.get_mahalanobis_squared(measurement, dt)
                this_mahal_sq = mahal_sq_state + mahal_sq_colour
                if this_mahal_sq < mahal_sq:
                    mahal_sq = this_mahal_sq

        return mahal_sq

    def to_terminate(self, kill_probability_threshold):

        # Get the probability that the target exists and is visible to at least one sensor
        prob_sum = Probability(0.0)
        for component in self.components:
            this_prob_visible_given_exist = Probability(1) - \
                reduce(operator.mul, [1-p for p in component.visibility_given_existence.values()])
            this_prob_visible = component.exist_probability * this_prob_visible_given_exist
            prob_sum += component.weight * this_prob_visible

        return prob_sum < kill_probability_threshold

    def predict_existence_and_visibility(self, survival_prob, vis_det_trans):

        for c in self.components:
            c.predict_existence_and_visibility(survival_prob, vis_det_trans)

    def predict_state(self, transition_model, measurement, colours):

        dt = measurement.timestamp - self.timestamp

        mean_lonlat = self.mean_coords()
        A = transition_model.matrix(lonlat=mean_lonlat, time_interval=dt)
        Q = transition_model.covar(lonlat=mean_lonlat, time_interval=dt)

        # Predict target kinematics
        for c in self.components:
            c.state_distribution = GaussianState(A @ c.state_distribution.mean,
                                                 A @ c.state_distribution.covar @ A.transpose() + Q)
            # Predict colour information
            c.colour_distribution.predict(dt)

        self.timestamp = measurement.timestamp

    def get_likelihood(self, measurement, dt):

        # Get likelihood for each component
        likelihood = Probability(0.0)
        association_weight = Probability(0.0)
        #null_weight = Probability(1.0)
        component_likelihoods = {}

        for c in self.components:
            this_component_likelihood = c.get_likelihood(measurement, dt)
            component_likelihoods[c] = this_component_likelihood
            likelihood += this_component_likelihood.likelihood * c.weight
            association_weight += this_component_likelihood.association_weight * c.weight
            #null_weight += this_component_likelihood.null_weight * c.weight

        return TrackLikelihood(likelihood, association_weight, component_likelihoods)

    def get_null_hypothesis_weight(self):

            return sum([c.get_null_hypothesis_weight() * c.weight for c in self.components])

    def update_detected(self, measurement, assign_probability, likelihood):

        # Update undetected components
        new_components_not_detected = []
        tot_weight = Probability(0.0)
        for i, c in enumerate(self.components):

            new_components_not_detected.append(c.get_updated_missed_detection(assign_probability))
            tot_weight += new_components_not_detected[-1].weight

        for c in new_components_not_detected:
            c.weight *= (1 - assign_probability) / tot_weight

        # Update detected components
        new_components_detected = []
        tot_weight = Probability(0.0)
        for c in self.components:
            new_components_detected.append(c.get_updated_measurement(
                assign_probability, likelihood.component_likelihoods[c], measurement))
            tot_weight += new_components_detected[-1].weight

        for c in new_components_detected:
            c.weight *= assign_probability / tot_weight

        self.components = new_components_not_detected + new_components_detected

    def update_missed_measurement(self):

        new_components = []
        tot_weight = Probability(0.0)
        for i, c in enumerate(self.components):

            new_components.append(c.get_updated_missed_detection(Probability(0.0)))
            tot_weight += new_components[-1].weight

        for c in new_components:

            c.weight /= tot_weight

        self.components = new_components

    def mixture_reduce(self, measurement_history_length, component_probability_threshold):

        clusters = []
        for component in self.components:
            if component.weight > component_probability_threshold:
                merged = False
                for cluster in clusters:
                    if component.to_merge(another=cluster[0], measurement_history_length=measurement_history_length):
                        cluster.append(component)
                        merged = True
                        break
                if not merged:
                    clusters.append([component])
            else:
                pass

        new_components = [ELINTTrackComponent.merge_components(cluster, measurement_history_length)
                          for cluster in clusters]

        self.components = new_components

        # Normalise weights
        tot_weight = np.sum([component.weight for component in self.components])
        for component in self.components:
            component.weight /= tot_weight
