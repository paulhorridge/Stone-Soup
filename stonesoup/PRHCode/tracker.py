import numpy as np
from datetime import datetime, timedelta

from stonesoup.types.numeric import Probability
from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel, ConstantVelocity,
                                                NthDerivativeDecay)
from elinttrack import ELINTTrack, ColourComponent, SingleColourDistribution, ColourDistribution
from elintmeasurement import ELINTMeasurement

from functions import kalman_update, lonlat2metres


class LonLatTransitionModel:

    pass

class LonLatConstantVelocityTransitionModel(LonLatTransitionModel):

    def __init__(self, q_metres):

        self.q_metres = q_metres

    def matrix(self, lonlat, time_interval):

        lon2m, lat2m = lonlat2metres(lonlat[1])
        return CombinedLinearGaussianTransitionModel(
            [ConstantVelocity(self.q_metres / lon2m ** 2),
             ConstantVelocity(self.q_metres / lat2m ** 2)]).matrix(time_interval=time_interval)

    def covar(self, lonlat, time_interval):

        lon2m, lat2m = lonlat2metres(lonlat[1])
        return CombinedLinearGaussianTransitionModel(
            [ConstantVelocity(noise_diff_coeff = self.q_metres / lon2m ** 2),
             ConstantVelocity(noise_diff_coeff = self.q_metres / lat2m ** 2)]).covar(time_interval=time_interval)


class LonLatOrnsteinUhlenbeckTransitionModel(LonLatTransitionModel):

    def __init__(self, q_metres, K):

        self.q_metres = q_metres
        self.K = K

    def matrix(self, lonlat, time_interval):

        lon2m, lat2m = lonlat2metres(lonlat[1])
        model = CombinedLinearGaussianTransitionModel(
            [NthDerivativeDecay(decay_derivative = 1, noise_diff_coeff = self.q_metres / lon2m ** 2,
                                damping_coeff = self.K),
             NthDerivativeDecay(decay_derivative = 1, noise_diff_coeff = self.q_metres / lat2m ** 2,
                                damping_coeff = self.K)])
        return model.matrix(time_interval=time_interval)

    def covar(self, lonlat, time_interval):

        lon2m, lat2m = lonlat2metres(lonlat[1])
        model = CombinedLinearGaussianTransitionModel(
            [NthDerivativeDecay(decay_derivative = 1, noise_diff_coeff = self.q_metres / lon2m ** 2,
                                damping_coeff = self.K),
             NthDerivativeDecay(decay_derivative = 1, noise_diff_coeff = self.q_metres / lat2m ** 2,
                                damping_coeff = self.K)])
        return model.covar(time_interval=time_interval)


class MixtureReducer:

    def __init__(self, measurement_history_length, component_probability_threshold):

        self.measurement_history_length = measurement_history_length
        self.component_probability_threshold = component_probability_threshold

    def reduce(self, track):

        track.mixture_reduce(self.measurement_history_length, self.component_probability_threshold)


class TrackDeleter:

    def __init__(self, kill_probability_threshold):

        self.kill_probability_threshold = kill_probability_threshold

    def to_terminate(self, track):

        return track.to_terminate(self.kill_probability_threshold)


class GaterELINT:

    def __init__(self, gate_sd):

        self.gate_sd = gate_sd

    def gate_tracks(self, tracks, measurement, transition_model, colours):

        gated_tracks = set()
        for track in tracks:

            mahal_sq = track.get_mahalanobis_squared(measurement, transition_model, colours)
            if mahal_sq <= self.gate_sd ** 2:
                gated_tracks.add(track)

        return gated_tracks


class ELINTTracker:

    def __init__(self, prior, gater, mixture_reducer, deleter, mean_track_life,
                 sensors, transition_model, colours):

        self.tracks = []
        self.prior = prior
        self.gater = gater
        self.mixture_reducer = mixture_reducer
        self.deleter = deleter
        self.mean_track_life = mean_track_life
        self.sensors = sensors
        self.transition_model = transition_model
        self.colours = colours
        self.max_track_id = -1
        self.current_time = None

    def iterate(self, measurement : ELINTMeasurement):

        # Get time delta from last measurement
        if self.current_time is None:
            dt = timedelta(seconds=0)
        else:
            dt = measurement.timestamp - self.current_time
        self.current_time = measurement.timestamp

        # Gate tracks
        gated_tracks = self.gate_tracks(measurement)

        # Print status information
        print(
            str(measurement.timestamp) + ": " + measurement.type, " " + str(len(self.tracks)) + " track(s) (" +
            str(len(gated_tracks)) + " gated, max_track_id = " + str(self.max_track_id) + ")")

        # Get weight of null hypothesis (measurement unassigned so new track)
        new_track_weight = self.prior.get_null_weight(measurement)

        # Predict the track existence and visibility
        self.predict_existence_and_visibility(dt)

        # Predict states and colours for gated tracks
        for track in gated_tracks:
            track.predict_state(self.transition_model, measurement, self.colours)

        # Get likelihood values for measurement and gated tracks
        likelihoods = {track: track.get_likelihood(measurement, dt)
                       for track in gated_tracks}

        # Get assignment weights for tracks and normalise to probabilities
        assign_probabilities = {track: likelihoods[track].association_weight / track.get_null_hypothesis_weight()
                                for track in gated_tracks}
        tot_weight = sum([p for p in assign_probabilities.values()]) + new_track_weight
        new_track_weight /= tot_weight
        for track in assign_probabilities.keys():
            assign_probabilities[track] = assign_probabilities[track] / tot_weight

        # Update the tracks
        for track in self.tracks:
            if track in gated_tracks and assign_probabilities[track] > 0:
                # Update track based on measurement
                track.update_detected(measurement, assign_probabilities[track], likelihoods[track])
            else:
                # Update track based on missed measurement
                track.update_missed_measurement()

        # Mixture reduction on gated tracks
        for track in gated_tracks:
            self.mixture_reducer.reduce(track)

        # Decide whether to create new track
        tracks_to_output = gated_tracks
        if len(gated_tracks) == 0 or new_track_weight > max(assign_probabilities.values()):
            # create new track
            self.create_new_track(new_track_weight, measurement)
            tracks_to_output.add(self.tracks[-1])

        # Kill tracks with low existence/visibility probs
        for track in self.tracks:
            if self.deleter.to_terminate(track):
                self.tracks.remove(track)

        return tracks_to_output

    def gate_tracks(self, measurement):

        return self.gater.gate_tracks(self.tracks, measurement,
                                      self.transition_model, self.colours)

    def predict_existence_and_visibility(self, dt):

        survival_prob = Probability(-dt / self.mean_track_life, log_value=True)
        vis_det_trans = {name: sensor.get_visibility_detection_transition(dt) for name, sensor in self.sensors.items()}

        for track in self.tracks:
            track.predict_existence_and_visibility(survival_prob, vis_det_trans)

    def create_new_track(self, exist_prob, measurement):

        state_distribution = self.prior.get_initial_state_prior(measurement.coords)
        state_distribution, _ = kalman_update(state_distribution, measurement.sensor.measurement_model.matrix(),
                                              measurement.R, measurement.coords)

        # Get colour components
        single_colour_distributions = {}
        for name, data in self.prior.colours.items():
            colour_measurement = measurement.colours.get(name, None)
            colour_dist = data.prior
            if colour_measurement is not None:
                colour_dist, _ = kalman_update(colour_dist, np.array([[1.0]]), data.R, colour_measurement)
            if data.is_switch:
                single_colour_distributions[name] = SingleColourDistribution({
                    0: ColourComponent(colour_dist, 1 - data.prior_switch_prob),
                    1: ColourComponent(colour_dist, data.prior_switch_prob)},
                    data
                )
            else:
                single_colour_distributions[name] = SingleColourDistribution(
                    {0: ColourComponent(colour_dist)}, data)
        colour_distribution = ColourDistribution(single_colour_distributions)

        mmsi = measurement.mmsi

        vis_given_existence = {sensor_type: probability for sensor_type, probability in
                                      self.prior.visibility_given_existence.items()}
        vis_given_existence[measurement.type] = Probability(1.0)

        self.max_track_id += 1
        self.tracks.append(ELINTTrack(self.max_track_id,
            state_distribution, colour_distribution, measurement, mmsi,
            exist_prob, vis_given_existence, measurement.timestamp))
