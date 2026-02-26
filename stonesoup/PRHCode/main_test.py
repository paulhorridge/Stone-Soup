import numpy as np
from datetime import datetime, timedelta, timezone
import json
import time

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.numeric import Probability

from tracker import LonLatConstantVelocityTransitionModel, LonLatOrnsteinUhlenbeckTransitionModel
from tracker import ELINTTracker, MixtureReducer, TrackDeleter, GaterELINT
from elintmeasurement import ELINTColour, ELINTMeasurement, SensorELINT
from priorelint import PriorELINT

from pymongo import MongoClient

class SensorMeasurementReader:

    pass

class SensorMeasurementReaderList(SensorMeasurementReader):
    """
    Read data for a single sensor from a supplied list of measurements
    """
    def __init__(self, measurement_data):

        self.measurement_data = measurement_data
        self.line_number = 0 # use iterators later?

    def next_measurement_timestamp(self):

        if self.line_number < len(self.measurement_data):
            return self.measurement_data[self.line_number].timestamp
        else:
            return None

    def at_end(self):

        return self.line_number >= len(self.measurement_data)

    def get_next(self):

        if self.line_number < len(self.measurement_data):
            self.line_number += 1
            return self.measurement_data[self.line_number-1]
        else:
            raise StopIteration


class SensorMeasurementReaderMongoDB(SensorMeasurementReader):
    """
    Read data for a single sensor from a database (load in batches of specified size)
    """
    def __init__(self, mongodb_collection, sensor, colours, start_time, end_time, batch_duration):

        self.mongodb_collection = mongodb_collection
        self.sensor = sensor
        self.colours = colours
        self.batch_end_time = start_time
        self.batch_duration = batch_duration
        self.end_time = end_time
        self.current_batch = None
        self.current_batch_line_number = None

    def _load_new_batch(self):

        min_time = self.batch_end_time
        max_time = min(self.batch_end_time + self.batch_duration, self.end_time)
        batch_data = self.mongodb_collection.find({"ptime": {"$gte": min_time.timestamp(),
                                                             "$lt": max_time.timestamp()}}, sort=[("ptime", 1)])

        self.current_batch = [ELINTMeasurement(m, self.sensor, self.sensor.type, self.colours) for m in batch_data]
        self.current_batch_line_number = 0
        self.batch_end_time = max_time

    def next_measurement_timestamp(self):

        if self.current_batch is None or self.current_batch_line_number >= len(self.current_batch):
            done = False
            while not done:
                self._load_new_batch()
                done = len(self.current_batch) > 0 or self.batch_end_time >= self.end_time

        if self.current_batch_line_number < len(self.current_batch):
            return self.current_batch[self.current_batch_line_number].timestamp
        else:
            return None

    def at_end(self):

        return self.next_measurement_timestamp() is None

    def get_next(self):

        if self.at_end():
            return None
        else:
            self.current_batch_line_number += 1
            return self.current_batch[self.current_batch_line_number-1]


class MeasurementReader:
    """
    Read data for multiple sensors from a file
    """
    def __init__(self, sensor_readers):

        self.sensor_readers = sensor_readers

    def __iter__(self):

        return self

    def __next__(self):

        # Find the next sensor to read based on which measurement is next
        next_time = None
        next_sensor_type = None
        for this_sensor_type, this_sensor_data in self.sensor_readers.items():
            this_next_time = self.sensor_readers[this_sensor_type].next_measurement_timestamp()
            if this_next_time is not None:
                if next_time is None or this_next_time < next_time:
                    next_time = this_next_time
                    next_sensor_type = this_sensor_type

        # If measurements to read, return it, otherwise raise StopIteration
        if next_time is not None:
            return self.sensor_readers[next_sensor_type].get_next()
        else:
            raise StopIteration

    def at_end(self):

        for i in self.sensor_readers.values():
            if not i.at_end():
                return False
        return True


def main():

    #------
    print(type(Probability(0) + Probability(1)))
    print(type(Probability(0) + Probability(0)))
    print(type(Probability(0.5) + Probability(0.5)))
    #------

    np.random.seed(1991)

    meas_filename = "Data/measdata_10target_q.json"
    trackout_filename = "Data/trackout_10target_q.json"

    # Read colour data
    f = open("Data/colours_q.json", "r")
    colour_data = json.load(f)
    f.close()
    colours = {c["name"] : ELINTColour(c) for c in colour_data}

    # Construct measurement model
    measurement_model = LinearGaussian(
        ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
        mapping=(0, 2),  # Mapping measurement vector index to state index
        noise_covar=np.eye(2)) # noise covariance is a dummy since we have measurement-specific errors

    # Construct sensors
    sensors = {"AIS":       SensorELINT("AIS",       measurement_model, timedelta(hours=1.), timedelta(days=1.), timedelta(days=2.)),
               "ELINT3GHz": SensorELINT("ELINT3GHz", measurement_model, timedelta(hours=1.1), timedelta(days=1.1), timedelta(days=2.1)),
               "ELINT9GHz": SensorELINT("ELINT9GHz", measurement_model, timedelta(hours=1.2), timedelta(days=1.2), timedelta(days=2.2))}
    mean_track_life = timedelta(days=4.)

    # Read data from file
    f = open(meas_filename, "r")
    measurement_data = json.load(f)
    f.close()
    sensor_readers = {}
    for sensor_type, sensor_data in measurement_data.items():
        this_measurement_list = sorted([ELINTMeasurement(m, sensors[sensor_type], sensor_type, colours) for m
                                        in sensor_data], key=lambda m: m.timestamp)
        sensor_readers[sensor_type] = SensorMeasurementReaderList(this_measurement_list)
    measurement_reader = MeasurementReader(sensor_readers)

    # start_time = datetime(2020, 10, 22, 0, 0, 0, tzinfo=timezone.utc)
    # end_time = start_time + timedelta(days=1)
    # batch_duration = timedelta(hours=1)
    # mongodb_server = "localhost"
    # mongodb_port = 27017
    # mongodb_dbname = "test_database"
    # mongodb_client = MongoClient(f"mongodb://{mongodb_server}:{mongodb_port}/")
    # mongodb_collections = {sensor_type: mongodb_client[mongodb_dbname]["pythontest_" + sensor_type]
    #                        for sensor_type in sensors.keys()}
    # sensor_readers = {sensor_type : SensorMeasurementReaderMongoDB(collection, sensors[sensor_type], colours,
    #                                                                start_time, end_time, batch_duration)
    #                   for sensor_type, collection in mongodb_collections.items()}
    # measurement_reader = MeasurementReader(sensor_readers)

    # Set up prior
    pos_min = np.array([-84, 34])
    pos_max = np.array([9, 62])
    prior_speed_sd_mps = 10
    vis_given_exist = {"AIS": Probability(0.5), "ELINT3GHz": Probability(0.4), "ELINT9GHz": Probability(0.3)}
    mmsi_prior_pdf = {"AIS": Probability(1.e-5)}
    expected_number_unconfirmed_targets = 1
    prior = PriorELINT(pos_min, pos_max, prior_speed_sd_mps, colours, vis_given_exist,
                       mmsi_prior_pdf,expected_number_unconfirmed_targets)

    # Construct transition model
    q_metres = 0.1
    stationary_velocity_mps = prior_speed_sd_mps
    damping_factor = q_metres / (2.0 * stationary_velocity_mps**2)
    transition_model = LonLatOrnsteinUhlenbeckTransitionModel(q_metres, damping_factor)
    #transition_model = LonLatConstantVelocityTransitionModel(q_metres)

    # Construct gater, mixture reducer and track deleter
    gate_sd = 5.
    gater = GaterELINT(gate_sd)
    measurement_history_length = 1
    component_probability_threshold = Probability(1e-3)
    mixture_reducer = MixtureReducer(measurement_history_length, component_probability_threshold)
    kill_probability_threshold = Probability(0.1)
    deleter = TrackDeleter(kill_probability_threshold)

    # Construct tracker
    tracker = ELINTTracker(prior, gater, mixture_reducer, deleter, mean_track_life,
                           sensors, transition_model, colours)

    track_outfile = open(trackout_filename, "w")
    track_outfile.write("[\n")

    #max_num_meas = len(measurements_list)

    start_clock = time.time()

    #for meas_num, measurement in enumerate(measurements_list[:max_num_meas]):
    for measurement in measurement_reader:

        #print(str(meas_num+1) + " / " + str(max_num_meas) + ": ", end="")
        tracks_to_output = tracker.iterate(measurement)

        # Output track data
        track_outfile.write('{\n"timestamp": "' + str(measurement.timestamp) + '",\n"tracks":\n[\n' + \
            ",\n".join([track.to_string() for track in tracks_to_output]) + '\n]\n}')
        #track_outfile.write(',\n' if  meas_num < len(measurements_list) - 1 and meas_num < max_num_meas - 1 else '\n')
        track_outfile.write('\n' if measurement_reader.at_end() else ',\n')

    track_outfile.write(']\n')
    track_outfile.close()

    end_clock = time.time()
    print(str(end_clock - start_clock) + " seconds")


main()

# # Read measurement data
# f = open(meas_filename, "r")
# meas_data = json.load(f)
# measurements = {sensor_type : [ELINTMeasurement(m, sensors[sensor_type], sensor_type, colours) for m in mlist]
#                 for sensor_type, mlist in meas_data.items()}
# f.close()
# # Get measurements in time-sorted order
# measurements_list = []
# for mlist in measurements.values():
#     for m in mlist:
#         measurements_list.append(m)
# measurements_list.sort(key=lambda m: m.timestamp)