import numpy as np

from stonesoup.custom.types.tracklet import SensorTracks


# Create a class to represent a node in a fusion hierarchy,
class FusionNode:
    """
    Class to represent a node in a fusion hierarchy
    """
    def __init__(self, tracker, children, statedim, use_two_state_tracks=True):
        # Tracker used at this node
        self.tracker = tracker
        # Child nodes used to supply tracks for fusion
        self.children = children
        # The dimension of a (single) target state
        self.statedim = statedim
        # Whether to propagate two-state tracks
        self.use_two_state_tracks = use_two_state_tracks

    @property
    def tracks(self):
        return self.tracker.tracks

    @property
    def detections(self):
        if self.is_leaf():
            return self.tracker.detector.detections
        else:
            sensor_scans = [d for scan in self.tracker._scans for d in scan.sensor_scans]
            detections = set([d for sscan in sensor_scans for d in sscan.detections])
            return detections

    # Process the tracks at this node by reading in the child tracks, creating pseudomeasurements and performing
    # tracking
    def process_tracks(self, timestamp):
        child_tracks = [child.tracker.tracks for child in self.children]
        input_tracks = [SensorTracks(tracks, i) for i, tracks in enumerate(child_tracks)]
        return self.tracker.process_tracks(input_tracks, timestamp)

    def is_leaf(self):
        return len(self.children) == 0

    def get_leaf_trackers(self):
        if self.is_leaf():
            return [self.tracker]
        else:
            leaf_trackers = []
            for child in self.children:
                leaf_trackers += child.get_leaf_trackers()
        return leaf_trackers

    """
    # TODO: Try to make the fusion engine more pythonic with some class structure (under construction)
    def runTracker(self, timestamp):
        if not self.is_leaf:
            child_tracks = []
            for child in self.child_trackers:
                child_tracks.append(child.runTracker(timestamp))
        else:
            for child in self.child_trackers:
                child_tracks.append()
            child_
        two_state_tracks
    """

    def __iter__(self):
        if not self.is_leaf():
            self.children_iter = zip(*[child for child in self.children])
        else:
            self.children_iter = iter(self.tracker)
        return self

    def __next__(self):
        child_output = next(self.children_iter)
        if not self.is_leaf():
            timestamp = child_output[0][0]
            child_tracks = [out[1] for out in child_output]
            input_tracks = [SensorTracks(tracks, i) for i, tracks in enumerate(child_tracks)]
            return (timestamp, self.tracker.process_tracks(input_tracks, timestamp))
        else:
            return child_output

