from abc import abstractmethod
from operator import attrgetter
from datetime import datetime
from statistics import mean

from stonesoup.dataassociator.base import Associator
from stonesoup.types.association import AssociationSet
from typing import Set, List, Union, Tuple, Dict, Optional
import numpy as np
from stonesoup.base import Property, Base
from stonesoup.measures import GenericMeasure, Measure
from stonesoup.dataassociator._assignment import assign2D
from stonesoup.types.association import Association
from stonesoup.types.track import Track
from stonesoup.types.state import State


class GeneralAssociationGate(Base):

    @abstractmethod
    def __call__(self, item1, item2) -> bool:
        raise NotImplementedError


class MeasureThresholdGate(GeneralAssociationGate):

    minimise_measure: bool = Property(default=True)
    association_threshold: float = Property()
    measure: GenericMeasure = Property()

    def __call__(self, item1, item2) -> bool:
        distance_measure = self.measure(item1, item2)
        if self.minimise_measure:
            return distance_measure <= self.association_threshold
        else:  # maximise measure
            return self.association_threshold <= distance_measure


class RecentTrackMeasure(GenericMeasure):
    state_measure: Measure = Property()
    n_states_to_compare: int = Property(default=10)

    def __call__(self, track1: Track, track2: Track) -> Optional[float]:

        track_1_dict: Dict[datetime, State] = \
            {state.timestamp: state for i, state in enumerate(reversed(track1.states))
             if i < self.n_states_to_compare}

        track_2_dict = {state.timestamp: state for i, state in enumerate(reversed(track2.states)) if
                        i < self.n_states_to_compare}

        all_times = set(track_1_dict.keys()) | set(track_2_dict.keys())

        measures = []
        for time in all_times:
            state1 = track_1_dict.get(time)
            state2 = track_2_dict.get(time)

            if state1 is not None and state2 is not None:
                measures.append(self.state_measure(state1, state2))

        if len(measures) != 0:
            return mean(measures)
        else:
            return None


class OneToOneAssociatorWithGates(Associator):

    gates: List[GeneralAssociationGate] = Property(default=None)

    fail_gate_value: Union[int, float, complex, np.number] = Property(default=None)

    measure: GenericMeasure = Property()
    association_threshold: float = Property(default=None)

    maximise_measure: bool = Property(default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.gates is None:
            self.gates = []

        if self.fail_gate_value is None:
            if self.maximise_measure:
                self.fail_gate_value = 0
            else:
                self.fail_gate_value = 1e6

        if self.association_threshold is None:
            if self.maximise_measure:
                self.association_threshold = 0
            else:
                self.association_threshold = 1e6

    def associate(self, objects_a: Set, objects_b: Set) \
            -> Tuple[AssociationSet, Set, Set]:
        """Associate two sets of tracks together.

        Parameters
        ----------
        objects_a : set of :class:`~.Track` objects
            Tracks to associate to track set 2
        objects_b : set of :class:`~.Track` objects
            Tracks to associate to track set 1

        Returns
        -------
        AssociationSet
            Contains a set of :class:`~.Association` objects

        """
        distance_matrix = np.empty((len(objects_a), len(objects_b)))

        list_of_as = list(objects_a)
        list_of_bs = list(objects_b)

        for i, a in enumerate(list_of_as):
            for j, b in enumerate(list_of_bs):
                distance_matrix[i, j] = self.individual_weighting(a, b)

        distance_matrix2 = np.copy(distance_matrix)

        # Use "shortest path" assignment algorithm on distance matrix
        # to assign tracks to nearest detection
        # Maximise flag = true for probability instance
        # (converts minimisation problem to maximisation problem)
        gain, col4row, row4col = assign2D(
            distance_matrix2, self.maximise_measure)

        # Ensure the problem was feasible
        if gain.size <= 0:
            raise RuntimeError("Assignment was not feasible")

        # Create dictionary for associations
        associations = AssociationSet()

        # Generate dict of key/value pairs
        for i, object_a in enumerate(list_of_as):
            index_of_objects_b = col4row[i]
            if index_of_objects_b == -1:
                continue
            value = distance_matrix[i, index_of_objects_b]

            if self.maximise_measure:
                if value < self.association_threshold:
                    continue
            else:  # Minimise measure
                if value > self.association_threshold:
                    continue

            associations.associations.add(Association({object_a, list_of_bs[index_of_objects_b]}))

        associated_all = {thing for assoc in associations.associations for thing in assoc.objects}

        unassociated_a = set(objects_a) - associated_all
        unassociated_b = set(objects_b) - associated_all

        return associations, unassociated_a, unassociated_b

    def individual_weighting(self, a, b):

        for gate in self.gates:
            if not gate(a, b):
                return self.fail_gate_value

        measure_output = self.measure(a, b)
        if measure_output is None:
            return self.fail_gate_value
        else:
            return measure_output







