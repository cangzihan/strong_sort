# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import STF
from . import rnn
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3, tracker="Kalman", img_size='1080p',
                 tracker_frame="DeepSORT", parameter=None, add_noise=[25, 0.4]):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.tracker_type = tracker
        self.tracker_frame = tracker_frame
        self.add_noise = add_noise

        if tracker == "Kalman":
            self.kf = kalman_filter.KalmanFilter()
        elif tracker == "SEKF":
            self.kf = STF.StrongEKF(lamda_max=parameter[1], weakening_factor=parameter[2])
        elif tracker == "RNN":
            self.kf = rnn.RNN(img_size=img_size, model_path=parameter[0])
            print("Image Size:", self.kf.img_size)
        elif tracker == "LSTM":
            self.kf = rnn.RNN(img_size=img_size, model_path=parameter[0])
            print("Image Size:", self.kf.img_size)
        elif tracker == "GRU":
            self.kf = rnn.RNN(img_size=img_size, model_path=parameter[0])
            print("Image Size:", self.kf.img_size)
        else:
            raise Exception("Unknow Tracker")
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx], self.add_noise)
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        if self.tracker_frame in ["POI"]:
            active_targets = [t.track_id for t in self.tracks]
        else:
            active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed() and self.tracker_frame in ["DeepSORT", "SORT"]:
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        if self.tracker_frame in ["POI"]:
            confirmed_tracks = [i for i, t in enumerate(self.tracks)]
        else:
            confirmed_tracks = [
                i for i, t in enumerate(self.tracks) if t.is_confirmed()]
            unconfirmed_tracks = [
                i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        if self.tracker_frame in ["POI", "DeepSORT"]:
            # Associate confirmed tracks using appearance features.
            matches_a, unmatched_tracks_a, unmatched_detections = \
                linear_assignment.matching_cascade(
                    gated_metric, self.metric.matching_threshold, self.max_age,
                    self.tracks, detections, confirmed_tracks)
            if self.tracker_frame in ["DeepSORT"]:
              iou_track_candidates = unconfirmed_tracks + [
                  k for k in unmatched_tracks_a if
                  self.tracks[k].time_since_update == 1]
              unmatched_tracks_a = [
                  k for k in unmatched_tracks_a if
                self.tracks[k].time_since_update != 1]
        else:
            matches_a = []
            unmatched_tracks_a = []
            iou_track_candidates = list(range(len(self.tracks)))
            unmatched_detections = list(range(len(detections)))

        if self.tracker_frame in ["DeepSORT", "SORT"]:
            # Associate remaining tracks together with unconfirmed tracks using IOU.
            matches_b, unmatched_tracks_b, unmatched_detections = \
                linear_assignment.min_cost_matching(
                    iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                    detections, iou_track_candidates, unmatched_detections)
        else:
            matches_b = []
            unmatched_tracks_b = []

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        if self.tracker_type == "Kalman" or self.tracker_type == "SEKF":
            mean, covariance = self.kf.initiate(detection.to_xyah())
            self.tracks.append(Track(
                mean, covariance, self._next_id, self.n_init, self.max_age,
                detection.feature))
            self._next_id += 1
        elif self.tracker_type in ["RNN", "GRU", "LSTM"]:
            mean = self.kf.initiate(detection.to_xyah())
            self.tracks.append(Track(
                mean, None, self._next_id, self.n_init, self.max_age,
                detection.feature))
            self._next_id += 1
        else:
            raise Exception("Unknow Tracker")

