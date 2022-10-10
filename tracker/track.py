# vim: expandtab:ts=4:sw=4
import random

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.mean_before = mean
        self.covariance_before = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        random.seed(10000)

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, tracking_filter):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        if "RNN" in str(type(tracking_filter)):
          self.mean_before = self.mean
          self.covariance_before = self.covariance
          self.mean = tracking_filter.predict(self.mean)
          self.age += 1
          self.time_since_update += 1
        else:
          self.mean_before = self.mean
          self.covariance_before = self.covariance
          self.mean, self.covariance = tracking_filter.predict(self.mean, self.covariance)
          self.age += 1
          self.time_since_update += 1
        

    def update(self, tracking_filter, detection, add_noise=[25, 0.4]):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        tracking_filter :
            Kalman filter or SEKF.
        detection : Detection
            The associated detection.

        """

        if "StrongEKF" in str(type(tracking_filter)):
            self.mean_before[0] = round(random.choice((0, random.random() * 2 - 1)) * add_noise[0]) + self.mean_before[0]
            self.mean_before[0] = max(0, self.mean_before[0])
            self.mean_before[1] = round(random.choice((0, random.random() * 2 - 1)) * add_noise[0]) + self.mean_before[1]
            self.mean_before[1] = max(0, self.mean_before[1])
            self.mean_before[3] = round((random.choice((0, random.random() * add_noise[1]))) * self.mean_before[3]) + self.mean_before[3]

            self.mean, self.covariance = tracking_filter.update(
                self.mean_before, self.covariance_before, detection.to_xyah())
        elif "KalmanFilter" in str(type(tracking_filter)):
            self.mean[0] = round(random.choice((0, random.random() * 2 - 1)) * add_noise[0]) + self.mean[0]
            self.mean[0] = max(0, self.mean[0])
            self.mean[1] = round(random.choice((0, random.random() * 2 - 1)) * add_noise[0]) + self.mean[1]
            self.mean[1] = max(0, self.mean[1])
            self.mean[3] = round((random.choice((0, random.random() * add_noise[1]))) * self.mean_before[3]) + self.mean[3]
            self.mean, self.covariance = tracking_filter.update(
                self.mean, self.covariance, detection.to_xyah())
        elif "RNN" in str(type(tracking_filter)):
            self.mean = tracking_filter.update(
                self.mean_before, detection.to_xyah())
        else:
          raise Exception("Unknow Tracker")

        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
