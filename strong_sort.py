from __future__ import division, print_function, absolute_import

import os
import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from tracker import nn_matching
from tracker.detection import Detection
from tracker.tracker import Tracker


def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(tracking_filter, tracker_frame, sequence_dir, detection_file, output_file,
        min_confidence, nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display, rnn_model, lamda_max, weakening_factor, add_noise):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    tracking_filter : str
        The filter of tracker(Kalman, SEKF, RNN, LSTM or GRU).
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file(.npy).
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.
    rnn_model : Path of saved RNN model. 
        This item is required  if the tracking_filter is RNN, LSTM or GRU
    lamda_max : STF Lamda Max.
    weakening_factor : STF Beta.
    """
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    if "MOT16-05" in sequence_dir:
      tracker = Tracker(metric, tracker=tracking_filter, img_size='VGA',
                        parameter=[rnn_model, lamda_max, weakening_factor],
                        tracker_frame=tracker_frame, add_noise=add_noise)
    elif "MOT20-03" in sequence_dir:
      tracker = Tracker(metric, tracker=tracking_filter, img_size=(1173, 880),
                        parameter=[rnn_model, lamda_max, weakening_factor],
                        tracker_frame=tracker_frame, add_noise=add_noise)
    elif "MOT20-05" in sequence_dir:
      tracker = Tracker(metric, tracker=tracking_filter, img_size=(1654, 1080),
                        parameter=[rnn_model, lamda_max, weakening_factor],
                        tracker_frame=tracker_frame, add_noise=add_noise)
    elif "MOT20-04" in sequence_dir:
      tracker = Tracker(metric, tracker=tracking_filter, img_size=(1545, 1080),
                        parameter=[rnn_model, lamda_max, weakening_factor],
                        tracker_frame=tracker_frame, add_noise=add_noise)
    elif "MOT20-06" in sequence_dir:
      tracker = Tracker(metric, tracker=tracking_filter, img_size=(1920, 734),
                        parameter=[rnn_model, lamda_max, weakening_factor],
                        tracker_frame=tracker_frame, add_noise=add_noise)
    elif "MOT20-08" in sequence_dir:
      tracker = Tracker(metric, tracker=tracking_filter, img_size=(1920, 734),
                        parameter=[rnn_model, lamda_max, weakening_factor],
                        tracker_frame=tracker_frame, add_noise=add_noise)
    elif "MOT16-06" in sequence_dir:
      tracker = Tracker(metric, tracker=tracking_filter, img_size='VGA',
                        parameter=[rnn_model, lamda_max, weakening_factor],
                        tracker_frame=tracker_frame, add_noise=add_noise)
    else:
      tracker = Tracker(metric, tracker=tracking_filter, img_size='1080p',
                        parameter=[rnn_model, lamda_max, weakening_factor],
                        tracker_frame=tracker_frame, add_noise=add_noise)
    results = []

    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


if __name__ == "__main__":
    pass
