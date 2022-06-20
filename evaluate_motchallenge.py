import motmetrics as mm
import argparse
import os
import strong_sort


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="MOTChallenge evaluation")
    parser.add_argument(
        "--tracking_filter", help="The filter of tracker(Kalman, SEKF, RNN, LSTM or GRU)",
        default="Kalman", type=str)
    parser.add_argument(
        "--mot_dir", help="Store sequences of MOT dataset(MOT16-2, MOT16-9, .....)",
        required=True)
    parser.add_argument(
        "--output_dir", help="Store output files of object tracking. Will "
                             "be created if it does not exist.", default="results")
    parser.add_argument(
        "--detection_dir", help="Store .npy files of detections.", default="detections",
        required=True)
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
                                 "all detections that have a confidence lower than this value.",
        default=0.0, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
                                       "box height. Detections with height smaller than this value are "
                                       "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap", help="Non-maxima suppression threshold: Maximum "
                                  "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
                                      "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
                            "gallery. If None, no budget is enforced.", type=int, default=100)
    parser.add_argument(
        "--rnn_model", help="Path of saved RNN model. This item is required "
                            "if the tracking_filter is RNN, LSTM or GRU", default=None)
    parser.add_argument(
        "--lamda_max", help="STF Lamda Max.", type=int, default=1.5)
    parser.add_argument(
        "--weakening_factor", help="STF Beta.", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Set the evaluate metrics
    metrics = list(mm.metrics.motchallenge_metrics)
    acc = []
    names = []

    os.makedirs(args.output_dir, exist_ok=True)
    sequences = os.listdir(args.mot_dir)

    import time
    t0 = time.time()
    # Loop all the MOT sequence
    for sequence in sequences:
        print("Running sequence %s" % sequence)
        sequence_dir = os.path.join(args.mot_dir, sequence)
        detection_file = os.path.join(args.detection_dir, "%s.npy" % sequence)
        output_file = os.path.join(args.output_dir, "%s.txt" % sequence)
        strong_sort.run(
            args.tracking_filter, sequence_dir, detection_file, output_file, args.min_confidence,
            args.nms_max_overlap, args.min_detection_height,
            args.max_cosine_distance, args.nn_budget, display=False,
            rnn_model=args.rnn_model, lamda_max=args.lamda_max,
            weakening_factor=args.weakening_factor)

        # load gt and ts files
        gt_file = os.path.join(args.mot_dir, "%s/gt/gt.txt" % sequence)
        print("gt_file:", gt_file)
        print("ts_file:", output_file)
        gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
        ts = mm.io.loadtxt(output_file, fmt="mot15-2D")

        name = os.path.splitext(os.path.basename(output_file))[0]
        names.append(name)
        print()

        # Calculate acc
        acc.append(mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5))

    t = time.time() - t0
    print("Time:", t)

    mh = mm.metrics.create()
    summary = mh.compute_many(acc, metrics=metrics, names=names, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

    # Save the sheet
    sheet_path = os.path.join(args.output_dir, "Evaluate_result.xls")
    summary.to_excel(sheet_path, "result")
