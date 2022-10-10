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
        "--tracker_frame", help="The framework of tracker(DeepSORT, SORT, POI)",
        default="DeepSORT", type=str)
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
        "--lamda_max", help="STF Lamda Max.", type=float, default=1.5)
    parser.add_argument(
        "--weakening_factor", help="STF Beta.", type=float, default=10)
    parser.add_argument(
        "--result_file", help="Json format result.", default=None)
    parser.add_argument(
        "--noise_xy", help="Noise in pixel.", type=int, default=0)
    parser.add_argument(
        "--noise_s", help="Noise in % area.", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Set the evaluate metrics
    metrics = list(mm.metrics.motchallenge_metrics)
    acc = []
    names = []

    os.makedirs(args.output_dir, exist_ok=True)
    sequences = os.listdir(args.mot_dir)

    if args.result_file is not None: # Check if josonfiler lib is installed
      import jsonfiler
    
    import time
    t0 = time.time()
    # Loop all the MOT sequence
    for sequence in sequences:
        print("Running sequence %s" % sequence)
        sequence_dir = os.path.join(args.mot_dir, sequence)
        detection_file = os.path.join(args.detection_dir, "%s.npy" % sequence)
        output_file = os.path.join(args.output_dir, "%s.txt" % sequence)
        strong_sort.run(
            args.tracking_filter, args.tracker_frame, sequence_dir, detection_file,
            output_file, args.min_confidence, args.nms_max_overlap,
            args.min_detection_height, args.max_cosine_distance, args.nn_budget, display=False,
            rnn_model=args.rnn_model, lamda_max=args.lamda_max,
            weakening_factor=args.weakening_factor,
            add_noise=[args.noise_xy, args.noise_s])

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

    if args.result_file is not None:
      save_file = os.path.join(args.output_dir, args.result_file)
      if args.result_file in os.listdir(args.output_dir):
          save_dict = jsonfiler.load(save_file)
      else:
          save_dict = {}
      save_list = [round(summary.loc['OVERALL']['idf1']*100+0.000001, 1), round(summary.loc['OVERALL']['mota']*100+0.000001, 1), round(summary.loc['OVERALL']['motp']+0.0000001, 3), 
              int(summary.loc['OVERALL']['mostly_tracked']), int(summary.loc['OVERALL']['mostly_lost']), int(summary.loc['OVERALL']['num_switches']), 
              int(summary.loc['OVERALL']['num_fragmentations']), int(summary.loc['OVERALL']['num_misses'])]
      
      save_item = {"lamda_max" : args.lamda_max, 
            "weakening_factor" : args.weakening_factor,
            "noise_xy" : args.noise_xy,
            "noise_s" : args.noise_s,
            "performance": save_list}
      
      if args.tracking_filter not in save_dict:
        save_dict[args.tracking_filter] = []
      
      for item in save_dict[args.tracking_filter]:
        if item["lamda_max"] != save_item["lamda_max"]:
          continue
        if item["weakening_factor"] != save_item["weakening_factor"]:
          continue
        if item["noise_xy"] != save_item["noise_xy"]:
          continue
        if item["noise_s"] != save_item["noise_s"]:
          continue
        save_dict[args.tracking_filter].remove(item)
      save_dict[args.tracking_filter].append(save_item)
        
      jsonfiler.dump(save_dict, save_file, indent = 4)

      print(args.tracking_filter, args.lamda_max, args.weakening_factor, save_list)


