import os
import jsonfiler
import argparse

show_performance = 3
performance_dict = ["IDF1", "MOTA", "MOTP", "MT", "ML", "ID Switch", "FM", "FN"]
# Display sequence
tracking_filter_seq = ("Kalman", "SEKF", "STF")

new_dict = {}  # New dict for replacing the old one
plot_dict = {}  # beta: [[lambda_max1, lambda_max2,...], [y1, y2, ...]]


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="MOTChallenge evaluation")
    parser.add_argument(
        "--result_dir", help="Store output files of object tracking. Will "
            "be created if it does not exist.", default="results")
    return parser.parse_args()


def read_type1_file(result_dict, show=False):
  # Print the first row
  print("\t         | IDF1  MOTA   MOTP  MT  ML ID Switch  FM   FN")

  for tracking_filter in tracking_filter_seq:
      if tracking_filter not in result_dict:
          continue
      lambda_max_list = list(map(float, result_dict[tracking_filter].keys()))
      lambda_max_list.sort()
      lambda_max_list = list(map(str, lambda_max_list))
      for lambda_max in lambda_max_list:
          beta_list = list(result_dict[tracking_filter][lambda_max].keys())
          beta_list.sort()
          for beta in beta_list:
              # prepare new dict
              if tracking_filter not in new_dict:
                  new_dict[tracking_filter] = {}
              if lambda_max not in new_dict[tracking_filter]:
                  new_dict[tracking_filter][lambda_max] = {}
              new_dict[tracking_filter][lambda_max][beta] = result_dict[tracking_filter][lambda_max][beta]
              # prepare plot dict
              if beta not in plot_dict:
                  plot_dict[beta] = [[], []]
              if float(lambda_max) <= 2.5:
                  plot_dict[beta][0].append(float(lambda_max))
                  plot_dict[beta][1].append(result_dict[tracking_filter][lambda_max][beta][show_performance])
              # Print result table
              print("%6s %4s %4s " % (tracking_filter, lambda_max, beta), end = "|")
              print("%.1f" % (result_dict[tracking_filter][lambda_max][beta][0]), "%", end = " ")
              print("%.1f" % (result_dict[tracking_filter][lambda_max][beta][1]), "%", end = " ")
              print("%.3f" % (result_dict[tracking_filter][lambda_max][beta][2]), end = " ")
              print("%d" % (result_dict[tracking_filter][lambda_max][beta][3]), end = " ")
              print("%d" % (result_dict[tracking_filter][lambda_max][beta][4]), end = " ")
              print("   %d   " % (result_dict[tracking_filter][lambda_max][beta][5]), end = " ")
              print("%d" % (result_dict[tracking_filter][lambda_max][beta][6]), end = " ")
              print("%d" % (result_dict[tracking_filter][lambda_max][beta][7]))
      print('-'*80)

  if os.name == 'nt' or show:
      import matplotlib.pyplot as plt
      fig = plt.figure(figsize=(8, 6))
      ax1 = fig.add_subplot(1, 1, 1)

      for beta in plot_dict:
          if 1.0 in plot_dict[beta][0]:
              ax1.hlines(plot_dict[beta][1][plot_dict[beta][0].index(1.0)],
                        min(plot_dict[beta][0][:]),
                        max(plot_dict[beta][0][:]), colors='r', linestyles='dashed')
          ax1.plot(plot_dict[beta][0][:], plot_dict[beta][1][:], label="Weakening factor:"+beta)
      ax1.legend(loc='upper left')
      ax1.set_xlabel("fading factor max")
      ax1.set_ylabel(performance_dict[show_performance])
      plt.title("OC-SORT with Strong KF")
      plt.show()
  return new_dict


def read_type2_file(result_dict, show=False):
  # Print the first row
  print("       lamda_max beta noise_xy noise_s | IDF1  MOTA   MOTP  MT  ML ID Switch  FM   FN")
  new_dict = result_dict
  
  for tracking_filter in tracking_filter_seq:
      if tracking_filter not in result_dict:
          continue

      noise_s_list = []
      noise_xy_list = []
      for item in result_dict[tracking_filter]:
        noise_s = item["noise_s"]
        noise_xy = item["noise_xy"]
        noise_s_list.append(noise_s)
        noise_xy_list.append(noise_xy)
      noise_s_list = list(set(noise_s_list))
      noise_xy_list = list(set(noise_xy_list))
      noise_s_list.sort()
      noise_xy_list.sort()
      
      sorted_items_1 = []
      for noise_s in noise_s_list:
        for item in result_dict[tracking_filter]:
          if item["noise_s"] == noise_s and item not in sorted_items_1:
            sorted_items_1.append(item)
      
      sorted_items_2 = []
      for noise_xy in noise_xy_list:
        for item in sorted_items_1:
          if item["noise_xy"] == noise_xy and item not in sorted_items_2:
            sorted_items_2.append(item)

      for item in sorted_items_2:
        lambda_max = item["lamda_max"]
        beta = item["weakening_factor"]
        noise_xy = item["noise_xy"]
        noise_s = item["noise_s"]
        performance = item["performance"]
        # Print result table
        print("%6s\t%.1f\t %.1f \t%d\t %.1f   " % (tracking_filter, lambda_max, beta, noise_xy, noise_s), end = "|")
        print("%.1f" % (performance[0]), "%", end = " ")
        print("%.1f" % (performance[1]), "%", end = " ")
        print("%.3f" % (performance[2]), end = " ")
        print("%d" % (performance[3]), end = " ")
        print("%d" % (performance[4]), end = " ")
        print("   %d   " % (performance[5]), end = " ")
        print("%d" % (performance[6]), end = " ")
        print("%d" % (performance[7]))

      print('-'*88)

  if os.name == 'nt' or show:
      print("show_fig")

  return result_dict

def read_result_file(result_path, show=False):
  # Read json file
  print("Read from:", result_path)
  result_dict = jsonfiler.load(result_path)

  if len(result_dict) == 0:
    print("Empty json file.")
  else:
    if type(result_dict[list(result_dict.keys())[0]]) is list:
      file_type = 2
    else:
      file_type = 1
  
  show_fig = show
  if "/content" in os.path.abspath(result_path):
    show_fig = True
  
  if file_type == 1:
    new_dict = read_type1_file(result_dict)
  elif file_type == 2:
    new_dict = read_type2_file(result_dict, show_fig)

  if len(result_dict) == len(new_dict):
    jsonfiler.dump(new_dict, result_path, indent = 4)
  else:
    print("Error")


if __name__ == "__main__":
  args = parse_args()
  result_list = [f for f in os.listdir(args.result_dir) if ".json" in f]

  if len(result_list) == 0:
    print("Warning! No result file detected in", args.result_dir)
    exit()

  for i in range(len(result_list)):
    result_path = os.path.join(args.result_dir, result_list[i]) # Result file path
    read_result_file(result_path) 

