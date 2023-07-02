import json
import pathlib
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from my_utils import get_datadir, compute_distance_matrix, compute_transition_matrix, make_gps, load_dataset
from grid import Grid
import glob
import tqdm
from datetime import datetime
from bisect import bisect_left
from logging import getLogger, config

# peopleflow_raw_data_dir = "/data/peopleflow/tokyo2008/p-csv/0000/*.csv"
format = '%H:%M:%S'
basic_time = datetime.strptime("00:00:00", format)

# check if the location is in the range
def in_range(lat_range, lon_range, lat, lon):
    return float(lat_range[0]) <= lat <= float(lat_range[1]) and float(lon_range[0]) <= lon <= float(lon_range[1])


def make_stay_trajectory(trajectories, time_threshold, location_threshold):

    print(f"make stay trajectory with threshold {location_threshold}m and {time_threshold}min")

    stay_trajectories = []
    time_trajectories = []
    for trajectory in tqdm.tqdm(trajectories):

        stay_trajectory = []
        # remove nan
        trajectory = [v for v in trajectory if type(v) is str]
        time_trajectory = []

        start_index = 0
        start_time = 0
        i = 0

        while True:
            # find the length of the stay
            start_location = trajectory[start_index].split(" ")
            start_location = (float(start_location[1]), float(start_location[2]))

            if i == len(trajectory)-1:
                time_trajectory.append((start_time, time))
                stay_trajectory.append(target_location)
                # print("finish", start_time, time, start_location)
                break

            for i in range(start_index+1, len(trajectory)):

                target_location = trajectory[i].split(" ")
                time = float(target_location[0])
                target_location = (float(target_location[1]), float(target_location[2]))
                distance = geodesic(start_location, target_location).meters
                if distance > location_threshold:
                    # print(f"move {distance}m", start_time, time, trajectory[i])
                    if time - start_time >= time_threshold:
                        # print("stay", start_time, time, start_location)
                        stay_trajectory.append(start_location)
                        time_trajectory.append((start_time, time))

                    start_time = time
                    # print(trajectory[i])
                    start_index = i
                    # print("start", start_time, start_index, len(trajectory))
                    # print(time, i)

                    break
        
        stay_trajectories.append(stay_trajectory)
        time_trajectories.append(time_trajectory)
    return time_trajectories, stay_trajectories


def make_complessed_dataset(time_trajectories, trajectories, grid):
    dataset = []
    times = []
    for trajectory, time_trajectory in tqdm.tqdm(zip(trajectories, time_trajectories)):
        state_trajectory = []
        for lat, lon in trajectory:
            state = grid.latlon_to_state(lat, lon)
            state_trajectory.append(state)

        if None in state_trajectory:
            continue

        # compless time trajectory according to state trajectory
        complessed_time_trajectory = []
        j = 0
        for i, time in enumerate(time_trajectory):
            if i != j:
                continue   
            target_state = state_trajectory[i]
            # find the max length of the same states
            for j in range(i+1, len(state_trajectory)+1):
                if j == len(state_trajectory):
                    break
                if (state_trajectory[j] != target_state):
                    break
            complessed_time_trajectory.append((time[0],time_trajectory[j-1][1]))

        # remove consecutive same states
        state_trajectory = [state_trajectory[0]] + [state_trajectory[i] for i in range(1, len(state_trajectory)) if state_trajectory[i] != state_trajectory[i-1]]

        dataset.append(state_trajectory)
        times.append(complessed_time_trajectory)

        assert len(state_trajectory) == len(complessed_time_trajectory), f"state trajectory length {len(state_trajectory)} != time trajectory length {len(complessed_time_trajectory)}"
        # times.append([time for time, _, _ in trajectory])
    return dataset, times


def str_to_minute(time_str):
    format = '%H:%M:%S'
    return int((datetime.strptime(time_str, format) - basic_time).seconds / 60)
    

def split(time, seq_len, start_hour, end_hour):
    start_time = start_hour * 60
    end_time = end_hour * 60
    
    time_range = (end_time - start_time) / seq_len
    target_times = [i*time_range for i in range(seq_len)]
    
    split_indices = []
    for target_time in target_times:
        split_indices.append(bisect_left(time, target_time))
        
    return split_indices

def save_with_nan_padding(save_path, trajectories, formater, verbose=False):
    # compute the max length in trajectories
    max_len = max([len(trajectory) for trajectory in trajectories])

    if verbose:
        print(f"save to {save_path}")
    with open(save_path, "w") as f:
        for trajectory in trajectories:
            for record in trajectory:
                f.write(formater(record))
            # padding with nan
            for _ in range(max_len - len(trajectory)):
                f.write(",")
            f.write("\n")

def save_timelatlon_with_nan_padding(save_path, trajectories):
    def formater(record):
        return f"{record[0]} {record[1]} {record[2]},"
    
    save_with_nan_padding(save_path, trajectories, formater)

def save_latlon_with_nan_padding(save_path, trajectories):
    def formater(record):
        return f"{record[1]} {record[2]},"
    
    save_with_nan_padding(save_path, trajectories, formater)

def save_state_with_nan_padding(save_path, trajectories, verbose=False):
    def formater(record):
        return f"{record},"
    
    save_with_nan_padding(save_path, trajectories, formater, verbose=verbose)

def save_time_with_nan_padding(save_path, trajectories, max_time):
    def formater(record):
        return f"{int(record[0])},"
    
    for trajectory in trajectories:
        trajectory.append([max_time])
    
    save_with_nan_padding(save_path, trajectories, formater)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--latlon_config', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--n_bins', type=int)
    parser.add_argument('--time_threshold', type=int)
    parser.add_argument('--location_threshold', type=int)
    parser.add_argument('--save_name', type=str)
    args = parser.parse_args()
    

    with open(pathlib.Path("./") / "dataset_configs" / args.latlon_config, "r") as f:
        configs = json.load(f)
    
    configs.update(vars(args))
    data_path = get_datadir() / args.dataset / args.data_name / args.save_name
    data_path.mkdir(exist_ok=True, parents=True)

    print("loading setting from", data_path / "params.json")

    with open(data_path / "params.json", "w") as f:
        json.dump(configs, f)
        
    lat_range = configs["lat_range"]
    lon_range = configs["lon_range"]
    n_bins = args.n_bins
    time_threshold = args.time_threshold
    location_threshold = args.location_threshold

    max_locs = (n_bins+2)**2
    max_time = 24*60-1

    save_path = data_path / f"training_data.csv"

    with open('./log_config.json', 'r') as f:
        log_conf = json.load(f)
    log_conf["handlers"]["fileHandler"]["filename"] = str(data_path / "log.log")
    config.dictConfig(log_conf)
    logger = getLogger(__name__)
    logger.info('log is saved to {}'.format(data_path / "log.log"))
    logger.info(f'used parameters {vars(args)}')

    if not save_path.exists():

        logger.info("make grid", lat_range, lon_range, n_bins)
        ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
        grid = Grid(ranges)

        raw_data_path = data_path.parent / "raw_data.csv"
        # if configs["dataset"] == "geolife" or configs["dataset"] == "geolife_test":

        logger.info(f"load raw data from {raw_data_path}")
        trajs = pd.read_csv(raw_data_path, header=None).values
        
        logger.info("make stay trajectory")
        time_trajs, trajs = make_stay_trajectory(trajs, time_threshold, location_threshold)
        logger.info("make complessed dataset")
        dataset, times = make_complessed_dataset(time_trajs, trajs, grid)
        logger.info(f"save complessed dataset to {save_path}")
        save_state_with_nan_padding(save_path, dataset)
        
        time_save_path = data_path / f"training_data_time.csv"
        save_time_with_nan_padding(time_save_path, times, max_time)

        training_data = pd.DataFrame(dataset).values
    else:
        logger.info(f"training data already exists: {save_path}")
        training_data = load_dataset(save_path, logger=logger)
    
    if not (data_path / "gps.csv").exists():
        gps = make_gps(lat_range, lon_range, n_bins)
        gps.to_csv(data_path / f"gps.csv", header=None, index=None)
        logger.info(gps)
    else:
        logger.info("GPS exists")

    if not (data_path / "distance_matrix.npy").exists():
        logger.info(f"make distance matrix using {lat_range}, {lon_range}, {n_bins}")
        ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
        grid = Grid(ranges)
        distance_matrix = compute_distance_matrix(grid.state_to_center_latlon, grid.vocab_size)
        np.save(data_path/f'distance_matrix.npy',distance_matrix)
    else:
        logger.info("distance matrix exists")

    if not (data_path / "transition_matrix.npy").exists():
        logger.info("make transition matrix")
        transition_matrix = compute_transition_matrix(training_data, max_locs)
        np.save(data_path / f'transition_matrix.npy',transition_matrix)
    else:
        logger.info("transition matrix exists")
