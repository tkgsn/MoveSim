import pandas as pd
import json
import numpy as np
import bisect
import random
from sklearn.preprocessing import normalize
import pathlib
from collections import Counter
from geopy.distance import geodesic
                 

def load_dataset(data_dir, *, logger):
    trajectories = []

    # for data_dir in data_dirs:
    logger.info(f"load data from {data_dir}")
    data = pd.read_csv(data_dir, header=None).astype(str).values
    for trajectory in data:
        trajectory = [int(float(v)) for v in trajectory if (v != 'nan')]
        trajectories.append(trajectory)
    logger.info(f"length of dataset: {len(trajectories)}")
    return trajectories
                 
def get_datadir():
    with open(f"config.json", "r") as f:
        config = json.load(f)
    return pathlib.Path(config["data_dir"])


def get_gps(dataset):
    df = pd.read_csv(get_datadir() / f"{dataset}/gps.csv", header=None)
    return df.values[:,1], df.values[:,0]

def make_gps(lat_range, lon_range, n_bins):
    
    x_axis = np.linspace(lon_range[0], lon_range[1], n_bins+2)
    y_axis = np.linspace(lat_range[0], lat_range[1], n_bins+2)
    
    def state_to_latlon(state):
        x_state = int(state % (n_bins+2))
        y_state = int(state / (n_bins+2))
        return y_axis[y_state], x_axis[x_state]
    
    return pd.DataFrame([state_to_latlon(i) for i in range((n_bins+2)**2)])


def compute_transition_matrix(training_data, max_locs):
    reg1 = np.zeros([max_locs,max_locs])
    for line in training_data:
        for j in range(len(line)-1):
            if (line[j] >= max_locs) or (line[j+1] >= max_locs):
#                 print("WARNING: outside location found")
                continue
            reg1[line[j],line[j+1]] +=1
    return reg1

def compute_distance_matrix(state_to_latlon, n_locations):
    distance_matrix = np.zeros((n_locations, n_locations))
    for i in range(n_locations):
        for j in range(n_locations):
            distance_matrix[i, j] = geodesic(state_to_latlon(i), state_to_latlon(j)).meters
    return distance_matrix


def load_latlon_range(name):
    with open(f"{name}.json", "r") as f:
        configs = json.load(f)
    lat_range = configs["lat_range"]
    lon_range = configs["lon_range"]
    return lat_range, lon_range
    
def latlon_to_state(lat, lon, lat_range, lon_range, n_bins):
    x_axis = np.linspace(lon_range[0], lon_range[1], n_bins+1)
    y_axis = np.linspace(lat_range[0], lat_range[1], n_bins+1)
    
    x_state = bisect.bisect_left(x_axis, lon)
    y_state = bisect.bisect_left(y_axis, lat)
    # if x_state == n_bins+2:
    #     x_state = n_bins
    # if y_state == n_bins+2:
    #     y_state = n_bins
    return y_state*(n_bins+2) + x_state

def make_hist_2d(counts, n_bins):
    hist2d = [[0 for i in range(n_bins+2)] for j in range(n_bins+2)]
    for state in range((n_bins+2)**2):
        x,y = state_to_xy(state, n_bins)
        hist2d[x][y] = counts[state]
    return np.array(hist2d)

def state_to_xy(state, n_bins):
    n_x = n_bins+2
    n_y = n_bins+2

    x = (state) % n_x
    y = int((state) / n_y)
    return x, y

def split_train_test(df, seed, split_ratio=0.5):
    random.seed(seed)
    n_records = len(df.index)
    choiced_indice = random.sample(range(n_records), int(n_records*split_ratio))
    removed_indice = [i for i in range(n_records) if i not in choiced_indice]
    training_df = df.loc[choiced_indice]
    test_df = df.loc[removed_indice]
    return training_df, test_df