import pandas as pd
from itertools import product
import math
import mne
import torch
import numpy as np

class Distance_Weight():
    def __init__(self):
        self.ref_names = {
        1: "FC5", 2: "FC3", 3: "FC1", 4: "FCz", 5: "FC2", 6: "FC4",
        7: "FC6", 8: "C5", 9: "C3", 10: "C1", 11: "Cz", 12: "C2",
        13: "C4", 14: "C6", 15: "CP5", 16: "CP3", 17: "CP1", 18: "CPz",
        19: "CP2", 20: "CP4", 21: "CP6", 22: "Fp1", 23: "Fpz", 24: "Fp2",
        25: "AF7", 26: "AF3", 27: "AFz", 28: "AF4", 29: "AF8", 30: "F7",
        31: "F5", 32: "F3", 33: "F1", 34: "Fz", 35: "F2", 36: "F4",
        37: "F6", 38: "F8", 39: "FT7", 40: "FT8", 41: "T7", 42: "T8",
        43: "T9", 44: "T10", 45: "TP7", 46: "TP8", 47: "P7", 48: "P5",
        49: "P3", 50: "P1", 51: "Pz", 52: "P2", 53: "P4", 54: "P6",
        55: "P8", 56: "PO7", 57: "PO3", 58: "POz", 59: "PO4", 60: "PO8",
        61: "O1", 62: "Oz", 63: "O2", 64: "Iz"}

        self.channels = range(len(self.ref_names))
        self.edge_index = torch.tensor([[a, b] for a, b in product(self.channels, self.channels)], dtype=torch.long).t().contiguous()

    	# only the spatial distance between electrodes - standardize between 0 and 1
        self.distance = self.get_sensor_distances()
        a = np.array(self.distance)
        self.distance = (a - np.min(a)) / (np.max(a) - np.min(a))
        
    def get_sensor_distances(self):
        coords_1010 = pd.read_csv("electrode_positions.txt", sep=' ')
        num_edges = self.edge_index.shape[1]
        distances = []
        for edge_idx in range(num_edges):
            sensor1_idx = self.edge_index[0, edge_idx]
            sensor2_idx = self.edge_index[1, edge_idx]
            dist = self.get_geodesic_distance(sensor1_idx + 1, sensor2_idx + 1, coords_1010)
            distances.append(dist)
        
        assert len(distances) == num_edges
        return distances
    
    def get_geodesic_distance(self, montage_sensor1_idx, montage_sensor2_idx, coords_1010):
        # get the reference sensor in the 10-10 system for the current montage pair in 10-20 system
        ref_sensor1 = self.ref_names.get(montage_sensor1_idx.item())
        ref_sensor2 = self.ref_names.get(montage_sensor2_idx.item())

        x1 = float(coords_1010[coords_1010.label == ref_sensor1]["x"])
        y1 = float(coords_1010[coords_1010.label == ref_sensor1]["y"])
        z1 = float(coords_1010[coords_1010.label == ref_sensor1]["z"])


        # print(ref_sensor2, montage_sensor2_idx, coords_1010[coords_1010.label == ref_sensor2]["x"])
        x2 = float(coords_1010[coords_1010.label == ref_sensor2]["x"])
        y2 = float(coords_1010[coords_1010.label == ref_sensor2]["y"])
        z2 = float(coords_1010[coords_1010.label == ref_sensor2]["z"])

        # https://math.stackexchange.com/questions/1304169/distance-between-two-points-on-a-sphere
        import math
        r = 1 # since coords are on unit sphere
        # rounding is for numerical stability, domain is [-1, 1]		
        dist = r * math.acos(round(((x1 * x2) + (y1 * y2) + (z1 * z2)) / (r**2), 2))
        return dist