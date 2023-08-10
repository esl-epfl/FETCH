import numpy as np
from itertools import permutations
from tqdm import tqdm
import json
import os


ones_count = 3

channels = [("FP1", "F7"),
            ("F7", "T3"),
            ("T3", "T5"),
            ("T5", "O1"),
            ("FP1", "F3"),
            ("F3", "C3"),
            ("C3", "P3"),
            ("P3", "O1"),

            ("FP2", "F8"),
            ("F8", "T4"),
            ("T4", "T6"),
            ("T6", "O2"),
            ("FP2", "F4"),
            ("F4", "C4"),
            ("C4", "P4"),
            ("P4", "O2"),

            ("Fz", "Cz"),
            ("Pz", "Cz"),
            ("C3", "Cz"),
            ("C4", "Cz"),
            ]


def check_feasibility(graph_edges, edge_weights):
    degrees = {'Cz': 0, 'C3': 0, 'F3': 0, 'P3': 0, 'T6': 0, 'O2': 0, 'C4': 0, 'Pz': 0, 'T4': 0, 'O1': 0, 'F7': 0,
               'F8': 0, 'P4': 0, 'FP1': 0, 'Fz': 0, 'F4': 0, 'T5': 0, 'FP2': 0, 'T3': 0}

    # Calculate degrees for each node
    for edge in graph_edges:
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1

    # Check if any edge with weight 0 connects two nodes with degree >= 1
    feasible = True
    for i, weight in enumerate(edge_weights):
        if weight == 0 and degrees[channels[i][0]] >= 1 and degrees[channels[i][1]] >= 1:
            feasible = False
            break

    return feasible


def generate_edge_weights():
    edge_weights = []
    total_bits = 20

    # Generate all combinations of 1s and 0s
    for i in tqdm(range((2 ** total_bits))):
        binary_str = bin(i)[2:].zfill(total_bits)

        # Count the number of 1s
        if binary_str.count('1') == ones_count:
            edge_weights.append([int(bit) for bit in binary_str])

    return edge_weights


# Generate all edge_weights lists
all_edge_weights = generate_edge_weights()

all_feasible_edge_weights = []
for edge_weight_list in tqdm(all_edge_weights):
    if check_feasibility([channels[i] for i, x in enumerate(edge_weight_list) if x==1], edge_weight_list):
        all_feasible_edge_weights.append([i for i, x in enumerate(edge_weight_list) if x==1])


# Save the list as a JSON file
filename = "../feasible_channels/feasible_{}edges.json".format(ones_count)
if os.path.exists(filename):
    raise ValueError("The feasible channels file already exists")

print("Total feasible combination", len(all_feasible_edge_weights))

with open(filename, 'w') as json_file:
    json.dump(all_feasible_edge_weights, json_file)
