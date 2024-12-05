from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.sparse import coo_matrix
import os
import gzip
import json
import numpy as np
import pandas as pd
import torch
import sys
from torch_geometric.data import Data

sys.path.append('../data/')  # Add the subdirectory to the Python path

data_path = '../data/'  # Path to the data directory

def getGRCIndex(x, y, xbst, ybst):
    while isinstance(xbst, tuple):
        if x < xbst[0]:
            xbst = xbst[1]
        else:
            xbst = xbst[2]
    while isinstance(ybst, tuple):
        if y < ybst[0]:
            ybst = ybst[1]
        else:
            ybst = ybst[2]
    return ybst, xbst

def buildBST(array, start=0, finish=-1):
    if finish < 0:
        finish = len(array)
    mid = (start + finish) // 2
    if mid - start == 1:
        ltl = start
    else:
        ltl = buildBST(array, start, mid)
    if finish - mid == 1:
        gtl = mid
    else:
        gtl = buildBST(array, mid, finish)
    return (array[mid], ltl, gtl)

def feature_engineering(feature_lst, instances):
    scaler = MinMaxScaler()
    onehotencoder = OneHotEncoder()

    # Normalize continuous features
    continuous_features = instances[feature_lst]
    normalized_continuous = scaler.fit_transform(continuous_features)

    # One-hot encode the 'orient' feature
    orient_encoded = onehotencoder.fit_transform(instances[['orient']]).toarray()

    # Combine normalized features and one-hot encoded features
    node_features = np.hstack([normalized_continuous, orient_encoded])

    return node_features

def create_data(design_number):
    # Load design data
    with gzip.open(os.path.join(data_path, f'xbar/{design_number}/xbar.json.gz'), 'rb') as f:
        design = json.loads(f.read().decode('utf-8'))

    # Create DataFrames
    instances = pd.DataFrame(design['instances'])
    nets = pd.DataFrame(design['nets'])

    # Load connectivity data
    conn = np.load(os.path.join(data_path, f'xbar/{design_number}/xbar_connectivity.npz'))
    A = coo_matrix((conn['data'], (conn['row'], conn['col'])), shape=conn['shape'])
    A = A.__mul__(A.T)

    # Load congestion data
    congestion_data = np.load(os.path.join(data_path, f'xbar/{design_number}/xbar_congestion.npz'))

    # Build boundary segment trees
    xbst = buildBST(congestion_data['xBoundaryList'])
    ybst = buildBST(congestion_data['yBoundaryList'])

    # Initialize demand and capacity arrays
    demand = np.zeros(instances.shape[0])
    capacity = np.zeros(instances.shape[0])

    # Map congestion data
    for k in range(instances.shape[0]):
        xloc, yloc = instances.iloc[k]['xloc'], instances.iloc[k]['yloc']
        try:
            i, j = getGRCIndex(xloc, yloc, xbst, ybst)
            d = sum(congestion_data['demand'][lyr][i][j] for lyr in range(len(congestion_data['layerList'])))
            c = sum(congestion_data['capacity'][lyr][i][j] for lyr in range(len(congestion_data['layerList'])))
            demand[k] = d
            capacity[k] = c
        except IndexError:
            print(f"Out-of-bounds for instance {k} at xloc: {xloc}, yloc: {yloc}")

    # Add columns to instances DataFrame
    instances['routing_demand'] = demand
    instances['routing_capacity'] = capacity
    instances['congestion'] = demand - capacity

    # Continuous features to normalize
    continuous_features = ['xloc', 'yloc', 'routing_demand', 'routing_capacity']
    node_features = feature_engineering(continuous_features, instances)
    x_nodes = torch.tensor(node_features, dtype=torch.float)

    num_nets = A.shape[1] - len(instances)  # Nets are the remaining rows/columns in adjacency matrix
    x_nets = torch.zeros(num_nets, x_nodes.shape[1])  # Initialize net features as zeros

    # Combine node and net features
    x_combined = torch.cat([x_nodes, x_nets], dim=0)

    # Ensure A is in COO format
    A_coo = A.tocoo()

    # Extract edge indices
    node_to_net_edge_index = torch.tensor(np.vstack((A_coo.row, A_coo.col)), dtype=torch.long)
    net_to_node_edge_index = torch.tensor(np.vstack((A_coo.col, A_coo.row)), dtype=torch.long)

    # Combine the edge indices
    edge_index = torch.cat([node_to_net_edge_index, net_to_node_edge_index], dim=1)

    # Use congestion as the target variable
    targets = torch.tensor(instances['congestion'].values, dtype=torch.float).view(-1, 1)

    # Create the Data object
    hypergraph_data = Data(x=x_combined, edge_index=edge_index, y=targets)

    return instances, hypergraph_data