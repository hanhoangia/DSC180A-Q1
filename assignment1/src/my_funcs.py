import pandas as pd
import networkx as nx

def load_data():
    stations = pd.read_csv('data/stations.csv')
    ridership = pd.read_csv('data/ridership.csv')
    # keep only 'Origin Station Complex ID', 'Origin Station Complex Name', 'Destination Station Complex ID', 'Destination Station Complex Name', 'Estimated Average Ridership', 'Day of Week', 'Hour of Day'
    ridership = ridership[['Origin Station Complex ID', 'Origin Station Complex Name', 'Destination Station Complex ID', 'Destination Station Complex Name', 'Estimated Average Ridership', 'Day of Week', 'Hour of Day']]

    return stations, ridership

# Load the data
stations, ridership = load_data()

# Get the station names
def getStationNames(station_ids):
    returnStr = ''

    if len(station_ids) == 0:
        returnStr = 'No stations found  '

    for id, _ in station_ids:
        returnStr += stations[stations['Complex ID'] == id]['Stop Name'].mode()[0] + ", "

    return returnStr[:-2]

# Get the names of a pair of stations
def getPairNames(id_pairs):
    returnStr = ''

    if len(id_pairs) == 0:
        returnStr = 'No pairs found '

    for id1, id2 in id_pairs:
        returnStr += stations[stations['Complex ID'] == id1]['Stop Name'].mode()[0] + ' - ' + stations[stations['Complex ID'] == id2]['Stop Name'].mode()[0] + '\n'

    return returnStr[:-1]

# Get the station names for the top 5 origin stations with the highest ridership
def computeSumOutDegreeEdgeWeights(subset):
    # using multi-directed graph representation
    G = nx.MultiDiGraph()

    # add nodes and edges to the graph
    edges_with_weights = list(zip(
        subset['Origin Station Complex ID'], 
        subset['Destination Station Complex ID'], 
        subset['Estimated Average Ridership']
    ))

    G.add_edges_from(edges_with_weights)

    # compute the sum of the out-degree edge weights for each origin station and get the top 5 with the highest sum
    top_5 = sorted(dict(G.out_degree()).items(), key=lambda x: x[1], reverse=True)[:5]

    return getStationNames(top_5)

# Get the station names for the top 5 destination stations with the highest ridership
def computeSumInDegreeEdgeWeights(subset):
    # using multi-directed graph representation
    G = nx.MultiDiGraph()

    # add nodes and edges to the graph
    edges_with_weights = list(zip(
        subset['Origin Station Complex ID'], 
        subset['Destination Station Complex ID'], 
        subset['Estimated Average Ridership']
    ))

    G.add_edges_from(edges_with_weights)

    # compute the sum of the in-degree edge weights for each origin station and get the top 5 with the highest sum
    top_5 = sorted(dict(G.in_degree()).items(), key=lambda x: x[1], reverse=True)[:5]

    return getStationNames(top_5)

# Get the top 10 station pairs with the highest ridership
def computeSumEdgeWeights(subset):
    # using undirected graph representation
    G = nx.Graph()

    ordered = subset.reset_index(drop=True)
    ordered['min'] = ordered[['Origin Station Complex ID', 'Destination Station Complex ID']].min(axis=1)
    ordered['max'] = ordered[['Origin Station Complex ID', 'Destination Station Complex ID']].max(axis=1)
    paired = ordered.groupby(['min', 'max'])['Estimated Average Ridership'].sum().reset_index()
    
    edges_with_weights = list(zip(
        paired['min'], 
        paired['max'], 
        paired['Estimated Average Ridership']
    ))

    G.add_weighted_edges_from(edges_with_weights)

    # compute the sum of edge weights and get the top 10 edges with the highest sums
    top_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:10]

    top_nodes = [(edge[0], edge[1]) for edge in top_edges]

    return getPairNames(top_nodes)

