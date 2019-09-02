# -*- coding: utf-8 -*


"""
[1]Y. Lou, C. Zhang, Y. Zheng, X. Xie, W. Wang, and Y. Huang,
“Map-matching for Low-sampling-rate GPS Trajectories,”
in Proceedings of the 17th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems,
New York, NY, USA, 2009, pp. 352–361.

"""


from NetworkDistance import network_distance


def normal_distribution(mu, sigma, x):
    """
    calculate normal distribution values
    input:
    mu: the mean value
    sigma; the standard deviation
    x: the given variables
    output:
    the probabilities of given input variables x
    """
    import numpy as np
    return 1.0/(sigma * np.sqrt(2*np.pi)) * (np.exp(-1.0 * (np.array(x)-mu)**2 / (2 * sigma**2)))


def calculate_observation_probability(candidates, mu, sigma):
    for i in range(len(candidates)):
        candidates[i]['observation prob'] = \
            candidates[i].apply(lambda row:
                                normal_distribution(mu, sigma, row['distance']), axis=1)


def calculate_transmission_probability(gc_distance, sp_distance):
    if sp_distance > 0:
        return gc_distance / sp_distance
    elif gc_distance == 0:
        return 0.5
    else:
        return 0.00001


def calculate_transmission_probability_improved(gc_distance, sp_distance):
    min_d = min(gc_distance, abs(sp_distance))
    max_d = max(gc_distance, abs(sp_distance))
    if max_d > 0:
        return min_d / max_d
    else:
        return 0.00001


def calculate_cosine_similarity(avg_speed, sp_edges):
    """
    temporal analysis function
    """
    import math
    a = 0
    b = 0
    c = 0
    for i in range(len(sp_edges)):
        v = sp_edges[i]['max speed']
        a = a + avg_speed * v
        b = b + v * v
        c = c + avg_speed * avg_speed
    if b == 0 or c == 0:
        return 0
    else:
        return a / (math.sqrt(b) * math.sqrt(c))


def calculate_weights_between_candidates(road_graph_utm, gpd_edges_utm, trip, candidates, idx):
    """
    calculate transit weights between consecutive candidates, i.e., the idx-th and (idx+1)-th
    :param road_graph_utm: a networkx digraph, road network
    :param gpd_edges_utm: a geopandas GeoDataFrame, road edges
    :param trip: a geopandas GeoDataFrame, gps trajectory
    :param candidates: a list of pandas DataFrame,
                        column names ['distance', 'from', 'to', 'proj_point', 'road']
    :param idx: a index of a gps point
    :return: weights: a pandas DataFrame,
                    column names ['from_id', 'to_id', 'sp distance', 'gc distance', 'avg speed(km/h)',
                                    'sp edges', 'transmission prob', 'temporal prob', 'weight')]
    """
    import pandas as pd
    # the great circle distance (euclidean distance) between the idx-1 and the idx-th sampling points
    great_circle_distance = trip.iloc[idx]['geometry'].distance(trip.iloc[idx - 1]['geometry'])
    # print great_circle_distance
    # the time gap between the idx-1 and the idx-th sampling points
    delta = trip.iloc[idx+1]['timestamp'] - trip.iloc[idx]['timestamp']
    weights = pd.DataFrame(columns=('from_id', 'to_id', 'sp distance', 'gc distance', 'avg speed(km/h)',
                                    'sp edges', 'transmission prob', 'temporal prob', 'weight'))
    for i in range(len(candidates[idx])):
        for j in range(len(candidates[idx + 1])):
            sp_distance, sp_edges = network_distance(road_graph_utm,
                                                     gpd_edges_utm,
                                                     candidates[idx].iloc[i],
                                                     candidates[idx+1].iloc[j])
            # transmission probability
            if len(sp_edges):
                t_p = calculate_transmission_probability_improved(great_circle_distance, sp_distance)
            else:
                t_p = 0.000000001  # no route between the two candidates
            # temporal analysis weight
            avg_speed = 200
            if delta > 0 and sp_distance >= 0:
                avg_speed = sp_distance / delta * 3.6
            elif sp_distance < 0:
                avg_speed = 0
            c_s = calculate_cosine_similarity(avg_speed, sp_edges)
            weight = candidates[idx+1].iloc[j]['observation prob'] * t_p * c_s
            s = pd.Series({'from_id': i,
                           'to_id': j,
                           'sp distance': sp_distance,
                           'gc distance': great_circle_distance,
                           'avg speed(km/h)': avg_speed,
                           'sp edges': sp_edges,
                           'transmission prob': t_p,
                           'temporal prob': c_s,
                           'weight': weight})
            weights = weights.append(s, ignore_index=True)
    weights[['from_id', 'to_id']] = weights[['from_id', 'to_id']].astype(int)
    return weights


def calculate_weights(road_graph_utm, gpd_edges_utm, trip, candidates):
    """
    calculate transit weights
    :param road_graph_utm: a networkx digraph, road network
    :param gpd_edges_utm: a geopandas GeoDataFrame, road edges
    :param trip: a geopandas GeoDataFrame, gps trajectory
    :param candidates: a list of pandas DataFrame,
                        column names ['distance', 'from', 'to', 'proj_point', 'road']
    :return: weights: a list of pandas DataFrame,
                    column names ['from_id', 'to_id', 'sp distance', 'gc distance', 'avg speed(km/h)',
                                    'sp edges', 'transmission prob', 'temporal prob', 'weight')]
    """
    weights = []
    for i in range(len(candidates)-1):
        weights_i = calculate_weights_between_candidates(road_graph_utm, gpd_edges_utm, trip, candidates, i)
        weights.append(weights_i)
    return weights


# find the optimal path
def find_optimal_path(candidates, weights):
    # forward search
    f = [list(candidates[0]['observation prob'])]
    pre = [[]]
    for i in range(1, len(candidates)):
        f_i = []
        pre_i = []
        for j in range(len(candidates[i])):
            f_max = -100000000.0
            parent_ind = 0
            for k in range(len(candidates[i-1])):
                ind = k*len(candidates[i]) + j
                alt = f[i-1][k] + weights[i-1].iloc[ind]['weight']
                if alt > f_max:
                    f_max = alt
                    parent_ind = k
            pre_i.append(parent_ind)
            f_i.append(f_max)
        f.append(f_i)
        pre.append(pre_i)
    # backward search
    optimal_path = [f[-1].index(max(f[-1]))]
    for i in range(len(pre) - 1, 0, -1):
        c = pre[i][optimal_path[0]]
        optimal_path.insert(0, c)
    return optimal_path, f, pre


def save_to_file_fscore(filename, fScores):
    with open(filename, 'w') as fWriter:
        for i in range(len(fScores)):
            for j in range(len(fScores[i])):
                fWriter.write('%f ' % fScores[i][j])
            fWriter.write('\n')


def save_to_file_pre(filename, pre):
    with open(filename, 'w') as fwritter:
        for i in range(len(pre)):
            for j in range(len(pre[i])):
                fwritter.write('%d ' % pre[i][j])
            fwritter.write('\n')


def save_to_file_observation_probabilities(filename, candidates):
    with open(filename, 'w') as fWriter:
        for i in range(len(candidates)):
            for j in range(len(candidates[i])):
                fWriter.write('%f ' % candidates[i].iloc[j]['observation prob'])
            fWriter.write('\n')


def save_to_file_weights(filename, weights):
    with open(filename, 'w') as fWriter:
        for i in range(len(weights)):
            pre_ind = 0
            for j in range(len(weights[i])):
                if pre_ind != weights[i].iloc[j]['from_id']:
                    pre_ind = weights[i].iloc[j]['from_id']
                    fWriter.write('\n')
                fWriter.write('%f ' % weights[i].iloc[j]['weight'])
            fWriter.write('\n\n')


def st_mapper(road_graph_utm, gpd_edges_utm, trip, candidates, debug=False):
    mu = 0  # mean, parameter for calculating observation probability
    sigma = 10  # standard deviation, parameter for calculating observation probability
    # calculate observation probability
    calculate_observation_probability(candidates, mu, sigma)
    # calculate weights
    weights = calculate_weights(road_graph_utm, gpd_edges_utm, trip, candidates)
    # finding the optimal path (viterbi algorithm)
    optimal_path, f_score, pre = find_optimal_path(candidates, weights)

    if debug:
        import os
        cur_dir = os.getcwd()
        debug_dir = os.path.join(cur_dir, r'debug_results')
        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)
        save_to_file_observation_probabilities(
            os.path.join(debug_dir, r'st_observation_probabilities.txt'), candidates)
        save_to_file_weights(
            os.path.join(debug_dir, r'st_weights.txt'), weights)
        save_to_file_fscore(
            os.path.join(debug_dir, r'st_fscore.txt'), f_score)
        save_to_file_pre(
            os.path.join(debug_dir, r'st_pre.txt'), pre)
    return optimal_path, weights


