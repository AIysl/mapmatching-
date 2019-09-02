# -*- coding: utf-8 -*


"""
[1]P. Newson and J. Krumm,
“Hidden Markov Map Matching Through Noise and Sparseness,”
in Proceedings of the 17th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems,
New York, NY, USA, 2009, pp. 336–343.

"""


from NetworkDistance import network_distance
from STMatching import calculate_observation_probability, save_to_file_observation_probabilities
from STMatching import save_to_file_weights, save_to_file_fscore, save_to_file_pre


def calculate_transition_probability(beta, gc_distance, sp_distance):
    import math
    diff = abs(gc_distance-sp_distance)
    return math.exp(-diff/beta) / beta


def calculate_weights_between_candidates(road_graph_utm, gpd_edges_utm, trip, candidates, idx, beta):
    """
    calculate weights between the idx-1 and the idx th points's candidates
    input:
    idx: the idx-th sampling point (idx > 0)
    """
    import pandas as pd
    # the great circle distance (euclidean distance) between the idx-1 and the idx-th sampling points
    great_circle_distance = trip.iloc[idx+1]['geometry'].distance(trip.iloc[idx]['geometry'])
    # the time gap between the idx-1 and the idx-th sampling points
    weights = pd.DataFrame(columns=('from_id', 'to_id', 'sp distance', 'gc distance',
                                    'sp edges', 'weight'))
    for i in range(len(candidates[idx])):
        for j in range(len(candidates[idx+1])):
            sp_distance, sp_edges = network_distance(road_graph_utm,
                                                     gpd_edges_utm,
                                                     candidates[idx].iloc[i],
                                                     candidates[idx+1].iloc[j])
            # transmission probility
            t_p = calculate_transition_probability(beta, great_circle_distance, sp_distance)
            s = pd.Series({'from_id': i,
                           'to_id': j,
                           'sp distance': sp_distance,
                           'gc distance': great_circle_distance,
                           'sp edges': sp_edges,
                           'weight': t_p})
            weights = weights.append(s, ignore_index=True)
    weights[['from_id', 'to_id']] = weights[['from_id', 'to_id']].astype(int)
    return weights


def calculate_weights(road_graph_utm, gpd_edges_utm, trip, candidates, beta):
    weights = []
    for i in range(len(candidates)-1):
        weight_i = calculate_weights_between_candidates(road_graph_utm, gpd_edges_utm, trip, candidates, i, beta)
        weights.append(weight_i)
    return weights


# find the optimal path viterbi algorithm
def find_optimal_path(candidates, weights):
    # forward search
    import math
    # f = [list(candidates[0]['observation prob'])]
    f = [list(candidates[0].apply(lambda x: math.log(x['observation prob']), axis=1))]
    pre = [[]]
    for i in range(1, len(candidates)):
        f_i = []
        pre_i = []
        for j in range(len(candidates[i])):
            f_max = -100000000.0
            parent_ind = 0
            for k in range(len(candidates[i-1])):
                ind = k*len(candidates[i]) + j
                # alt = f[i-1][k] * weights[i-1].iloc[ind]['weight']
                alt = f[i-1][k] + math.log(weights[i-1].iloc[ind]['weight'])
                if alt > f_max:
                    f_max = alt
                    parent_ind = k
            pre_i.append(parent_ind)
            # f_i.append(f_max * candidates[i].iloc[j]['observation prob'])
            f_i.append(f_max + math.log(candidates[i].iloc[j]['observation prob']))
        f.append(f_i)
        pre.append(pre_i)
    # backward search
    optimal_path = [f[-1].index(max(f[-1]))]
    for i in range(len(pre) - 1, 0, -1):
        c = pre[i][optimal_path[0]]
        optimal_path.insert(0, c)
    return optimal_path, f, pre


def hmm_mapper(road_graph_utm, gpd_edges_utm, trip, candidates, debug=False):
    mu = 0  # mean, parameter for calculating observation probability
    sigma = 10  # standard deviation, parameter for calculating observation probability
    # calculate observation probability
    calculate_observation_probability(candidates, mu, sigma)
    # calculate weights
    beta = 1000
    weights = calculate_weights(road_graph_utm, gpd_edges_utm, trip, candidates, beta)
    # finding the optimal path (viterbi algorithm)
    optimal_path, f_score, pre = find_optimal_path(candidates, weights)
    if debug:
        import os
        cur_dir = os.getcwd()
        debug_dir = os.path.join(cur_dir, r'debug_results')
        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)
        save_to_file_observation_probabilities(
            os.path.join(debug_dir, r'debug_hmm_observation_probabilities.txt'), candidates)
        save_to_file_weights(
            os.path.join(debug_dir, r'debug_hmm_weights.txt'), weights)
        save_to_file_fscore(
            os.path.join(debug_dir, r'debug_hmm_fscore.txt'), f_score)
        save_to_file_pre(
            os.path.join(debug_dir, r'debug_hmm_pre.txt'), pre)
    return optimal_path, weights

