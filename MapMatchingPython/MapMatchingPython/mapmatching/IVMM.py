# -*- coding: utf-8 -*


"""
[1]J. Yuan, Y. Zheng, C. Zhang, X. Xie, and G.-Z. Sun,
“An Interactive-Voting Based Map Matching Algorithm,”
in Proceedings of the 2010 Eleventh International Conference on Mobile Data Management,
Washington, DC, USA, 2010, pp. 43–52.

"""


from STMatching import calculate_observation_probability, calculate_weights


# static score matrix
def calculate_static_score_matrix(weights):
    import numpy as np
    static_score = []
    for i in range(len(weights)):
        weights_matrix = []
        pre_ind = 0
        row = []
        for j in range(len(weights[i])):
            if pre_ind != weights[i].iloc[j]['from_id']:
                pre_ind = weights[i].iloc[j]['from_id']
                weights_matrix.append(row)
                row = []
            row.append(weights[i].iloc[j]['weight'])
        weights_matrix.append(row)
        static_score.append(np.array(weights_matrix))
    return static_score


# calculate distance weight
def calculate_distance_weight(trip, idx, beta):
    """
    beta: is the parameter of the distance weight function
    """
    import math
    weights = []
    for i in range(len(trip)):
        if i != idx:
            distance = trip.iloc[idx]['geometry'].distance(trip.iloc[i]['geometry'])
            weight = math.exp(- (distance * distance) / (beta * beta))
            weights.append(weight)
        # else:
        #    weights.append(1)
    return weights


# calculate weighted score matrix
def calculate_weighted_score_matrix(trip, idx, static_score, beta):
    weighted_score = []
    distance_weight = calculate_distance_weight(trip, idx, beta)
    for i in range(len(distance_weight)):
        weighted_score.append(static_score[i] * distance_weight[i])
    return distance_weight, weighted_score


def update_weighted_score_matrix(weighted_score, point_idx, candi_idx):
    import numpy as np
    new_weighted_score = []
    for i in range(len(weighted_score)):
        if i != point_idx - 1:
            new_weighted_score.append(weighted_score[i])
        else:
            ones = np.ones(weighted_score[point_idx - 1].shape, dtype=float)
            ones = ones * -100000000.0
            ones[:, candi_idx] = weighted_score[point_idx - 1][:, candi_idx]
            new_weighted_score.append(ones)
    return new_weighted_score


def find_sequence(candidates, distance_weight, weighted_score, point_idx, candi_idx):
    # forward search
    f = []
    if point_idx == 0:
        obs_prob = [0 for item in range(len(candidates[0]))]
        obs_prob[candi_idx] = candidates[0].iloc[candi_idx]['observation prob']
        # f.append(list(candidates[0]['observation prob']))
        f.append(obs_prob)
    else:
        w = distance_weight[0]
        f.append([item * w for item in candidates[0]['observation prob']])
    pre = [[]]
    for i in range(1, len(candidates)):
        f_i = []
        pre_i = []
        for j in range(len(candidates[i])):
            f_max = -100000000.0
            parent_ind = 0
            for k in range(len(candidates[i-1])):
                alt = f[i-1][k] + weighted_score[i-1][k][j]
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


def interactive_voting(trip, candidates, static_score, beta):
    import numpy as np
    votes = []
    for i in range(len(candidates)):
        votes.append(np.zeros(len(candidates[i])))
    for i in range(len(candidates)):
        distance_weight, weighted_score = calculate_weighted_score_matrix(trip, i, static_score, beta)
        for k in range(len(candidates[i])):
            updated_weighted_score = update_weighted_score_matrix(weighted_score, i, k)
            path, f, pre = find_sequence(candidates, distance_weight, updated_weighted_score, i, k)
            for j in range(len(path)):
                votes[j][path[j]] = votes[j][path[j]] + 1
    final_optimal = []
    for i in range(len(votes)):
        final_optimal.append(np.argmax(votes[i]))
    return final_optimal, votes


def save_to_file_static_score(filename, static_score):
    with open(filename, 'w') as fWriter:
        for i in range(len(static_score)):
            for j in range(len(static_score[i])):
                for k in range(len(static_score[i][j])):
                    fWriter.write('%f ' % static_score[i][j][k])
                fWriter.write('\n')
            fWriter.write('\n')


def save_to_file_votes(filename, votes):
    with open(filename, 'w') as fWriter:
        for i in range(len(votes)):
            for j in range(len(votes[i])):
                fWriter.write('%d ' % votes[i][j])
            fWriter.write('\n')


def ivmm_mapper(road_graph_utm, gpd_edges_utm, trip, candidates, debug=False):
    mu = 0  # mean, parameter for calculating observation probability
    sigma = 10  # standard deviation, parameter for calculating observation probability
    # calculate observation probability
    calculate_observation_probability(candidates, mu, sigma)
    # calculate weights
    weights = calculate_weights(road_graph_utm, gpd_edges_utm, trip, candidates)
    static_score = calculate_static_score_matrix(weights)
    # save_to_file_static_score('debug_ivmm_static_score.txt', static_score)
    # finding the optimal path
    beta = 7000
    optimal_path, votes = interactive_voting(trip, candidates, static_score, beta)
    if debug:
        import os
        cur_dir = os.getcwd()
        debug_dir = os.path.join(cur_dir, r'debug_results')
        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)
        save_to_file_static_score(
            os.path.join(debug_dir, r'debug_ivmm_static_score.txt'), static_score)
        save_to_file_votes(
            os.path.join(debug_dir, r'debug_ivmm_votes.txt'), votes)
    return optimal_path, weights
