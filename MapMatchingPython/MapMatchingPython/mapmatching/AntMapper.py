# -*- coding: utf-8 -*
"""
[1]Y. J. Gong, E. Chen, X. Zhang, L. M. Ni, and J. Zhang,
“AntMapper: An Ant Colony-Based Map Matching Approach for Trajectory-Based Applications,”
IEEE Transactions on Intelligent Transportation Systems,
vol. 19, no. 2, pp. 390–401, Feb. 2018.

"""


import math
import pandas as pd
from NetworkDistance import network_distance


def heuristic_normalization(heuristic):
    return 2*math.atan(heuristic)/math.pi


def heading_difference(A, B):
    d = max(A, B)-min(A, B)
    if d > 180:
        d = 360-d
    return d


# compute parameter mu
def compute_mu(candidates):
    sum_nearest_dist = 0.0
    for i in range(len(candidates)):
        sum_nearest_dist = sum_nearest_dist + candidates[i].iloc[0]['distance']
    return sum_nearest_dist/(len(candidates))


# geometric error
def calculate_geometric_error(s, t, mu):
    """
    input: s: a candidate of the i-th point
           t: a candidate of the i+1-th point
           mu: mean error parameter
    """
    return max(s['distance'], t['distance'])/mu


def calculate_heading_error(h1, h2, s, t):
    """
    input: h1: the heading direction of the i-th point
           h2: the heading direction of the i+1-th point
           s: a candidate of the i-th point
           t: a candidate of the i+1-th point
    """
    sai_1 = s['road']['bearing']
    sai_2 = t['road']['bearing']
    return max(heading_difference(h1, sai_1), heading_difference(h2, sai_2))


def calculate_heuristics_one_step(road_graph_utm, gpd_edges_utm, trip, candidates, idx, mu, epsilon):
    """
    calculate weights between the idx and the idx+1 th points's candidates
    input:
    idx: the idx-th sampling point (idx >= 0)
    """
    # the great circle distance (euclidean distance) between the idx-1 and the idx-th sampling points
    great_circle_distance = trip.iloc[idx]['geometry'].distance(trip.iloc[idx+1]['geometry'])
    # print great_circle_distance
    # the time gap between the idx-1 and the idx-th sampling points
    # delta = trip.iloc[idx+1]['timestamp'] - trip.iloc[idx]['timestamp']
    weights = pd.DataFrame(columns=('from_id','to_id', 'sp distance', 'gc distance', 'sp edges', 'heuristic'))
    for i in range(len(candidates[idx])):
        for j in range(len(candidates[idx+1])):
            # geometric error h1
            h1 = calculate_geometric_error(candidates[idx].iloc[i], candidates[idx+1].iloc[j], mu)
            # heading error h2
            h2 = 1
            if 'heading' in trip.columns:
                phi_1 = trip.iloc[idx]['heading']
                phi_2 = trip.iloc[idx+1]['heading']
                h2 = calculate_heading_error(phi_1, phi_2, candidates[idx].iloc[i], candidates[idx+1].iloc[j])
            # routing error h3
            sp_distance, sp_edges = network_distance(road_graph_utm,
                                                     gpd_edges_utm,
                                                     candidates[idx].iloc[i],
                                                     candidates[idx+1].iloc[j])
            if great_circle_distance < 0.000000001:
                great_circle_distance = 0.000000001
            h3 = sp_distance/great_circle_distance
            h = (h1+epsilon) * h2 * (h3+epsilon)
            s = pd.Series({'from_id': i,
                           'to_id': j,
                           'sp distance': sp_distance,
                           'gc distance': great_circle_distance,
                           'sp edges': sp_edges,
                           'heuristic': h})
            weights = weights.append(s, ignore_index=True)
    weights[['from_id', 'to_id']] = weights[['from_id', 'to_id']].astype(int)
    # ranking heuristics
    weights['heuristic_rank'] = weights['heuristic'].rank(method='first', ascending=False)
    # normalization
    weights['heuristic_norm'] = weights['heuristic_rank'].apply(heuristic_normalization)
    return weights


def calculate_heuristics(road_graph_utm, gpd_edges_utm, trip, candidates, mu, epsilon):
    heuristics = []
    for i in range(len(trip)-1):
        heuristics_i = calculate_heuristics_one_step(road_graph_utm, gpd_edges_utm, trip, candidates, i, mu, epsilon)
        heuristics.append(heuristics_i)
    return heuristics


# feature vector of a matched path (candidate path)
def calculate_feature_vector_matched(trip, candidates, heuristics, path, x_min, y_min):
    feature = []
    if type(path) != list:
        raise TypeError("the path parameter should be a list!")
    if len(candidates) != len(path):
        print len(path)
        raise ValueError('trip and matched path has different length!')
    for i in range(len(path)):
        p = candidates[i].iloc[path[i]]['proj_point']
        feature.append(p.x-x_min)
        feature.append(p.y-y_min)
    if 'velocity' in trip:
        rho = []
        sum_rho = 0.0
        for i in range(len(path)-1):
            ind = path[i]*len(candidates[i+1]) + path[i+1]
            d = heuristics[i].iloc[ind]['sp distance']
            sum_rho = sum_rho + d
            rho.append(d)
        rho.append(sum_rho)
        feature.extend(rho)
    return feature


def calculate_minimum_ratio(X, Y):
    if (type(X) != list) or (type(Y) != list):
        raise TypeError("Only lists are supported as arguments")
    if len(X) != len(Y) or len(X) == 0:
        raise ValueError('X and Y should have the same length')
    s_value = 0.0
    for i in range(len(X)):
        min_v = min(X[i], Y[i])
        max_v = max(X[i], Y[i])
        if max_v > 0.0:
            s_value = s_value + min_v/max_v
        else:
            raise ValueError('X or Y contains invalid values')
    return s_value/len(X)


def calculate_fitness(trip, candidates, trip_feature_vector, heuristics, path, x_min, y_min):
    if type(path) != list:
        raise TypeError("the path parameter should be a list!")
    first_item = 0.0
    for i in range(len(path)-1):
        ind = path[i]*len(candidates[i+1]) + path[i+1]
        first_item = first_item + heuristics[i].iloc[ind]['heuristic_norm']
    first_item = first_item/(len(path)-1)
    path_feature_vector = calculate_feature_vector_matched(trip, candidates, heuristics, path, x_min, y_min)
    second_item = calculate_minimum_ratio(trip_feature_vector, path_feature_vector)
    return (first_item + second_item)*0.5


# the transition probability is related to the heuristics and current pheromone
# of links
# return a list of transition probabilities that from the ind-th candidate of
# point i to all candidates of point i+1
def calculate_transition_probability(heuristics, pheromone, ind, s, size_j, beta):
    """
    ind: the index of the i-th
    s: index of the i-th selected candidate
    size_i: number of candidates of the i-th point
    size_j: number of candidates of the i+1-th point
    beta: the exponential coefficient
    """
    import math
    J = []
    for j in range(size_j):
        ind_hnorm = s * size_j + j
        J.append(heuristics[ind].iloc[ind_hnorm]['heuristic_norm'] * math.pow(pheromone[ind][s][j], beta))
    sum_J = sum(J)
    for j in range(len(J)):
        J[j] = J[j] / sum_J
    return J


# an ant chooses its next state according to the transition probability and
# the random proportional selection rule
def choose_next_state(J, q0):
    import random
    import numpy
    q = random.uniform(0, 1)
    # print q
    if q <= q0:
        return numpy.argmax(J)
    else:
        return random.sample(range(len(J)), 1)[0]


# update the link's pheromone right after an ant choose the link
# this update will be done after each ant jump to its next state
def update_pheromone_locally(pheromone, ind, s, t, tao_0, rho):
    # print s, t
    tao = pheromone[ind][s][t]
    tao_new = (1 - rho) * tao + rho * tao_0
    pheromone[ind][s][t] = tao_new
    # debug
    # if tao_new < tao:
    #    print 'local update: %f, %f' % (tao, tao_new)


# update all links' pheromones that on the best so far path
# this update will be done after each iteration
def update_pheromone_globally(pheromone, path, fitness, alpha):
    for i in range(len(path) - 1):
        tao = pheromone[i][path[i]][path[i + 1]]
        tao_new = (1 - alpha) * tao + alpha * fitness
        pheromone[i][path[i]][path[i + 1]] = tao_new
        # debug
        if tao_new < tao:
            print 'global update: %f, %f' % (tao, tao_new)


def ant_optimize(trip, candidates, trip_feature_vector, heuristics,
                 x_min, y_min, num_ants, num_iterations, tao_0, beta, q_0, rho,
                 alpha, debug=False):
    import random
    import numpy
    # initialize pheromone
    pheromone_values = []  #
    for i in range(len(trip) - 1):
        plist_i = []
        for j in range(len(candidates[i])):
            plist_i_k = [tao_0 for k in range(len(candidates[i + 1]))]
            plist_i.append(plist_i_k)
        pheromone_values.append(plist_i)
    # main loop
    g = 0  # iteration counts
    while g < num_iterations:
        # randomly choose a start position for each ant
        path_list = [random.sample(range(len(candidates[0])), 1) for ant in range(num_ants)]
        for i in range(1, len(candidates)):
            # transit each ant to next state according to the state transition rule, i.e., determine the next position
            for ant in range(num_ants):
                # print i, ant
                J_values = calculate_transition_probability(heuristics,
                                                            pheromone_values,
                                                            i - 1,
                                                            path_list[ant][i - 1],
                                                            len(candidates[i]),
                                                            beta)
                next_state = choose_next_state(J_values, q_0)
                # print next_state
                # print J_values
                path_list[ant].append(next_state)
                # update pheromone of the link passed by each ant at current step
                update_pheromone_locally(pheromone_values, i - 1, path_list[ant][i - 1], next_state, tao_0, rho)

        # calculate fitness value of the path passed by each ant
        fitness_value_list = [calculate_fitness(trip, candidates, trip_feature_vector, heuristics,
                                                path_list[ant], x_min, y_min)
                              for ant in range(num_ants)]
        # find the best so far path from all the ants' path
        path_bsf_ind = numpy.argmax(fitness_value_list)
        if debug:
            print fitness_value_list[path_bsf_ind], path_list[path_bsf_ind]
        # globally update the pheromone of links on the best so far path
        update_pheromone_globally(pheromone_values, path_list[path_bsf_ind], fitness_value_list[path_bsf_ind], alpha)
        g = g + 1
    return fitness_value_list[path_bsf_ind], path_list[path_bsf_ind]


# feature vector of the original GPS Trace
def calculate_feature_vector(trip, x_min, y_min):
    feature = []
    for i in range(len(trip)):
        feature.append(trip.iloc[i]['geometry'].x - x_min)
        feature.append(trip.iloc[i]['geometry'].y - y_min)
    if 'velocity' in trip:
        r = []
        sum_r = 0.0
        for i in range(len(trip)-1):
            delta_t = trip.iloc[i+1]['timestamp']-trip.iloc[i]['timestamp']
            r_i = (trip.iloc[i]['velocity']*0.25 + trip.iloc[i+1]['velocity']*0.25) * delta_t
            sum_r = sum_r + r_i
            r.append(r_i)
        r.append(sum_r)
        feature.extend(r)
    return feature


# all steps in one function
def ant_mapper(road_graph_utm, gpd_edges_utm, trip, candidates, x_min, y_min):
    """
    Ant Mapper
    :param road_graph_utm: a networkx digraph, road network
    :param gpd_edges_utm: a geopandas GeoDataFrame, road edges
    :param trip: a geopandas GeoDataFrame, gps trajectory
    :param candidates: a list of pandas DataFrame, candidate edges and projection points of each gps point,
                        column names ['distance', 'from', 'to', 'proj_point', 'road']
    :param x_min: a float, the minimum value of the x-coordinate
    :param y_min: a float, the minimum value of the y-coordinate
    :return:
    """
    # step 3 computing heuristics
    mu = compute_mu(candidates)
    EPSILON = 0.01  # a small constant to avoid the situation where a zero factor
    heuristics = calculate_heuristics(road_graph_utm, gpd_edges_utm, trip, candidates, mu, EPSILON)

    # step 4 ant colony optimized map matching
    trip_feature_vector = calculate_feature_vector(trip, x_min, y_min)
    # get the nearest neighbor path
    pnn = [0 for i in range(len(trip))]
    # calculate the initial pheromone value
    tao_0 = calculate_fitness(trip, candidates, trip_feature_vector, heuristics, pnn, x_min, y_min) / len(trip)
    U = 10  # the ant colony size
    BETA = 0.5  # the amplification coefficient
    Q_0 = 0.9  # the probability in the state transition
    ALPHA = 0.1  # global evaporation rate
    RHO = 0.1  # local evaporation rate
    G = 100  # the number of iterations
    fitness, optimal_path = ant_optimize(trip, candidates, trip_feature_vector, heuristics,
                                         x_min, y_min, U, G, tao_0, BETA, Q_0, RHO, ALPHA)
    return optimal_path, heuristics

