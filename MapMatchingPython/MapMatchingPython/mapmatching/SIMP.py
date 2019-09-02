# -*- coding: utf-8 -*
"""
[1]H. Li, L. Kulik, and K. Ramamohanarao,
“Spatio-temporal Trajectory Simplification for Inferring Travel Paths,”
in Proceedings of the 22Nd ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems,
New York, NY, USA, 2014, pp. 63–72.

"""


from shapely.geometry import Point, LineString
import geopandas as gpd
from HMM import calculate_observation_probability, calculate_weights
from HMM import find_optimal_path
from HMM import save_to_file_pre, save_to_file_fscore, save_to_file_weights, save_to_file_observation_probabilities


def simplify_trip(trip, tolerance):
    """
    trajectory simplification (before computing candidates)
    All points in the simplified object will be within the tolerance distance of the original geometry.
    By default a slower algorithm is used that preserves topology.
    If preserve topology is set to False the much quicker Douglas-Peucker algorithm is used.
    :param trip: the original trajectory
    :param tolerance:  a distance threshold
    :return:  the simplified trajectory
    """
    line = LineString(list(trip['geometry_utm']))
    simplified_line = line.simplify(tolerance, preserve_topology=False)
    simplified_trip = gpd.GeoDataFrame(columns=trip.columns)
    k = 0
    for t in simplified_line.coords:
        for i in range(k, len(trip)):
            if trip.iloc[i]['geometry_utm'] == Point(t):
                simplified_trip = simplified_trip.append(trip.iloc[i], ignore_index=True)
                k = i+1
                break
    return simplified_trip


def filter_rect(trip, candidates, epsilon):
    """
    remove points that are far away from the road network (after computing candidates)
    :param trip: the original trajectory
    :param candidates: candidates of the original trajectory
    :param epsilon: a distance threshold (=10)
    :return: the filtered trajectory and its corresponding candidates
    """
    filterd_trip = gpd.GeoDataFrame(columns=trip.columns)
    new_candidates = []
    for i in range(len(trip)):
        if candidates[i].iloc[0]['distance'] < epsilon:
            filterd_trip = filterd_trip.append(trip.iloc[i], ignore_index=True)
            new_candidates.append(candidates[i])
    return filterd_trip, new_candidates


def simp_mapper(road_graph_utm, gpd_edges_utm, trip, candidates, debug=False):
    mu = 0  # mean, parameter for calculating observation probability
    sigma = 10  # standard deviation, parameter for calculating observation probability
    # simplification
    epsilon = 10
    filtered_trip, filtered_candidates = filter_rect(trip, candidates, epsilon)
    # calculate observation probability
    calculate_observation_probability(filtered_candidates, mu, sigma)
    # calculate weights
    beta = 1000
    weights = calculate_weights(road_graph_utm, gpd_edges_utm, filtered_trip, filtered_candidates, beta)
    # finding the optimal path (viterbi algorithm)
    optimal_path, f_score, pre = find_optimal_path(filtered_candidates, weights)
    if debug:
        import os
        cur_dir = os.getcwd()
        debug_dir = os.path.join(cur_dir, r'debug_results')
        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)
        save_to_file_observation_probabilities(
            os.path.join(debug_dir, r'simp_observation_probabilities.txt'), filtered_candidates)
        save_to_file_weights(
            os.path.join(debug_dir, r'simp_weights.txt'), weights)
        save_to_file_fscore(
            os.path.join(debug_dir, r'simp_fscore.txt'), f_score)
        save_to_file_pre(
            os.path.join(debug_dir, r'simp_pre.txt'), pre)
    return optimal_path, weights, filtered_candidates
