# -*- coding: utf-8 -*


import os
from RoadNetwork import load_road_network_graphml, load_road_network_seattle, load_road_network_melbourne
from Trip import load_gps_data_seattle, load_gps_data_porto, load_gps_data_melbourne
from Rtree import find_candidates, build_rtree_index_edges
from AntMapper import ant_mapper
from Visualize import visualize_matching_results
from STMatching import st_mapper
from IVMM import ivmm_mapper
from HMM import hmm_mapper
from SIMP import simp_mapper
from OBRHMM import obr_mapper_v1



def get_optimal_path_edges(candidates, weights, optimal_path, debug=False):
    edge_list = []
    for i in range(len(candidates)-1):
        to_id = optimal_path[i+1]
        from_id = optimal_path[i]
        ind = from_id * len(candidates[i+1]) + to_id
        sub_edge_list = weights[i].iloc[ind]['sp edges']
        # if len(edge_list)>0:
        #    if edge_list[-1]['geometry'] != sub_edge_list[0]['geometry']:
        #        print 'path broken!'
        if debug:
            print i
        for j in range(len(sub_edge_list)-1):
            edge_list.append(sub_edge_list[j])
            if debug:
                print sub_edge_list[j]['from'], sub_edge_list[j]['to']
        if i == len(candidates)-1:
            edge_list.append(sub_edge_list[-1])
            if debug:
                print sub_edge_list[-1]['from'], sub_edge_list[-1]['to']
    edge_id_list = []
    for edge in edge_list:
        edge_id_list.append(edge.name)
    return edge_list, edge_id_list


def save_to_file_matching_result(filename, opt_route):
    with open(filename, 'w') as fWriter:
        fWriter.write('%d\n' % len(opt_route))
        for i in range(len(opt_route)):
            fWriter.write('%d\n' % opt_route[i])


def map_osm_edge_id(edges_gpd, opt_route):
    opt_route_osm_edge_id = []
    for edge_id in opt_route:
        opt_route_osm_edge_id.append((edges_gpd.iloc[edge_id]['osm_edge_id'], edges_gpd.iloc[edge_id]['from_to']))
    return opt_route_osm_edge_id


def save_to_file_matching_result_seattle(filename, opt_route):
    with open(filename, 'w') as fWriter:
        fWriter.write('%d\n' % len(opt_route))
        for i in range(len(opt_route)):
            fWriter.write('%d\t%d\n' % (opt_route[i][0], opt_route[i][1]))


def map_matching_test(data_name, algo_name):
    """
    :param data_name: trajectory data name
    :param algo_name: algorithm name
    :return:
    """
    print('Set data name as %s' % data_name)
    print('Set algorithm name as %s' % algo_name)
    crs = {'init': 'epsg:4326'}
    to_crs = {'init': 'epsg:3395'}
    optimal_path = []
    candidates = []
    weights = []
    if data_name is 'Seattle':

        road_file = 'D:/zongshu/MapMatchingPython/MapMatchingPython/data/Seattle/road_network.txt'
        trip_file = 'D:/zongshu/MapMatchingPython/MapMatchingPython/data/Seattle/gps_data.txt'
        road_graph_utm, gpd_edges_utm = load_road_network_seattle(road_file, crs, to_crs)
        trip = load_gps_data_seattle(trip_file, crs, to_crs)
       
       
    elif data_name is 'Melbourne':
        road_file = 'D:/zongshu/MapMatchingPython/MapMatchingPython/data/Melbourne/complete-osm-map/streets.txt'
        trip_file = 'D:/zongshu/MapMatchingPython/MapMatchingPython/data/Melbourne/gps_track.txt'
        road_graph_utm, gpd_edges_utm = load_road_network_melbourne(road_file, crs, to_crs)
        trip = load_gps_data_melbourne(trip_file, crs, to_crs)
  

    elif data_name is 'Porto':
        road_file = 'porto.graphml'
        road_folder = 'D:/zongshu/MapMatchingPython/MapMatchingPython/data/Porto'
        trip_folder = 'D:/zongshu/MapMatchingPython/MapMatchingPython/data/Porto/trips'
        trip_file = 'trip_1.txt'
        road_graph_utm, gpd_edges_utm, wgs_crs, utm_crs = load_road_network_graphml(road_folder, road_file)
        trip = load_gps_data_porto(trip_folder+'/'+trip_file, wgs_crs, utm_crs)
        sample_rate=trip.iloc[1]["timestamp"]-trip.iloc[0]["timestamp"]

    else:
        print('Unknown data name!\n')

    # print('The trip data and road network are loaded!')
        # finding candidates for each gps points using knn query
    edge_idx = build_rtree_index_edges(gpd_edges_utm)

    if algo_name in ['HMM', 'ST', 'IVMM', 'Ant', 'SIMP']:
        # print('The edge r-tree index prepared!')
        k = 3  # number of candidate points of each gps points
        candidates = find_candidates(trip, edge_idx, k)
        # print('Candidates prepared!')

    if algo_name is 'Ant':
        print('*******Ant Mapper********')
        x_min, y_min, x_max, y_max = gpd_edges_utm.total_bounds  # bounding box of the road network
        optimal_path, weights = ant_mapper(road_graph_utm, gpd_edges_utm, trip, candidates, x_min, y_min)
    elif algo_name is 'ST':
        print('*******ST Map-Matching*********')
        optimal_path, weights = st_mapper(road_graph_utm, gpd_edges_utm, trip, candidates, True)
    elif algo_name is 'IVMM':
        print('*******IVMM Map-Matching*******')
        optimal_path, weights = ivmm_mapper(road_graph_utm, gpd_edges_utm, trip, candidates, True)
    elif algo_name is 'HMM':
        print('*******HMM Map-Matching**********')
        optimal_path, weights = hmm_mapper(road_graph_utm, gpd_edges_utm, trip, candidates, True)
    # visualize_matching_results(trip, candidates, edge_idx, weights_st, optimal_path_st, fig_name)
    elif algo_name is 'OBRHMM':
        print('*******OBRHMM Map-Matching*******')
        if data_name is 'Seattle':
            d_error = 20  # error range of gps positioning
        elif data_name is 'Melbourne':
            d_error = 10
        a_error = 45  # angle difference bound
        edge_id_list = obr_mapper_v1(road_graph_utm, gpd_edges_utm, trip, d_error, a_error, True)
        # print optimal_path
    elif algo_name is 'SIMP':
        print('******SIMP Map-Matching*********')
        optimal_path, weights, filtered_candidates = \
            simp_mapper(road_graph_utm, gpd_edges_utm, trip, candidates, debug=False)
        candidates = filtered_candidates
    else:
        print('Unknown algorithm name!\n')

    # print optimal_path
    if algo_name in ['HMM', 'ST', 'IVMM', 'Ant', 'SIMP']:
        edge_list, edge_id_list = get_optimal_path_edges(candidates, weights, optimal_path)

    # seq = [data_name, sample_rate,algo_name, 'matching_result.txt']
    seq = [data_name,algo_name, 'matching_result.txt']
    Path="D:/zongshu/MapMatchingPython/MapMatchingPython/mapmatching/output"
    connect_str = '_'
    filename = os.path.join(Path,connect_str.join(seq))
    if data_name is 'Seattle':
        opt_route_osm_edge_id = map_osm_edge_id(gpd_edges_utm, edge_id_list)
        save_to_file_matching_result_seattle(filename, opt_route_osm_edge_id)
    else:
        save_to_file_matching_result(filename, edge_id_list)


# data name includes:
# ['Seattle', 'Melbourne', 'Porto']
# algorithms name includes:
# ['HMM', 'ST', 'IVMM', 'Ant', 'SIMP', 'OBRHMM']


# map_matching_test('Seattle', 'OBRHMM')
# map_matching_test('Seattle', 'SIMP')
# map_matching_test('Melbourne', 'OBRHMM')
# map_matching_test('Melbourne', 'SIMP')
map_matching_test('Melbourne', 'Ant')

# map_matching_test('Porto', 'Ant')
# map_matching_test('Porto', 'IVMM')
# map_matching_test('Porto', 'ST')
# map_matching_test('Porto', 'HMM')
# map_matching_test('Porto', 'OBRHMM')
visualize_matching_results(trip, candidates, edge_idx, weights, optimal_path, figname='temp.pdf')

