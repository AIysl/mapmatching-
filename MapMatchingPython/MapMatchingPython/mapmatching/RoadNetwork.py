# -*- coding: utf-8 -*

import math
import osmnx as ox
import pandas as pd
import geopandas as gpd
import shapely.wkt
from shapely.geometry import Point, LineString
import networkx as nx


def road_graph_to_edge_gpd(road_graph):
    '''
    store road segments into a geppandas dataframe
    :param road_graph: a networkx graph object to store road network
    :return gpd_edges: a geopandas dataframe to store road segments
    '''
    gpd_edges = gpd.GeoDataFrame(columns=('from', 'to', 'geometry', 'length', 'highway'))
    for e_from, e_to, data in road_graph.edges(data=True):
        if 'geometry' in data:
            s = gpd.GeoSeries({'from': e_from,
                               'to': e_to,
                               'geometry': data['geometry'],
                               'length': data['length'],
                               'highway': data['highway']})
            gpd_edges = gpd_edges.append(s, ignore_index=True)
        else:
            p1 = Point(road_graph.nodes[e_from]['x'], road_graph.nodes[e_from]['y'])
            p2 = Point(road_graph.nodes[e_to]['x'], road_graph.nodes[e_to]['y'])
            data.update({'geometry': LineString((p1, p2))})
            s = gpd.GeoSeries({'from': e_from,
                               'to': e_to,
                               'geometry': LineString((p1, p2)),
                               'length': data['length'],
                               'highway': data['highway']})
            gpd_edges = gpd_edges.append(s, ignore_index=True)
    gpd_edges.crs = road_graph.graph['crs']
    gpd_edges.name = 'edges'
    # create bounding box for each edge geometry
    gpd_edges['bbox'] = gpd_edges.apply(lambda row: row['geometry'].bounds, axis=1)
    return gpd_edges


def get_max_speed(highway):
    '''
    return the corresponding max speed in kmph
    '''
    if highway == 'mortorway':
        return 100
    elif highway == 'mortorway_link':
        return 60
    elif highway == 'trunk':
        return 80
    elif highway == 'trunk_link':
        return 40
    elif highway == 'primary':
        return 60
    elif highway == 'primary_link':
        return 40
    elif highway == 'secondary':
        return 50
    elif highway == 'secondary_link':
        return 20
    elif highway == 'residential':
        return 30
    elif highway == 'teritiary':
        return 50
    elif highway == 'teritiary_link':
        return 20
    elif highway == 'living_street':
        return 20
    elif highway == 'road':
        return 20
    elif highway == 'service':
        return 20
    else:
        return 50


# add the speed limits information to the network
def add_max_speeds(gpd_edges_utm):
    max_speeds = []
    for idx, row in gpd_edges_utm.iterrows():
        if isinstance(row['highway'], list):
            max_speed1 = get_max_speed(row['highway'][0])
            max_speed2 = get_max_speed(row['highway'][1])
            if row['length'] > 100:
                max_speed = max(max_speed1, max_speed2)
                max_speeds.append(max_speed)
            else:
                max_speed = min(max_speed1, max_speed2)
                max_speeds.append(max_speed)
        else:
            max_speeds.append(get_max_speed(row['highway']))
    return max_speeds


def calculate_bearing(pt1, pt2):
    """
    calculate bearing of the segment (pt1, pt2)
    :param pt1: a shapely.geometry.Point, a utm coordinate
    :param pt2: a shapely.geometry.Point, a utm coordinate
    :return: the bearing degree, float range in [0, 360)
    """
    import math
    x_diff = pt2.x - pt1.x
    y_diff = pt2.y - pt1.y
    bearing = math.degrees(math.atan2(y_diff, x_diff))
    if bearing < 0:
        return bearing + 360
    else:
        return bearing


def calculate_initial_compass_bearing(pointA, pointB):
    """
    URL：https://gist.github.com/jeromer/2005586
    Calculates the bearing between two points.
    The formulae used is the following:
     theta = atan2(sin(delta(long)).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(delta(long)))
    :Parameters:
     `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
     `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])
    diffLong = math.radians(pointB[1] - pointA[1])
    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))
    initial_bearing = math.atan2(x, y)
    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing


def calculate_edge_bearing(road_graph, from_id, to_id):
    if 'geometry' in road_graph[from_id][to_id][0]:
        linestring = road_graph[from_id][to_id][0]['geometry']
        p1 = linestring.coords[0]
        p2 = linestring.coords[-1]
        pA = (p1[1], p1[0])
        pB = (p2[1], p2[0])
        return calculate_initial_compass_bearing(pA, pB)
    else:
        pA = (road_graph.nodes[from_id]['y'], road_graph.nodes[from_id]['x'])
        pB = (road_graph.nodes[to_id]['y'], road_graph.nodes[to_id]['x'])
        return calculate_initial_compass_bearing(pA, pB)


def add_edge_bearing(road_graph, gpd_edges_utm):
    gpd_edges_utm['bearing'] = gpd_edges_utm.apply(
        lambda row: calculate_edge_bearing(road_graph, row['from'], row['to']), axis=1)


# load road network from .graphml file
def load_road_network_graphml(road_folder, road_filename):
    """
    :param road_folder:
    :param road_filename:
    :return: road network, WGS crs, UTM crs
    """
    road_graph = ox.load_graphml(filename=road_filename, folder=road_folder)
    # WGS to UTM projection
    road_graph_utm = ox.project_graph(road_graph)
    # convert osmnx typed road network to geopandas dataframe storing edges
    gpd_edges_utm = road_graph_to_edge_gpd(road_graph_utm)
    # add speed limits information
    gpd_edges_utm['max speed'] = add_max_speeds(gpd_edges_utm)
    # add road segments bearings information
    add_edge_bearing(road_graph, gpd_edges_utm)
    return road_graph_utm, gpd_edges_utm, road_graph.graph['crs'], road_graph_utm.graph['crs']


def remove_edges_with_single_point(road):
    index_list = []
    for row_index, row in road.iterrows():
        if row['geometry'].coords[0] == row['geometry'].coords[-1] and 2 == len(row['geometry'].coords):
            index_list.append(row_index)
    road.drop(index_list, inplace=True)
    # road.reset_index()
    print('Removed edges with single point: ')
    print index_list
    # column_names = ['Edge_ID', 'from', 'to', 'two_way', 'max speed', 'vertex_count', 'geometry']
    # return pd.DataFrame(road.values, columns=column_names )


def get_nodes_id_coordinate_dict(road):
    nodes_dict = {}  # key: node_id, value: a list (the gps coordinate of the node, edge_ids....)
    for row_index, row in road.iterrows():
        start_point = row['geometry'].coords[0]
        start_node_id = row['from']
        if nodes_dict.has_key(start_node_id):
            if nodes_dict[start_node_id][0] != start_point:
                print(start_node_id, nodes_dict[start_node_id], (start_point, row_index))
            nodes_dict[start_node_id].append(row_index)
        else:
            nodes_dict[start_node_id] = [start_point, row_index]
        end_point = row['geometry'].coords[-1]
        end_node_id = row['to']
        if nodes_dict.has_key(end_node_id):
            if nodes_dict[end_node_id][0] != end_point:
                print(end_node_id, nodes_dict[end_node_id], (end_point, row_index))
            nodes_dict[end_node_id].append(row_index)
        else:
            nodes_dict[end_node_id] = [end_point, row_index]
    print('There are %d nodes in the road network!' % len(nodes_dict))
    return nodes_dict


def get_nodes_coordinate_id_dict(nodes_dict):
    point_id_dict = {}  # key: a gps coordinate, value: a tuple (edge_id, node_id)
    for node_id, value in nodes_dict.items():
        point = value[0]
        edge_index_list = value[1:]
        if point_id_dict.has_key(point):
            point_id_dict[point].append((edge_index_list, node_id))
        else:
            point_id_dict[point] = [(edge_index_list, node_id)]
    return point_id_dict


def get_nodes_with_different_ids(road):
    nodes_dict = get_nodes_id_coordinate_dict(road)
    point_id_dict = get_nodes_coordinate_id_dict(nodes_dict)
    nodes_with_different_ids = []
    # id_dict = {}
    for point, value in point_id_dict.items():
        if len(value) > 1:
            nodes_with_different_ids.append(value)
            # for i in range(len(value)):
            #    id_list = []
            #    for j in range(len(value)):
            #        if j != i:
            #            id_list.append(value[j])
            #    id_dict[value[i][1]] = id_list
    return nodes_with_different_ids  # , id_dict


def combine_same_nodes(road, nodes_with_different_ids):
    road_values = road.values
    for i in range(len(nodes_with_different_ids)):
        final_node_id = nodes_with_different_ids[i][0][1]
        ind_list = []
        node_id_ind = 0
        for j in range(len(nodes_with_different_ids[i])):
            edge_id_list = nodes_with_different_ids[i][j][0]
            node_id = nodes_with_different_ids[i][j][1]
            if len(edge_id_list) < 2:
                ind_list.append(j)
            else:
                if len(edge_id_list) > len(nodes_with_different_ids[i][0][0]):
                    final_node_id = node_id
                    node_id_ind = j
        print i, final_node_id, ind_list
        # update node ids
        for k in ind_list:
            if k != node_id_ind:
                node_id = nodes_with_different_ids[i][k][1]
                edge_id_list = nodes_with_different_ids[i][k][0]
                for edge_id in edge_id_list:
                    from_node_id = road.iloc[edge_id]['from']
                    to_node_id = road.iloc[edge_id]['to']
                    # print edge_id, from_node_id, to_node_id, node_id
                    if from_node_id == node_id:
                        road_values[edge_id][1] = final_node_id
                        print('Change %d to %d in edge %d ' % (node_id, final_node_id, edge_id))
                    elif to_node_id == node_id:
                        road_values[edge_id][2] = final_node_id
                        print('Change %d to %d in edge %d ' % (node_id, final_node_id, edge_id))
                    else:
                        print('Node %d is not in edge %d.' % (node_id, edge_id))
    column_names = ['Edge_ID', 'from', 'to', 'two_way', 'max speed', 'vertex_count', 'geometry']
    return pd.DataFrame(road_values, columns=column_names)


def load_road_network_seattle(filename, crs, to_crs):
    """
    prepare Seattle road network data
    :param filename:
    :param crs:
    :param to_crs
    :return:
    """
    from shapely import wkt
    print('loading Seattle Road Network ...')
    # column_names = ['Edge_ID', 'From_Node_ID', 'To_Node_ID', 'Two_Way', 'Speed(m/s)', 'Vertex_Count', 'geometry']
    column_names = ['Edge_ID', 'from', 'to', 'two_way', 'max speed', 'vertex_count', 'geometry']
    road = pd.read_csv(filename, header=None, names=column_names, skiprows=[0], sep='\t')
    road['geometry'] = road.apply(lambda row: wkt.loads(row['geometry']), axis=1)
    # filtering road network data

    # update node ids
    nodes_with_different_ids = get_nodes_with_different_ids(road)
    road = combine_same_nodes(road, nodes_with_different_ids)
    remove_edges_with_single_point(road)
    # print road.values.shape, updated_road.values.shape

    # coordinates transformation
    road = gpd.GeoDataFrame(road, crs=crs, geometry='geometry')
    road.to_crs(to_crs, inplace=True)
    road['length'] = road.apply(lambda row: row['geometry'].length, axis=1)
    road['bbox'] = road.apply(lambda row: row['geometry'].bounds, axis=1)
    # print len(road)
    direct_edges_list = []
    idx = 0
    for row_index, row in road.iterrows():
        direct_edges_list.append([idx, row['Edge_ID'], row['from'], row['to'],
                                  row['max speed'], row['geometry'], row['length'], row['bbox'], 1])
        idx = idx + 1
        if 1 == row['two_way']:
            direct_edges_list.append([idx, row['Edge_ID'], row['to'], row['from'],
                                      row['max speed'], LineString(list(row['geometry'].coords)[::-1]),
                                      row['length'], row['bbox'], 0])
            idx = idx + 1
    edges = pd.DataFrame(
        direct_edges_list,
        columns=('Edge_ID', 'osm_edge_id', 'from', 'to', 'max speed', 'geometry', 'length', 'bbox', 'from_to'))
    edges['bearing'] = \
        edges.apply(lambda edge:
                    calculate_bearing(Point(edge['geometry'].coords[0]), Point(edge['geometry'].coords[1])), axis=1)
    # nodes_with_different_ids, id_dict = get_nodes_with_different_ids(edges)
    road_graph = nx.from_pandas_edgelist(edges,
                                         'from',
                                         'to',
                                         ['Edge_ID', 'max speed', 'geometry', 'length'],
                                         create_using=nx.MultiDiGraph())
    # print len(edges)
    # print len(direct_edges_list)
    return road_graph, gpd.GeoDataFrame(edges, crs=to_crs, geometry='geometry')


def max_speed(road_type):
    if road_type == 1:
        return 100
    elif road_type == 2:
        return 60
    elif road_type == 3:
        return 50
    elif road_type == 4:
        return 50
    elif road_type == 5:
        return 80
    elif road_type == 6:
        return 30
    else:
        return 40


def load_road_network_melbourne(filename, crs, to_crs):
    print('loading Melbourne Road Network ...')
    road = pd.read_csv(filename,
                       header=None,
                       names=['Edge_ID', 'from', 'from lon', 'from lat',
                              'to', 'to lon', 'to lat',
                              'length', 'road type', 'bearing'],
                       skiprows=[0],
                       sep=' ')
    road['geometry'] = road.apply(lambda row:
                                  LineString([(row['from lon'], row['from lat']), (row['to lon'], row['to lat'])]),
                                  axis=1)
    road['max speed'] = road.apply(lambda row: max_speed(row['road type']), axis=1)
    road['gps'] = road['geometry']
    road = gpd.GeoDataFrame(road, crs=crs, geometry='geometry')
    road.to_crs(to_crs, inplace=True)
    road['bbox'] = road.apply(lambda row: row['geometry'].bounds, axis=1)
    road_network_edges = pd.DataFrame(road, columns=(
        'Edge_ID', 'from', 'to', 'gps', 'geometry', 'max speed', 'length', 'bbox'))
    road_graph = nx.from_pandas_edgelist(road_network_edges,
                                         'from',
                                         'to',
                                         ['Edge_ID', 'max speed', 'geometry', 'length'],
                                         create_using=nx.MultiDiGraph())
    return road_graph, gpd.GeoDataFrame(road, crs=to_crs, geometry='geometry')
