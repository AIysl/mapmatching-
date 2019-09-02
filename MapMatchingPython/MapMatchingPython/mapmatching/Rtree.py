#from rtree import index
import rtree
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import pandas as pd


def build_rtree_index_edges(gpd_edges):
    """
    build a r-tree index for road segments
    input:
        gpd_edges: a geopandas dataframe that contains road segments (edge geometries)
    output:
        idx: a r-tree index of the edge geometries
    """
    # r-tree index for edges
    p = rtree.index.Property()
    idx = rtree.index.Index(properties=p)
    for index, row in gpd_edges.iterrows():
        idx.insert(index, row['bbox'], obj=row)
    return idx


def query_k_nearest_road_segments(edge_idx, point, k):
    """
    query k-nearest road segments of a given point
    :param edge_idx: the road segments r-tree index
    :param point: the given point
    :param k: the number of segments needed to query
    :return: k candidates as a pandas DataFrame
    """
    candidates = pd.DataFrame(columns=('distance', 'from', 'to', 'proj_point', 'road'))
    hits = edge_idx.nearest((point.x, point.y, point.x, point.y), k, objects=True)
    for item in hits:
        results = nearest_points(point, item.object['geometry'])
        d = point.distance(results[1])
        s = pd.Series({'distance': d,
                       'from': item.object['from'],
                       'to': item.object['to'],
                       'proj_point': results[1],
                       'road': item.object})
        candidates = candidates.append(s, ignore_index=True)
    # candidates['observation prob'] = candidates.apply(lambda row: normal_distribution())
    candidates.sort_values(by='distance', axis=0, inplace=True)
    return candidates


def query_edges_by_range(edge_index, point, diameter):
    """
    point range query
    :param edge_index: a R-tree index of road edges
    :param point: a shapely.geometry.Point
    :param diameter: the circle diameter of the range query circle
    :return:
    """
    kk = 3
    # query_results = pd.DataFrame()
    edge_id_list = []
    while True:
        query_results = query_k_nearest_road_segments(edge_index, point, kk)
        if max(list(query_results['distance'])) > diameter:
            break
        else:
            kk = kk + 2
    for i in range(len(query_results)):
        if query_results.iloc[i]['distance'] < diameter:
            edge_id_list.append(query_results.iloc[i]['road']['Edge_ID'])
    return edge_id_list


def find_candidates(trip, edge_idx, k):
    """
    given a trip, find candidates points for each point in the trip
    :param trip: a GPS trajectory (without coordinates transform)
    :param edge_idx: road segments r-tree index of the corresponding road network
    :param k: the number of candidates
    :return: a list, each element is a list of corresponding candidates
    """
    candi_list = []
    for i in range(len(trip)):
        candidates = query_k_nearest_road_segments(edge_idx, trip.iloc[i]['geometry'], k)
        candi_list.append(candidates)
    return candi_list


#K = 6
#candidates = find_candidates(trip, edge_idx, K)
#candidates[0]
