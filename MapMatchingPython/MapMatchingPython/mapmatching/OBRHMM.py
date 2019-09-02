# -*- coding: utf-8 -*


"""
[1]H. Li, L. Kulik, and K. Ramamohanarao,
“Robust inferences of travel paths from GPS trajectories,”
International Journal of Geographical Information Science,
vol. 29, no. 12, pp. 2194–2222, Dec. 2015.

"""


from shapely.geometry import Point
import networkx as nx
# import math
from RoadNetwork import calculate_bearing


def compute_density(trip):
    if len(trip) < 1:
        raise ValueError('The length of the trajectory is too short!')
    sum_dist = 0.0
    for i in range(len(trip)-1):
        sum_dist = sum_dist + trip.iloc[i]['geometry_utm'].distance(trip.iloc[i+1]['geometry_utm'])
    return 1.0/(sum_dist/(len(trip)-1))


def heading_difference(a, b):
    d = max(a, b)-min(a, b)
    if d > 180:
        d = 360-d
    return d


# OBR Calculation
def obr(track, d_error, min_num_points):
    """
    divide trajectory into segments
    :param track:
    :param d_error:
    :param min_num_points:
    :return:
    """
    # create initial rectangle
    clusters = []
    outliers = []
    removed = []
    anchor = [False for i in range(len(track))]
    cluster = [0, 1]
    c1 = track.iloc[cluster[0]]['geometry'].buffer(d_error).exterior
    for i in range(2, len(track)):
        c2 = track.iloc[i]['geometry'].buffer(d_error).exterior
        mbr_new = c1.union(c2).minimum_rotated_rectangle
        possible_outliers = []
        for item in cluster[1:]:
            if not mbr_new.contains(track.iloc[item]['geometry']):
                anchor[item] = True
                possible_outliers.append(item)
        # remove outliers
        j = 0
        while j < len(possible_outliers):
            if anchor[possible_outliers[j]-1] is False and anchor[possible_outliers[j]+1] is False:
                outliers.append(possible_outliers[j])
                cluster.remove(possible_outliers[j])
                possible_outliers.remove(possible_outliers[j])
            else:
                j = j+1
        if len(possible_outliers) > 0:
            # divide current cluster into two parts
            ind = cluster.index(possible_outliers[-1])
            first_part = [cluster[k] for k in range(ind+1)]
            second_part = [cluster[k] for k in range(ind, len(cluster))]
            # print possible_outliers
            # print first_part
            # print second_part
            # print '-----'
            if len(first_part) > min_num_points:
                clusters.append(first_part)
            else:
                removed.append(first_part)
            cluster = second_part
            cluster.append(i)
            c1 = track.iloc[cluster[0]]['geometry'].buffer(d_error).exterior
        else:
            cluster.append(i)
    if len(cluster) > min_num_points:
        clusters.append(cluster)
    return clusters, outliers, removed


def calculate_heading_difference(edges_gpd, edge_id, mbr, mbr_heading):
    import shapely
    edge_bearing = edges_gpd.iloc[edge_id]['bearing']
    a = edges_gpd.iloc[edge_id]['geometry'].difference(mbr)
    if isinstance(a, shapely.geometry.multilinestring.MultiLineString):
        p1 = shapely.geometry.Point(list(a.geoms)[0].coords[-1])
        p2 = shapely.geometry.Point(list(a.geoms)[1].coords[0])
        edge_bearing = calculate_bearing(p1, p2)
    elif isinstance(a, shapely.geometry.LineString):
        if a.coords[0] == edges_gpd.iloc[edge_id]['geometry'].coords[0] and \
                a.coords[-1] == edges_gpd.iloc[edge_id]['geometry'].coords[-1]:
            pass
        elif a.coords[0] == edges_gpd.iloc[edge_id]['geometry'].coords[0]:
            p1 = shapely.geometry.Point(a.coords[-1])
            p2 = shapely.geometry.Point(edges_gpd.iloc[edge_id]['geometry'].coords[-1])
            edge_bearing = calculate_bearing(p1, p2)
        else:
            p1 = shapely.geometry.Point(edges_gpd.iloc[edge_id]['geometry'].coords[0])
            p2 = shapely.geometry.Point(a.coords[0])
            edge_bearing = calculate_bearing(p1, p2)
    else:
        if a.is_empty:
            pass
        else:
            print type(a)
    diff = heading_difference(mbr_heading, edge_bearing)
    return diff


def filter_edges(mbr, mbr_heading, edges_gpd, edges, a_error):
    flag = [True for i in range(len(edges))]
    count = 0
    for i in range(len(edges)):
        diff = calculate_heading_difference(edges_gpd, edges[i], mbr, mbr_heading)
        if diff > a_error:
            flag[i] = False
            count = count+1
    new_edges = []
    for i in range(len(flag)):
        if flag[i]:
            new_edges.append(edges[i])
    return new_edges


def filter_routes(routes):
    flag = [True for i in range(len(routes))]
    for i in range(len(routes)-1):
        for j in range(i+1, len(routes)):
            if flag[i] and flag[j]:
                if len(routes[i]) < len(routes[j]):
                    if (routes[i][0] in routes[j]) and (routes[i][-1] in routes[j]):
                        ind_1 = routes[j].index(routes[i][0])
                        ind_2 = routes[j].index(routes[i][-1])
                        if ind_2-ind_1+1 == len(routes[i]):
                            flag[i] = False
                            break
                else:
                    if (routes[j][0] in routes[i]) and (routes[j][-1] in routes[i]):
                        ind_1 = routes[i].index(routes[j][0])
                        ind_2 = routes[i].index(routes[j][-1])
                        if ind_2-ind_1+1 == len(routes[j]):
                            flag[j] = False
    new_routes = []
    for i in range(len(flag)):
        if flag[i]:
            new_routes.append(routes[i])
    return new_routes


def filter_duplicate_routes(routes):
    flag = [True for i in range(len(routes))]
    for i in range(len(routes)-1):
        for j in range(i+1, len(routes)):
            if flag[i] or flag[j]:
                if len(routes[i]) < len(routes[j]):
                    if (routes[i][0] in routes[j]) and (routes[i][-1] in routes[j]):
                        ind_1 = routes[j].index(routes[i][0])
                        ind_2 = routes[j].index(routes[i][-1])
                        if ind_2-ind_1+1 == len(routes[i]):
                            flag[j] = False
                else:
                    if (routes[j][0] in routes[i]) and (routes[j][-1] in routes[i]):
                        ind_1 = routes[i].index(routes[j][0])
                        ind_2 = routes[i].index(routes[j][-1])
                        if ind_2-ind_1+1 == len(routes[j]):
                            flag[i] = False
    new_routes = []
    for i in range(len(flag)):
        if flag[i]:
            new_routes.append(routes[i])
    return new_routes


def calculate_shortest_path(road_graph, edges_gpd, start_edge_id, end_edge_id):
    route = []
    if start_edge_id == end_edge_id:
        route.append(start_edge_id)
    elif edges_gpd.iloc[start_edge_id]['to'] == edges_gpd.iloc[end_edge_id]['from']:
        route.append(start_edge_id)
        route.append(end_edge_id)
    else:
        source = edges_gpd.iloc[start_edge_id]['to']
        target = edges_gpd.iloc[end_edge_id]['from']
        try:
            # net_distance = nx.shortest_path_length(road_graph, source, target, weight='length')
            sp = nx.shortest_path(road_graph, source, target, weight='length')
        except Exception as err:
            print err
            # net_distance = 3*eu_distance
        else:
            route.append(start_edge_id)
            for i in range(1, len(sp)):
                route.append(road_graph[sp[i-1]][sp[i]][0]['Edge_ID'])
            route.append(end_edge_id)
    return route


def query_edges_by_point_range(edges_gpd, point, diameter):
    from shapely.ops import nearest_points
    edge_ids = list(edges_gpd.sindex.intersection(point.buffer(diameter).exterior.bounds, objects='raw'))
    edge_id_list = []
    for edge_id in edge_ids:
        edge = edges_gpd.iloc[edge_id]
        results = nearest_points(point, edge['geometry'])
        d = point.distance(results[1])
        if d <= diameter:
            edge_id_list.append(edge['Edge_ID'])
    return edge_id_list


def query_candidate_routes(road_graph, edges_gpd, gps_track, clusters, d_error, a_error):
    candidate_routes = []
    candidate_edges = []
    mbr_headings = []
    mbrs = []
    work_clusters = []
    for i in range(len(clusters)):
        segment = gps_track.iloc[clusters[i]]
        c1 = segment.iloc[0]['geometry'].buffer(d_error).exterior
        c2 = segment.iloc[-1]['geometry'].buffer(d_error).exterior
        mbr = c1.union(c2).minimum_rotated_rectangle
        mbr_heading = calculate_bearing(segment.iloc[0]['geometry'], segment.iloc[-1]['geometry'])
        # search for candidate edges
        start_edges = query_edges_by_point_range(edges_gpd, segment.iloc[0]['geometry'], d_error)
        end_edges = query_edges_by_point_range(edges_gpd, segment.iloc[-1]['geometry'], d_error)
        ind = 1
        while (not start_edges) and ind < len(segment) - 1:
            ind = ind + 1
            start_edges = query_edges_by_point_range(edges_gpd, segment.iloc[ind]['geometry'], d_error)
        ind = len(segment) - 2
        while (not end_edges) and ind > 0:
            ind = ind - 1
            end_edges = query_edges_by_point_range(edges_gpd, segment.iloc[ind]['geometry'], d_error)
        # filtering candidate edges
        start_edges = filter_edges(mbr, mbr_heading, edges_gpd, start_edges, a_error)
        end_edges = filter_edges(mbr, mbr_heading, edges_gpd, end_edges, a_error)
        # construct candidate routes
        routes = []
        for start in start_edges:
            for end in end_edges:
                route = calculate_shortest_path(road_graph, edges_gpd, start, end)
                if route:
                    routes.append(route)

        if not routes:
            if start_edges:
                for edge_id in start_edges: routes.append([edge_id])
            if end_edges:
                for edge_id in end_edges: routes.append([edge_id])

        if routes:
            if i < len(clusters) - 1:
                routes = filter_duplicate_routes(routes)
            # else:
            #    routes = filter_routes(routes)
            candidate_routes.append(routes)
            candidate_edges.append((start_edges, end_edges))
            mbr_headings.append(mbr_heading)
            mbrs.append(mbr)
            work_clusters.append(clusters[i])
        else:
            print('cluster %d does not have candidate routes!' % i)
            print start_edges, end_edges
    return candidate_routes, candidate_edges, mbr_headings, mbrs, work_clusters


def calculate_observation_probabilities(edges_gpd, candidate_routes, mbr_headings, mbr_list):
    obs_prob = []
    for i in range(len(candidate_routes)):
        obs_prob_i = []
        for route in candidate_routes[i]:
            diff_sum = 0.0
            for edge_id in route:
                diff = calculate_heading_difference(edges_gpd, edge_id, mbr_list[i], mbr_headings[i])
                diff_sum = diff_sum + diff
            p = 1 - diff_sum/(180*len(route))
            obs_prob_i.append(p)
        obs_prob.append(obs_prob_i)
    return obs_prob


def save_to_file_candidate_routes(filename, candidate_routes, obs_prob):
    with open(filename, 'w') as fWriter:
        for i in range(len(candidate_routes)):
            fWriter.write('%d:\n' % i)
            for j in range(len(candidate_routes[i])):
                for edge_id in candidate_routes[i][j]:
                    fWriter.write('%d, ' % edge_id)
                fWriter.write('[%f]' % obs_prob[i][j])
                fWriter.write('\n')
            fWriter.write('\n')


def calculate_transit_route(road_graph, edges_gpd, gps_track, cluster, next_cluster, route, next_route):
    new_route = []
    net_distance = 0
    if next_route[0] in route:
        ind1 = route.index(next_route[0])
        if len(route)-ind1 <= len(next_route):
            if route[ind1:] == next_route[0:len(route)-ind1]:
                new_route = [edge_id for edge_id in route]
                new_route.extend(next_route[len(route)-ind1:])
                # print route, next_route, new_route
    if not new_route:
        sp = calculate_shortest_path(road_graph, edges_gpd, route[-1], next_route[0])
        if sp:
            new_route.extend(route)
            new_route.extend(sp[1:])
            new_route.extend(next_route[1:])
    # calculate network distance
    if new_route:
        for i in range(len(new_route)-1):
            net_distance = net_distance + edges_gpd.iloc[new_route[i]]['length']
        end_point = gps_track.iloc[next_cluster[-1]]['geometry']
        end_edge = edges_gpd.iloc[new_route[-1]]['geometry']
        net_distance = net_distance + end_edge.project(end_point)
        start_point = gps_track.iloc[cluster[0]]['geometry']
        start_edge = edges_gpd.iloc[new_route[0]]['geometry']
        net_distance = net_distance - start_edge.project(start_point)
    return net_distance, new_route


'''
def viterbi_forward(road_graph, edges_gpd, gps_track, clusters, candidate_routes, obs_prob):
    # forward computation
    pre = []
    f = [obs_prob[0]]
    possible_paths = []
    tran_possibilities = []
    for i in range(1, len(candidate_routes)):
        pre_i = []
        f_i = []
        possible_paths_i = []
        tran_possibilities_i = []
        for j in range(len(candidate_routes[i])):
            temp_f = []
            possible_paths_i_j = []
            tran_possibilities_i_j = []
            for k in range(len(candidate_routes[i - 1])):
                # calculate euclidean distance
                eu_distance = gps_track.iloc[clusters[i - 1][0]]['geometry'].distance(
                    gps_track.iloc[clusters[i - 1][-1]]['geometry'])
                eu_distance = eu_distance + gps_track.iloc[clusters[i][0]]['geometry'].distance(
                    gps_track.iloc[clusters[i][-1]]['geometry'])
                eu_distance = eu_distance + gps_track.iloc[clusters[i - 1][-1]]['geometry'].distance(
                    gps_track.iloc[clusters[i][0]]['geometry'])
                net_distance, new_route = calculate_transit_route(road_graph,
                                                                  edges_gpd,
                                                                  gps_track,
                                                                  clusters[i - 1],
                                                                  clusters[i],
                                                                  candidate_routes[i - 1][k],
                                                                  candidate_routes[i][j])
                if (not new_route) and (net_distance == 0):
                    tp_k_j = 0.0000001
                elif net_distance > 2 * eu_distance:
                    tp_k_j = 0.0000001
                else:
                    tp_k_j = 1 - abs(eu_distance - net_distance) / eu_distance
                # if not new_route:
                #    print (candidate_routes[i-1][k], candidate_routes[i][j])
                tran_possibilities_i_j.append([eu_distance, net_distance, tp_k_j])
                temp_f.append(f[-1][k] * tp_k_j)
                possible_paths_i_j.append(new_route)
            tran_possibilities_i.append(tran_possibilities_i_j)
            f_i.append(max(temp_f) * obs_prob[i][j])
            max_ind = temp_f.index(max(temp_f))
            pre_i.append(max_ind)
            # print tran_possibilities_i_j
            possible_paths_i.append(possible_paths_i_j)
        tran_possibilities.append(tran_possibilities_i)
        f.append(f_i)
        pre.append(pre_i)
        possible_paths.append(possible_paths_i)
    return f, pre, possible_paths, tran_possibilities
'''


def viterbi_backward(f, pre):
    # backward search
    r_list = []
    c = f[-1].index(max(f[-1]))
    r_list.append(c)
    for i in range(len(pre) - 1, -1, -1):
        c = pre[i][c]
        r_list.insert(0, c)
    return r_list


def get_optimal_path(r_list, possible_paths):
    opt_route = []
    opt_route.extend(possible_paths[0][r_list[1]][r_list[0]])
    for i in range(len(possible_paths) - 1):
        route = possible_paths[i][r_list[i + 1]][r_list[i]]
        next_route = possible_paths[i + 1][r_list[i + 2]][r_list[i + 1]]
        # print route, next_route
        k = 0
        while k < len(next_route) and (next_route[k] in route):
            k = k + 1
        opt_route.extend(next_route[k:])
    return opt_route


def save_to_file_transit_routes(filename, possible_paths, tran_possibilities):
    with open(filename, 'w') as fWriter:
        for i in range(len(possible_paths)):
            fWriter.write('%d:\n' % i)
            for j in range(len(possible_paths[i])):
                for k in range(len(possible_paths[i][j])):
                    for edge_id in possible_paths[i][j][k]:
                        fWriter.write('%d, ' % edge_id)
                    fWriter.write(' [')
                    for item in tran_possibilities[i][j][k]:
                        fWriter.write('%f ' % item)
                    fWriter.write(']\n')
            fWriter.write('\n')

'''
def viterbi_search(candidates, start_ind):
    opt_route = []
    ind = start_ind
    while ind < len(candidates):
        if candidates.iloc[ind]['pre_cluster'] >= 0:
            break
        else:
            ind = ind + 1

    if ind < len(candidates):
        # forward search
        ind_pre = candidates.iloc[ind]['pre_cluster']
        f = [candidates.iloc[ind_pre]['obs_prob']]
        pre = []
        cluster_ids = [ind_pre]
        while ind < len(candidates):
            # print ind
            pre_ind = []
            f_ind = []
            ind_pre = candidates.iloc[ind]['pre_cluster']
            for j in range(len(candidates.iloc[ind]['candidate_routes'])):
                temp_f = []
                for k in range(len(candidates.iloc[ind_pre]['candidate_routes'])):
                    temp_f.append(f[-1][k] * candidates.iloc[ind]['tran_prob'][j][k])
                pre_ind.append(temp_f.index(max(temp_f)))
                f_ind.append(max(temp_f) * candidates.iloc[ind]['obs_prob'][j])
            if pre_ind:
                pre.append(pre_ind)
                f.append(f_ind)
                cluster_ids.append(ind)
            if ind < len(candidates) - 2:
                if candidates.iloc[ind + 1]['pre_cluster'] < 0 and candidates.iloc[ind + 2]['pre_cluster'] < 0:
                    ind = ind + 3
                    break
                elif candidates.iloc[ind + 1]['pre_cluster'] < 0:
                    ind = ind + 2
                else:
                    ind = ind + 1
            elif ind < len(candidates) - 1:
                if candidates.iloc[ind + 1]['pre_cluster'] < 0:
                    ind = ind + 2
                else:
                    ind = ind + 1
            else:
                ind = ind + 1

        # backwork search
        if len(cluster_ids) > 1:
            print cluster_ids
            r_list = []
            c = f[-1].index(max(f[-1]))
            r_list.append(c)
            for k in range(len(pre) - 1, -1, -1):
                c = pre[k][c]
                r_list.insert(0, c)
            print r_list
            # optimal route
            opt_route.extend(candidates.iloc[cluster_ids[1]]['tran_routes'][r_list[1]][r_list[0]])
            for i in range(1, len(cluster_ids) - 1):
                route = candidates.iloc[cluster_ids[i]]['tran_routes'][r_list[i]][r_list[i - 1]]
                next_route = candidates.iloc[cluster_ids[i + 1]]['tran_routes'][r_list[i + 1]][r_list[i]]
                # print route, next_route
                k = 0
                while k < len(next_route) and (next_route[k] in route):
                    k = k + 1
                opt_route.extend(next_route[k:])
        if not opt_route:
            ind = ind + 1
        # print ('end cluster %d' % ind)
    return opt_route, ind



def obr_mapper(road_graph_utm, gpd_edges_utm, edge_index, trip, d_error, debug=False):
    # segment trajectories
    clusters, outliers, removed = obr(trip, d_error, 2)
    # query candidate routes and calculate observation probabilities
    candidate_routes, candidate_edges, mbr_headings, mbr_list, work_clusters = \
        query_candidate_routes(road_graph_utm, gpd_edges_utm, edge_index, trip, clusters, d_error)
    obs_prob = calculate_observation_probabilities(gpd_edges_utm, candidate_routes, mbr_headings, mbr_list)
    # Viterbi algorithm for finding the optimal route
    f, pre, possible_paths, tran_possibilities = viterbi_forward(road_graph_utm,
                                                                 gpd_edges_utm,
                                                                 trip,
                                                                 work_clusters,
                                                                 candidate_routes,
                                                                 obs_prob)
    r_list = viterbi_backward(f, pre)
    opt_route = get_optimal_path(r_list, possible_paths)
    if debug:
        import os
        cur_dir = os.getcwd()
        debug_dir = os.path.join(cur_dir, r'debug_results')
        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)
        save_to_file_transit_routes(
            os.path.join(debug_dir, r'obr_transit_prob.txt'), possible_paths, tran_possibilities)
        save_to_file_candidate_routes(
            os.path.join(debug_dir, r'obr_candidate_routes.txt'), candidate_routes, obs_prob)

    return opt_route
'''


def viterbi_forward_search(road_graph, edges_gpd, gps_track, clusters, candidate_routes, obs_prob):
    # forward computation
    i_pre = 0
    i = 1
    pre = []
    f = [obs_prob[0]]
    possible_paths = []
    tran_possibilities = []
    work_cluster_ids = []
    while i < len(candidate_routes):
        # print i
        pre_i = []
        f_i = []
        possible_paths_i = []
        tran_possibilities_i = []
        # calculate euclidean distance
        eu_distance = gps_track.iloc[clusters[i_pre][0]]['geometry'].distance(
            gps_track.iloc[clusters[i_pre][-1]]['geometry'])
        eu_distance = eu_distance + gps_track.iloc[clusters[i_pre][-1]]['geometry'].distance(
            gps_track.iloc[clusters[i][0]]['geometry'])
        eu_distance = eu_distance + gps_track.iloc[clusters[i][0]]['geometry'].distance(
            gps_track.iloc[clusters[i][-1]]['geometry'])
        # print i_pre, i
        for j in range(len(candidate_routes[i])):
            temp_f = []
            possible_paths_i_j = []
            tran_possibilities_i_j = []
            for k in range(len(candidate_routes[i_pre])):
                net_distance, new_route = calculate_transit_route(road_graph,
                                                                  edges_gpd,
                                                                  gps_track,
                                                                  clusters[i_pre],
                                                                  clusters[i],
                                                                  candidate_routes[i_pre][k],
                                                                  candidate_routes[i][j])
                if (not new_route) and (net_distance == 0):
                    tp_k_j = 0.0000001
                elif net_distance > 2 * eu_distance:
                    tp_k_j = 0.0000001
                else:
                    tp_k_j = 1 - abs(eu_distance - net_distance) / eu_distance
                # if not new_route:
                #    print (candidate_routes[i-1][k], candidate_routes[i][j])
                tran_possibilities_i_j.append([eu_distance, net_distance, tp_k_j])
                temp_f.append(f[-1][k] * tp_k_j)
                possible_paths_i_j.append(new_route)
            tran_possibilities_i.append(tran_possibilities_i_j)
            f_i.append(max(temp_f) * obs_prob[i][j])
            max_ind = temp_f.index(max(temp_f))
            pre_i.append(max_ind)
            # print tran_possibilities_i_j
            possible_paths_i.append(possible_paths_i_j)
        if max(f_i) < 0.000001:
            # broken
            if i - i_pre == 1:
                if work_cluster_ids:
                    tran_possibilities.pop(-1)
                    f.pop(-1)
                    pre.pop(-1)
                    possible_paths.pop(-1)
                    i_pre = work_cluster_ids[-1]
                else:
                    i_pre = i_pre + 1
                    i = i_pre + 1
                    f = [obs_prob[i_pre]]
            else:
                i = i + 1
        else:
            tran_possibilities.append(tran_possibilities_i)
            f.append(f_i)
            pre.append(pre_i)
            possible_paths.append(possible_paths_i)
            work_cluster_ids.append(i_pre)
            i_pre = i
            i = i + 1
            # is_broken = False
        if len(candidate_routes) - 1 == i_pre:
            work_cluster_ids.append(i_pre)
        if i - i_pre > 4:
            break
    return f, pre, possible_paths, tran_possibilities, work_cluster_ids


def find_optimal_route_broken(road_graph, edges_gpd, gps_track, clusters, candidate_routes, obs_prob):
    final_cluster_id = 0
    opt_route_list = []
    work_clusters = []
    poss_paths = []
    tran_prob = []
    while final_cluster_id < len(clusters) - 1:
        f, pre, possible_paths, tran_possibilities, work_cluster_ids = \
            viterbi_forward_search(road_graph,
                                   edges_gpd,
                                   gps_track,
                                   clusters[final_cluster_id:],
                                   candidate_routes[final_cluster_id:],
                                   obs_prob[final_cluster_id:])
        if possible_paths:
            r_list = viterbi_backward(f, pre)
            opt_route = get_optimal_path(r_list, possible_paths)
            opt_route_list.append(opt_route)
            temp = []
            for item in work_cluster_ids:
                temp.append(item + final_cluster_id)
            work_clusters.append(temp)
            poss_paths.append(possible_paths)
            tran_prob.append(tran_possibilities)
            final_cluster_id = final_cluster_id + work_cluster_ids[-1] + 1
        else:
            final_cluster_id = final_cluster_id + 1
        print final_cluster_id

    if len(opt_route_list) == 1:
        return opt_route_list[0]
    else:
        opt_route = opt_route_list[0]
        for i in range(len(opt_route_list) - 1):
            start_edge_id = opt_route_list[i][-1]
            end_edge_id = opt_route_list[i + 1][0]
            connect_route = calculate_shortest_path(road_graph, edges_gpd, start_edge_id, end_edge_id)
            # print len(opt_route)
            if connect_route:
                opt_route.extend(connect_route[1:-1])
                opt_route.extend(opt_route_list[i + 1])
            else:
                print('Route is broken from %d to %d.' % (start_edge_id, end_edge_id))
        return opt_route
    # save_to_file_matching_result('matching_result.txt', opt_route)


def obr_mapper_v1(road_graph_utm, gpd_edges_utm, trip, d_error, a_error, debug=False):
    # segment trajectories
    clusters, outliers, removed = obr(trip, d_error, 2)
    # query candidate routes and calculate observation probabilities
    candidate_routes, candidate_edges, mbr_headings, mbr_list, work_clusters = \
        query_candidate_routes(road_graph_utm, gpd_edges_utm, trip, clusters, d_error, a_error)
    obs_prob = calculate_observation_probabilities(gpd_edges_utm, candidate_routes, mbr_headings, mbr_list)
    # Viterbi algorithm for finding the optimal route
    opt_route = find_optimal_route_broken(road_graph_utm,
                                          gpd_edges_utm,
                                          trip,
                                          work_clusters,
                                          candidate_routes,
                                          obs_prob)
    if debug:
        import os
        cur_dir = os.getcwd()
        debug_dir = os.path.join(cur_dir, r'debug_results')
        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)
        save_to_file_candidate_routes(
            os.path.join(debug_dir, r'obr_candidate_routes.txt'), candidate_routes, obs_prob)

    return opt_route
