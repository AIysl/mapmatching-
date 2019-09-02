def get_optimal_mapping_points(candidates, path):
    import geopandas as gpd
    points = [candidates[i].iloc[path[i]]['proj_point'] for i in range(len(path))]
    return gpd.GeoSeries(points)


def get_optimal_path_edges(candidates, heuristics, optimal_path, debug=False):
    edge_list=[]
    #idxs = trip['optimal candidate']
    for i in range(len(candidates)-1):
        #weights = trip['weights'].iloc[i]
        to_id = optimal_path[i+1]
        from_id = optimal_path[i]
        #print i
        ind = from_id * len(candidates[i+1]) + to_id
        sub_edge_list = heuristics[i].iloc[ind]['sp edges']
        #if len(edge_list)>0:
        #    if edge_list[-1]['geometry'] != sub_edge_list[0]['geometry']:
        #        print 'path broken!'
        if debug:
            print i
        for j in range(len(sub_edge_list)-1):
            edge_list.append(sub_edge_list[j])
            if debug: print sub_edge_list[j]['from'], sub_edge_list[j]['to']
        if i==len(candidates)-1:
            edge_list.append(sub_edge_list[-1])
            if debug: print sub_edge_list[-1]['from'], sub_edge_list[-1]['to']
    return edge_list


def trip_bbox_utm(trip):
    '''
    get the bounding box of the given trip
    input: trip: a trajectory
    output: (minx, miny, maxx, maxy)
    '''
    from shapely.geometry import LineString
    line = LineString(list(trip['geometry_utm']))
    return line.buffer(500).bounds


def get_candidates_as_geodataframe(candidates):
    import geopandas as gpd
    candi_points = []
    for i in range(len(candidates)):
        for j in range(len(candidates[i])):
            candi_points.append(candidates[i].iloc[j]['proj_point'])
    return gpd.GeoDataFrame(data={'geometry':candi_points})


def get_edge_path_lines(edge_path):
    import geopandas as gpd
    lines = []
    for edge in edge_path:
        lines.append(edge['geometry'])
    lines_gpd = gpd.GeoDataFrame(data={'geometry': lines})
    return lines_gpd


def visualize_matching_results(trip, candidates, edge_idx, heuristics, optimal_path, figname='temp.pdf'):
    '''visualize trip points and their corresponding candidates'''
    # edge_idx = build_rtree_index_edges(gpd_edges_utm)
    import geopandas as gpd
    # prepare data
    edges_collection = gpd.GeoDataFrame(list(edge_idx.intersection(trip_bbox_utm(trip), objects='raw')))
    candidates_points = get_candidates_as_geodataframe(candidates)
    trip_utm = gpd.GeoDataFrame(data={'geometry': trip['geometry_utm']})
    opt_proj_points = get_optimal_mapping_points(candidates, optimal_path)
    edge_path = get_optimal_path_edges(candidates, heuristics, optimal_path)
    edge_path_lines = get_edge_path_lines(edge_path)
    # plotting
    ax = edges_collection.plot(figsize=(16, 14), color='black', alpha=0.1)
    trip_utm.plot(ax=ax, color='red', marker='o', markersize=10, label='raw points')
    candidates_points.plot(ax=ax, color='green', marker='+', markersize=50, label='candidates', alpha=0.5)
    opt_proj_points.plot(ax=ax, color='blue', marker='*', markersize=60, label='optimal', alpha=0.5)
    edge_path_lines.plot(ax=ax, color='blue', alpha=0.3, linewidth=5)
    ax.get_legend_handles_labels()
    ax.legend()
    ax.axis('off')
    ax.figure.savefig(figname,dpi=800, format='pdf')

