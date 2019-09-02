# -*- coding: utf-8 -*


def get_edge_info(gpd_edges_utm, from_id, to_id):
    results = gpd_edges_utm[(gpd_edges_utm['from'] == from_id) & (gpd_edges_utm['to'] == to_id)]
    if len(results) > 1:
        if results.iloc[0]['length']<results.iloc[1]['length']:
            return results.iloc[0]
        else:
            return results.iloc[1]
    else:
        return results.iloc[0]


def network_distance(road_graph_utm, gpd_edges_utm, candidate1, candidate2):
    import networkx as nx
    '''
    calculate network distance between two candidates
    input:
    G: the road network
    candidate1: a candidate (from, to, proj_point, distance) represents as a pandas series 
    candidate2: a candidate (from, to, proj_point, distance) represents as a pandas series
    output: [d, sp]
             d is the shortest path distance, 
             sp is the shortest path distance between the given two candidates
    '''
    d = 0
    sp_edges = []
    p1 = candidate1['proj_point']
    edge1 = candidate1['road']
    p2 = candidate2['proj_point']
    edge2 = candidate2['road']
    if edge1['geometry'] == edge2['geometry']:
        # part 2 of the distance from the projected point of candidate1 to the end point of the corresponding edge
        d2 = edge1['geometry'].project(p1)
        # part 3 of the distance from the start point of the corresponding edge to the projected candidate2
        d3 = edge2['geometry'].project(p2)
        d = d3-d2
        if d < 0:
            #d = 2*edge1['length']-d2-d3
            #sp_edges.append(edge1)
            #sp_edges.append(edge2)
            d = 0
        #else:
            #sp_edges.append(edge1)
        sp_edges.append(edge1)
    elif candidate1['to'] == candidate2['from']:
        # part 2 of the distance from the projected point of candidate1 to the end point of the corresponding edge
        d2 = edge1['length'] - edge1['geometry'].project(p1)
        # part 3 of the distance from the start point of the corresponding edge to the projected candidate2
        d3 = edge2['geometry'].project(p2)
        d = d2+d3
        sp_edges.append(edge1)
        sp_edges.append(edge2)
        #print('case2')
        #print d2, d3
        #if d<0:
        #    print 'case2'
        #    print d2
        #    print d3
    else:
        # part 1 of the distance
        source = candidate1['to']
        target = candidate2['from']
        try:
            d1 = nx.shortest_path_length(road_graph_utm, source, target, weight='length')
            sp = nx.shortest_path(road_graph_utm, source, target, weight='length')
        except Exception as err:
            print err
            d = 100000000
        else:
            # part 2 of the distance from the projected point of candidate1 to the end point of the corresponding edge
            d2 = edge1['length'] - edge1['geometry'].project(p1)
            # part 3 of the distance from the start point of the corresponding edge to the projected candidate2
            d3 = edge2['geometry'].project(p2)
            d = d1+d2+d3
            if d1 > 0 :
                sp_edges.append(edge1)
            for i in range(len(sp)-1):
                sp_edges.append(get_edge_info(gpd_edges_utm,sp[i],sp[i+1]))
            if d2 > 0 :
                sp_edges.append(edge2)
            #print('case1')
            #print d1, d2, d3
            #if d<0:print 'case3'
        #if d < 0:
        #    print(candidate1, candidate2)
    return d, sp_edges
