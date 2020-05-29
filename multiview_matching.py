import os
import sys
import time
import numpy as np
import itertools
import networkx as nx
import cv2
from pulp import *

from multiview_calib.twoview_geometry import draw_epilines, fundamental_from_poses, compute_epilines, distance_point_line
from multiview_calib.singleview_geometry import undistort_points

__all__ = ["find_candidate_matches", "build_graph", "find_cliques", "compute_clique_cost",
           "solve_ilp", "triangulate_detections"]

def find_candidate_matches(detections, views, extrinsics, max_dist=10, n_candidates=2):
    """
    Give a detection in one view, find the best candidates detection on the other view
    
    Parameters
    ----------
    detections : dict
        {'view1':[[x1,y1], [x2, y2], ...], 'view2':[...]}
    views : list
        list cotaining the name of the views i.e. ['view1', 'view2', ...]
    extrinsics : dict
        extrinsic and instrinsic parameters {'view1':{'R':.., 't':.., 'K':.., 'dist':..}}
    max_dist : float
        a detection is considered a candidate if its distance to a epiline is less than this param
    n_candidates : int
        max number of candidates per detection
    """

    sel_indexes = {}
    for view1 in views:

        K1 = np.array(extrinsics[view1]['K'])
        R1 = np.array(extrinsics[view1]['R'])
        t1 = np.array(extrinsics[view1]['t'])
        dist1 = np.array(extrinsics[view1]['dist'])

        pos1 = np.reshape(detections[view1], (-1,2))
        if len(pos1)==0:
            continue
        
        pts1_undist = undistort_points(pos1, K1, dist1)

        sel_indexes[view1] = {}

        for view2 in views:

            if view1!=view2:

                sel_indexes[view1][view2] = []

                K2 = np.array(extrinsics[view2]['K'])
                R2 = np.array(extrinsics[view2]['R'])
                t2 = np.array(extrinsics[view2]['t'])
                dist2 = np.array(extrinsics[view2]['dist'])

                F = fundamental_from_poses(K1, R1, t1, K2, R2, t2)

                pos2 = np.reshape(detections[view2], (-1,2))
                if len(pos2)==0:
                    continue
            
                pts2_undist = undistort_points(pos2, K2, dist2) 

                _, lines2 = compute_epilines(pts1_undist, None, F)

                for i1,line in enumerate(lines2):

                    distances = [distance_point_line(p2, line) for p2 in pts2_undist]
                    idx_min = []
                    for idx in np.argsort(distances)[:n_candidates]:
                        if distances[idx]>max_dist:
                            break
                        idx_min.append(idx)

                    sel_indexes[view1][view2].append(idx_min)
                    
    return sel_indexes

def build_graph(detections, sel_indexes, views, extrinsics):
    
    g = nx.Graph()

    for view1 in sel_indexes.keys():

        K1 = np.array(extrinsics[view1]['K'])
        R1 = np.array(extrinsics[view1]['R'])
        t1 = np.array(extrinsics[view1]['t'])
        dist1 = np.array(extrinsics[view1]['dist'])    

        for view2 in sel_indexes[view1].keys():

            K2 = np.array(extrinsics[view2]['K'])
            R2 = np.array(extrinsics[view2]['R'])
            t2 = np.array(extrinsics[view2]['t'])
            dist2 = np.array(extrinsics[view2]['dist'])    

            F = fundamental_from_poses(K1, R1, t1, K2, R2, t2)

            for idx1, idxs2 in enumerate(sel_indexes[view1][view2]):
                n1 = "{}-{}".format(view1, idx1)
                pos1 = detections[view1][idx1]
                pt1_undist = undistort_points(pos1, K1, dist1)
                _, lines2 = compute_epilines(pt1_undist, None, F)
                line2 = lines2[0]

                g.add_node(n1, view=view1, id_det=idx1, pos=pos1, pos_undist=pt1_undist[0])

                if len(idxs2)==0:
                    n2 = "{}-None".format(view2)
                    g.add_node(n2, pos=(-1,-1), pos_undist=(-1,-1))
                    g.add_edge(n1, n2, distances=[-2]) 
                    continue

                n2_none = "{}-None".format(view2)
                if not g.has_edge(n1, n2_none):
                    g.add_edge(n1, n2_none, distances=[-2])               

                for idx2 in idxs2:
                    n2 = "{}-{}".format(view2, idx2)
                    pos2 = detections[view2][idx2]
                    pt2_undist = undistort_points(pos2, K2, dist2)[0]
                    d = distance_point_line(pt2_undist, line2)

                    g.add_node(n2, view=view2, id_det=idx2, pos=pos2, pos_undist=pt2_undist)

                    if g.has_edge(n1, n2):
                        data = g.get_edge_data(n1, n2)
                        data['distances'].append(d)
                    else:
                        g.add_edge(n1, n2, distances=[d])


    for view1 in sel_indexes.keys():
        for view2 in sel_indexes[view1].keys():
            n1 = "{}-None".format(view1)
            n2 = "{}-None".format(view2)
            g.add_edge(n1, n2, distances=[-2])
            
    return g

def find_cliques(g, n=4):
    all_cliques = []
    for clique in list(nx.enumerate_all_cliques(g)):
        if len(clique)==n:
            all_cliques.append(sorted(clique)) 
    return all_cliques

def compute_clique_cost(g, clique, dist_none=10, weight_f=None):
    
    if weight_f is None:
        weight_f = weight_funcion   

    cost_clique = []
    for s,t in list(itertools.combinations(clique, 2)):
        if 'None' in s or 'None' in t:
            d = weight_f(dist_none)
        else:
            d = weight_f(np.mean(g.get_edge_data(s,t)['distances']))
        d = np.maximum(d, 1e-8)
        cost_clique.append(d)  
        
    return sum(cost_clique)/len(cost_clique)

def weight_funcion(distance):
    return 1*np.exp(-distance**2/4)

def solve_ilp(g, views, dist_none=3, weight_f=None, verbose=2):

    start_time = time.time()
    
    n_views = len(views)
    if weight_f is None:
        weight_f = weight_funcion
    
    # find all the candidate cliques
    all_cliques = find_cliques(g, n=n_views)           

    var_names = ["/".join(sorted(clique)) for clique in all_cliques]

    prob = LpProblem("Matching detections", LpMaximize)
    x = LpVariable.dicts("clique", var_names, 0, 1, LpInteger)

    cost_fun = []
    for clique in all_cliques:

        cost_clique = compute_clique_cost(g, clique, dist_none, weight_f)

        clique_name = "/".join(sorted(clique))
        w = np.sum(cost_clique)*x[clique_name]
        cost_fun.append(w)

    prob += lpSum(cost_fun), "Objective function"

    for node in g.nodes():

        if 'None' in node:
            continue

        cliques = nx.cliques_containing_node(g, node)

        constraint = []
        for clique in cliques:
            if len(clique)==n_views:
                clique_name = "/".join(sorted(clique))
                constraint.append(x[clique_name])

        prob += lpSum(constraint) <= 1, ""

    prob.writeLP("graph_matching.lp")

    prob.solve()
    final_cost = value(prob.objective)

    elapsed_time = time.time()-start_time

    if verbose>0:
        print("status:{} final cost:{} elapsed_time:{:0.2}s".format(LpStatus[prob.status], final_cost, elapsed_time))
        
        
    matches = []
    costs = []
    for clique_name, val in x.items():
        if val.value()>0:
            clique = clique_name.split('/')
            if sum(['None' not in x for x in clique])>1:
                matches.append(clique)  
                costs.append(compute_clique_cost(g, clique, dist_none, weight_f))
                
                if verbose>1:
                    print(clique, val.value())
                    
    idxs_sort = np.argsort(costs)[::-1]
    matches = [matches[i] for i in idxs_sort]
    costs = [costs[i] for i in idxs_sort]
                    
    return matches, costs

def triangulate(K1, R1, t1, K2, R2, t2, pts1_undist, pts2_undist):
    P1 = np.dot(K1, np.hstack([R1, t1.reshape(3,1)]))
    P2 = np.dot(K2, np.hstack([R2, t2.reshape(3,1)]))
    tri = cv2.triangulatePoints(P1, P2, np.float64(pts1_undist).T, np.float64(pts2_undist).T).T
    tri = tri[:,:3]/tri[:,[3]]  
    return tri

def triangulate_detections(matches, g, views, extrinsics):

    detections_3d = []
    traing_std = []
    for match in matches:
        match_ = [node for node in match if 'None' not in node]

        p3ds = []
        for n1,n2 in list(itertools.combinations(match_, 2)):

            data1 = g.nodes[n1]
            view1 = data1['view']
            pt1_undist = np.float64([data1['pos_undist']])
            K1 = np.array(extrinsics[view1]['K'])
            R1 = np.array(extrinsics[view1]['R'])
            t1 = np.array(extrinsics[view1]['t'])
            dist1 = np.array(extrinsics[view1]['dist']) 

            data2 = g.nodes[n2]
            view2 = data2['view']
            pt2_undist = np.float64([data2['pos_undist']])
            K2 = np.array(extrinsics[view2]['K'])
            R2 = np.array(extrinsics[view2]['R'])
            t2 = np.array(extrinsics[view2]['t'])
            dist2 = np.array(extrinsics[view2]['dist'])    

            p3d = triangulate(K1, R1, t1, K2, R2, t2, pt1_undist, pt2_undist)[0]
            p3ds.append(p3d)

        # this can be replaced with an least-squares optimization for better precision
        p3d = np.mean(p3ds, 0)
        p3d_std = np.std(p3ds,0).max()

        detections_3d.append(p3d)
        traing_std.append(p3d_std)
        
    return np.float64(detections_3d), np.float32(traing_std)