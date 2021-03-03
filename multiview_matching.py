import os
import sys
import time
import numpy as np
import itertools
import networkx as nx
import cv2
from pulp import *

from multiview_calib.twoview_geometry import draw_epilines, fundamental_from_poses, compute_epilines, distance_point_line

__all__ = ["Detection","Detection2D", "Detection3D", "find_candidate_matches", 
           "build_graph", "find_cliques", "compute_clique_cost", "solve_ilp", 
           "triangulate_cliques"]

class Detection(object):
    
    def __init__(self, index=None, position=None, confidence=0.5, datetime=None, id=None):
        self.index = index
        self.position = position
        self.confidence = confidence
        self.datetime  = datetime
        self.id = id
        
    def __str__(self):
        return """{self.__class__.__name__}(index={self.index}, confidence={self.confidence}, datetime={self.datetime}, position={self.position})""".format(self=self)
    
class Detection2D(Detection):
    
    def __init__(self, view=None, index=None, position=None, 
                 position_undist=None, confidence=0.5, 
                 datetime=None, id=None):
        super(Detection2D, self).__init__(index, position, confidence, datetime, id)
        self.view = view
        self.position_undist = position_undist
        self.node = None
        
    def __str__(self):
        return """{self.__class__.__name__}(view={self.view}, index={self.index}, confidence={self.confidence}, datetime={self.datetime}, position={self.position}, position_undist={self.position_undist})""".format(self=self) 
    
class Detection3D(Detection):
    
    def __init__(self, index=None, position=None, confidence=0.5, 
                 datetime=None, clique=None, id=None):
        super(Detection3D, self).__init__(index, position, confidence, datetime, id)
        self.clique = clique
        self.node = None
        
    def __str__(self):
        return """{self.__class__.__name__}(index={self.index}, position={self.position}, confidence={self.confidence}, datetime={self.datetime})""".format(self=self)    

def find_candidate_matches(detections, views, calibration, max_dist=10, n_candidates=2,
                           verbose=0):
    """
    Given a detection in one view, find the best candidates detections on the other views
    
    Parameters
    ----------
    detections : dict of lists of objects of type Detection2D
        {'view1':[Detection1, Detection2, ...], 'view2':[...]}
    views : list
        list cotaining the name of the views i.e. ['view1', 'view2', ...]
    calibration : dict
        extrinsic and instrinsic parameters {'view1':{'R':.., 't':.., 'K':..., 'dist':...}}
    max_dist : float
        a detection is considered a candidate if its distance to a epiline is less than this
    n_candidates : int
        max number of candidates per detection in each view
    """

    sel_indexes = {}
    for view1 in views:
        
        sel_indexes[view1] = {}

        K1 = np.array(calibration[view1]['K'])
        R1 = np.array(calibration[view1]['R'])
        t1 = np.array(calibration[view1]['t'])

        if len(detections[view1])==0:
            continue        
        positions_undist1 = np.reshape([detection.position_undist 
                                        for detection in detections[view1]], (-1,2))

        for view2 in views:

            if view1!=view2:

                sel_indexes[view1][view2] = []

                K2 = np.array(calibration[view2]['K'])
                R2 = np.array(calibration[view2]['R'])
                t2 = np.array(calibration[view2]['t'])

                F = fundamental_from_poses(K1, R1, t1, K2, R2, t2)

                if len(detections[view2])==0:
                    continue                
                positions_undist2 = np.reshape([detection.position_undist 
                                                for detection in detections[view2]], (-1,2))

                _, lines2 = compute_epilines(positions_undist1, None, F)

                for i1,line in enumerate(lines2):

                    distances = [distance_point_line(x, line) for x in positions_undist2]
                    idx_sorted = np.argsort(distances)
                    idxs_candidates = []
                    sel_distances = []
                    for idx in idx_sorted:
                        # exit this loop if the distance start to be 
                        # o high or the number candidates is reached   
                        if verbose==2:
                            print("{}-{} {}-{} {:0.2f}".format(view1, i1, view2, idx, distances[idx]))                        
                        if distances[idx]>max_dist:
                            if verbose==2:
                                print("{}-{} {}-{} discarded because of distance {:0.2f}".format(view1, i1, view2, idx, distances[idx]))
                            else:
                                break 
                        elif len(idxs_candidates)>=n_candidates:
                            if verbose==2:
                                print("{}-{} {}-{} discarded because of number of candidates reached.".format(view1, i1, view2, idx))
                            else:
                                break 
                        else:
                            if verbose==2:
                                print("{}-{} {}-{} selected distance {:0.2f}".format(view1, i1, view2, idx, distances[idx]))   
                            idxs_candidates.append(idx)
                            sel_distances.append(distances[idx])

                    sel_indexes[view1][view2].append((idxs_candidates, sel_distances))
                    
    return sel_indexes

def build_graph(detections, views, calibration, max_dist=10, n_candidates=2,
                distance_none=10, verbose=0):
    """
    Build graph. An edge in this graph connects a detection in one view
    to another detection in another view.
    
    Parameters
    ----------
    detections : dict of lists of objects of type Detection
        {'view1':[Detection1, Detection2, ...], 'view2':[...]}
    views : list
        list cotaining the name of the views i.e. ['view1', 'view2', ...]
    calibration : dict
        extrinsic and instrinsic parameters {'view1':{'R':.., 't':.., 'K':..., 'dist':...}}
    max_dist : float
        a detection is considered a candidate if its distance to a epiline is less than this
    n_candidates : int
        max number of candidates per detection in each view
    """    

    sel_indexes = find_candidate_matches(detections, views, calibration, 
                                         max_dist=max_dist, n_candidates=n_candidates,
                                         verbose=verbose)
    
    if np.all([len(sel_indexes[view])==0 for view in views]):
        return None
    
    g = nx.Graph()

    for view1 in views:

        K1 = np.array(calibration[view1]['K'])
        R1 = np.array(calibration[view1]['R'])
        t1 = np.array(calibration[view1]['t'])   

        for view2 in sel_indexes[view1].keys():

            K2 = np.array(calibration[view2]['K'])
            R2 = np.array(calibration[view2]['R'])
            t2 = np.array(calibration[view2]['t'])    

            F = fundamental_from_poses(K1, R1, t1, K2, R2, t2)

            for idx1, (idxs2, distances) in enumerate(sel_indexes[view1][view2]):
                n1 = "{}-{}".format(view1, idx1)
                detection1 = detections[view1][idx1]
                detection1.node = n1
                g.add_node(n1, detection=detection1)

                if len(idxs2)==0:
                    # no candidates in view2 therefore connect to a fake detection
                    n2 = "{}-None".format(view2)
                    g.add_node(n2, detection=None)
                    g.add_edge(n1, n2, distances=[distance_none]) 
                else:
                    
                    # TODO: can I simplify this by simply addign an edge to None all the times?
                    n2_none = "{}-None".format(view2)
                    if not g.has_edge(n1, n2_none):
                        g.add_edge(n1, n2_none, distances=[distance_none])               

                    for idx2,distance in zip(idxs2, distances):
                        n2 = "{}-{}".format(view2, idx2)
                        detection2 = detections[view2][idx2]
                        detection2.node = n2
                        g.add_node(n2, detection=detection2)

                        if g.has_edge(n1, n2):
                            data = g.get_edge_data(n1, n2)
                            data['distances'].append(distance)
                        else:
                            g.add_edge(n1, n2, distances=[distance])

    # make sure there is an edge connecting all fake detections!
    # this serves the purpose of closing the cliques
    for view1 in views:
        for view2 in sel_indexes[view1].keys():
            n1 = "{}-None".format(view1)
            n2 = "{}-None".format(view2)
            g.add_edge(n1, n2, distances=[distance_none])
            
    return g

def find_cliques(g, n=4):
    all_cliques = []
    for clique in list(nx.enumerate_all_cliques(g)):
        if len(clique)==n:
            # discard cliques that are made by only fake detections
            if sum(['None' in node_name for node_name in clique])!=n:
                all_cliques.append(clique) 
    return all_cliques

def compute_clique_cost(g, clique, weight_f=None):
    
    if weight_f is None:
        weight_f = weight_funcion   

    cost_clique = []
    for s,t in list(itertools.combinations(clique, 2)):
        distances = g.get_edge_data(s,t)['distances']
        # trick to speed it up as len(distances) always 1 or 2
        d = weight_f((distances[0]+distances[-1])/2)
        #d = weight_f(np.mean(distances))
        #d = np.maximum(d, 1e-24)
        cost_clique.append(d)  
        
    return sum(cost_clique)/len(cost_clique)

def weight_funcion(distance):
    return 1*np.exp(-distance**2/4)

def solve_ilp(g, views, weight_f=None, verbose=2):

    start_time = time.time()
    
    n_views = len(views)
    if weight_f is None:
        weight_f = weight_funcion
    
    # find all the candidate cliques (this is the bottleneck of this algorithem)
    all_cliques = find_cliques(g, n=n_views)           
    all_cliques_names = [str(i) for i in range(len(all_cliques))]
    all_cliques_costs = [compute_clique_cost(g, clique, weight_f) for clique in all_cliques]   

    prob = LpProblem("Matching detections", LpMaximize)
    x = LpVariable.dicts("clique", all_cliques_names, 0, 1, LpInteger)

    # cost function
    prob += lpSum([cost*x[name] for name, cost in zip(all_cliques_names, all_cliques_costs)]), "Objective function"

    # constraint: no two cliques share the same detection/node
    for node in g.nodes():
        if 'None' in node: continue
        
        # find all cliques (of degree=n_views) that contain this node
        clique_names = [name for clique,name in zip(all_cliques, all_cliques_names) if node in clique]

        # define a constraint so that only one of these cliques can be active or none
        prob += lpSum([x[name] for name in clique_names]) <= 1, ""

    #prob.writeLP("graph_matching.lp")

    prob.solve()
    final_cost = value(prob.objective)

    elapsed_time = time.time()-start_time

    if verbose>0:
        print("status:{} final cost:{} elapsed_time:{:0.2}s".format(LpStatus[prob.status], final_cost, elapsed_time))
        
    # get the active cliques
    cliques = []
    costs = []
    for clique,name,cost in zip(all_cliques, all_cliques_names, all_cliques_costs):
        if x[name].value()>0:
            if verbose>1:
                print(clique, x[name].value())
            clique_dets = [g.nodes[n]['detection'] for n in clique if 'None' not in n]
            if len(clique_dets)>1:
                cliques.append(clique_dets)
                costs.append(cost)
                    
    return cliques, costs

def triangulate(K1, R1, t1, K2, R2, t2, pts1_undist, pts2_undist):
    P1 = np.dot(K1, np.hstack([R1, t1.reshape(3,1)]))
    P2 = np.dot(K2, np.hstack([R2, t2.reshape(3,1)]))
    tri = cv2.triangulatePoints(P1, P2, np.float64(pts1_undist).T, np.float64(pts2_undist).T).T
    tri = tri[:,:3]/tri[:,[3]]  
    return tri

def triangulate_cliques(cliques, calibration):

    detections_3d = []
    #traing_std = []
    for clique in cliques:

        p3ds = []
        for detection1,detection2 in list(itertools.combinations(clique, 2)):

            view1 = detection1.view
            pt1_undist = detection1.position_undist
            K1 = np.array(calibration[view1]['K'])
            R1 = np.array(calibration[view1]['R'])
            t1 = np.array(calibration[view1]['t'])

            view2 = detection2.view
            pt2_undist = detection2.position_undist
            K2 = np.array(calibration[view2]['K'])
            R2 = np.array(calibration[view2]['R'])
            t2 = np.array(calibration[view2]['t'])   

            p3d = triangulate(K1, R1, t1, K2, R2, t2, pt1_undist, pt2_undist)[0]
            p3ds.append(p3d)

        p3d = np.median(p3ds, 0)
        #p3d_std = np.std(p3ds,0).max()

        detections_3d.append(Detection3D(index=clique[0].index, 
                                         position=p3d, 
                                         clique=clique,
                                         datetime=clique[0].datetime))
        #traing_std.append(p3d_std)
        
    return detections_3d