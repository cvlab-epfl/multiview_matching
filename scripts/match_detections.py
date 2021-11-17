#----------------------------------------------------------------------------
# Created By  : Leonardo Citraro leonardo.citraro@epfl.ch
# Date: 2021
# --------------------------------------------------------------------------
import numpy as np
import argparse
import os
from itertools import repeat

import multiview_matching as mm
from multiview_matching.utils import utils

def weight_function(d, sigma):
    return np.exp(-d**2/sigma**2)

def match(i, positions, views, calibration, dist_max, dist_none, n_candidates, sigma_weights, verbose):
    
    detections = {}
    for view in positions.keys():
        detections[view] = [mm.Detection2D(view=view, index=i, position=position) 
                            for i,position in enumerate(positions[view])]

    G = mm.build_graph(detections, views, calibration, dist_max, n_candidates, 
                       distance_none=dist_none, verbose=False)

    cliques, costs = mm.solve_ilp(G, views, weight_f=lambda d: np.exp(-d**2/sigma_weights**2), verbose=False)

    positions_3d = mm.triangulate_cliques(cliques, calibration, outliers_rejection=True, m=2.0)        

    if verbose>=1:
        msg = "Sample {:06d} - number of cliques of each size: ".format(i)
        for n in range(2,len(views)+1):
            size = len([c for c in cliques if len(c)==n])
            msg += "{}:{}|".format(n,size if size>0 else '-') 
        print(msg)
            
    return np.array(positions_3d).tolist(), \
           [{d.view:d.position for d in c} for c in cliques], \
           costs

def main(detections='detections.json',
         calibration='calibration.json',
         n_candidates=4,         
         dist_max=10,
         dist_none=20,
         sigma_weights=10,
         verbose=2,
         output_path='.',
         n_threads=16):
    """
    Parameters:
    -----------
    detections : filename
        JSON file containing a list of dictionaries of detections for each view.
        Each dictionary represent one time/event.
        [{'view1':[[x1,y1],..], 'view2':[[x2,y2],...]}, ...]
    calibration : filename
        JSON file containing intrinsics and extrinsics parameters for each view
        {"view1": {'K': ...3x3..., 'R': ...3x3..., 'dist': ...1x5..., 't': ...1x3...},
         "view2": {'K': ...3x3..., 'R': ...3x3..., 'dist': ...1x5..., 't': ...1x3...},
         ...}
    n_candidates : int
        maximum number of detections considered candidated matches to another detection. 
        Lowering this value reduces the computation cost. Usually 2-4 is good.         
    dist_max : float
        a detection is considered a candidate match to a detection in another
        view if its ditance to the epiline is less than this value. Otherwise is disacrded right away.
        For 1080x1920 images the a typical range is 5-15 pixels, for 4K images could be the double.
    dist_none : float
        In order to always form clqiues of degreen N where N is the number of views, we
        have to create one dummy detections in each view. The distance to this dummy detection
        is defined by this parameter.
        Typical value is dist_none==dist_max*2
    sigma_weights : float
        In ILP we convert the distances between the epilines and the detections to probabilities using a Gaussian function.
        sigma_weights defines the sigma of the Gaussian.
        Typical value sigma_weights=dist_max
    verbose : int
        0 : silent
        1 : verbose
        2 : very verbose
    output_path : string
        output folder where to save the results
    """
    
    utils.mkdir(output_path)
    
    all_detections = utils.json_read(detections)
    print("Number of frames:{}".format(len(all_detections)))

    calibration = utils.json_read(calibration)
    views = list(calibration.keys())

    results = []
    if n_threads<1:
        for i,positions in enumerate(all_detections):
            results.append(match(i,
                                 positions, 
                                 views, 
                                 calibration, 
                                 dist_max, 
                                 dist_none, 
                                 n_candidates, 
                                 sigma_weights, 
                                 verbose))
        
    else:
        import multiprocessing
        with multiprocessing.Pool(n_threads) as pool:
            results = pool.starmap(match, zip(range(len(all_detections)),
                                              all_detections, 
                                              repeat(views), 
                                              repeat(calibration), 
                                              repeat(dist_max), 
                                              repeat(dist_none), 
                                              repeat(n_candidates), 
                                              repeat(sigma_weights), 
                                              repeat(verbose)))              
            
    filename_out = os.path.join(output_path, "detections_3d_n{}_dmax{}_dnone{}_sigma{}_{}.json".format(n_candidates,
                                                                                                       dist_max,
                                                                                                       dist_none,
                                                                                                       sigma_weights,
                                                                                                       len(all_detections)))
    utils.json_write(filename_out, results)
    
if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')    

    parser = argparse.ArgumentParser()   
    parser.add_argument("--detections", "-d", type=str, required=True, default="detections.json",
                        help="""JSON file containing a list of dictionaries of detections for each view.
                                Each dictionary represent one time/event.
                                [{'view1':[[x1,y1],..], 
                                  'view2':[[x2,y2],...]}, ...]""")
    parser.add_argument("--calibration", "-c", type=str, required=True, default="calibration.json",
                        help="""JSON file containing intrinsics and extrinsics parameters for each view
                                {"view1": {'K': ...3x3..., 'R': ...3x3..., 
                                           'dist': ...1x5..., 't': ...1x3...},
                                 "view2": {'K': ...3x3..., 'R': ...3x3..., 
                                           'dist': ...1x5..., 't': ...1x3...},
                                 ...}""")
    parser.add_argument("--n_candidates", "-n", type=int, required=False, default=4,
                        help="""maximum number of detections considered candidated matches to another detection. 
                                Lowering this value reduces the computation cost. Usually 2-4 is good.""")     
    parser.add_argument("--dist_max", "-md", type=float, required=False, default=10.0,
                        help="""A detection is considered a candidate match to a detection in another
                                view if its ditance to the epiline is less than this value. Otherwise is disacrded right away.
                                For 1080x1920 images the a typical range is 5-15 pixels, for 4K images could be the double.""") 
    parser.add_argument("--dist_none", "-dn", type=float, required=False, default=20.0,
                        help="""In order to always form clqiues of degreen N where N is the number of views, we
                                have to create one dummy detections in each view. The distance to this dummy detection
                                is defined by this parameter.
                                Typical value is dist_none==dist_max*2""") 
    parser.add_argument("--verbose", "-v", type=int, required=False, default=1) 
    parser.add_argument("--output_path", "-o", type=str, required=False, default='.')
    parser.add_argument("--n_threads", "-t", type=int, required=False, default=16) 
    
    args = parser.parse_args()

    main(**vars(args))

# python find_matches.py -d detections.json -c calibration.json --dist_max 10.0 --dist_none 20.0