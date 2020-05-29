import numpy as np
import argparse
import imageio
import os

import multiview_matching as mm
from multiview_matching import utils

def main(detections='detections.json',
         calibration='calibration.json',
         max_dist=10,
         n_candidates=4,
         dist_none=20,
         verbose=0,
         output_path='.',
         dump_visuals=True,
         n_visuals=1,
         filename_images='filenames.json',
         radius=18):
    """
    Parameters:
    -----------
    detections : filename
        list of detections for each view saved as JSON
        [{'view1':[[x1,y1],..], 'view2':[[x2,y2],...]}, ...]
    calibration : filename
        intrinsics and extrinsics parameters for each view saved as JSON
        {"view1": {'K': ...3x3..., 'R': ...3x3..., 'dist': ...1x5..., 't': ...1x3...},
         "view2": {'K': ...3x3..., 'R': ...3x3..., 'dist': ...1x5..., 't': ...1x3...},
         ...}
    max_dist : float
        a detection in one view is considered a candidate match to a detection in another
        view if the disatnce to the epiline is less than this parameter.
        For 1080x1920 images the a typical range is 5-15 for 4K image could be the double.
    n_candidates : int
        maximum number of candidates matching detections. This is to reduce computation.
    dist_none : float
        The variables in the ILP represent cliques of degree n where n is the number of views. 
        When detections are missing, cliques of degree n cannot be formed unless the graph is 
        augmented with artifical detections.
        dist_none is the distance associated to the edges connecting a detection and these artifical detections.
        In general this is dist_none>=max_dist
    verbose : int
        0 : silent
        1 : verbose
        2 : very verbose
    dump_visuals : bool
        enable saving visual results. The filename of the images for each view must be provided.
    n_visuals : int
        the number of images to dump. i.e if n_visuals=3 the first 3 frames will be dumped. 
    filename_images : filename
        list of dictionaries with filenames for each view saved as JSON.
        [{'view1':../file_view1_1.jpg, 'view1':../file_view2_1.jpg},
         {'view1':../file_view1_2.jpg, 'view1':../file_view2_2.jpg},...]
    """
    
    utils.mkdir(output_path)
    
    all_detections = utils.json_read(detections)
    print("Number of frames:{}".format(len(all_detections)))

    calibration = utils.json_read(calibration)
    views = list(calibration.keys())
    views_ = sorted(views)
    
    idx_visuals = list(range(n_visuals))
    filename_images = utils.json_read(filename_images)

    detections_3d = []
    for i,detections in enumerate(all_detections):
        
        if verbose>0:
            print("---timestamp:{}---".format(i))

        sel_indexes = mm.find_candidate_matches(detections, views_, calibration, max_dist, n_candidates)

        G = mm.build_graph(detections, sel_indexes, views_, calibration)

        matches, costs = mm.solve_ilp(G, views_, dist_none,
                                      weight_f=lambda distance: 1*np.exp(-distance**2/max_dist), 
                                      verbose=verbose)

        d3d, d3d_std = mm.triangulate_detections(matches, G, views, calibration)
        if verbose>=2:
            for p,std in zip(d3d, d3d_std):
                print("pos3d:{} std_triang:{}".format(p, std))

        matches_ = []
        for clique,cost in zip(matches,costs):
            det_views = []
            det_ids = []
            for node in clique:
                data = G.nodes[node]
                if 'view' not in data or 'id_det' not in data:
                    continue
                view = data['view']
                id_box = data['id_det']
                det_views.append(view)
                det_ids.append(int(id_box))
            matches_.append((det_views, det_ids, cost))

        detections_3d.append([{'det_views': det_views,
                              'det_ids': det_ids, 
                              'clique_cost':cost,
                              'pos':pos} for (det_views, det_ids, cost), pos in zip(matches_, d3d.tolist())])
        
        if dump_visuals:
            import cv2
            if filename_images is None:
                raise ValeuError("filename_images is None! Provide them or disable dump_visuals.")
                
            colors = [[0,255,0], 
                      [100,100,255], [255,255,0], 
                      [0,255,255], [255,0,255],
                      [255,255,255], [0,0,0],
                      [128,128,128], [50,128,50]]+[np.random.randint(0,255,3).tolist() for _ in range(30)]                
            if i in idx_visuals:
                
                for view in views:
                    
                    img = imageio.imread(filename_images[i][view])
                    
                    img = cv2.putText(img, "detections", (10,50), 
                                      fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                      color=[255,0,0],
                                      fontScale=2)
                    img = cv2.putText(img, "3d points projected", (10,100), 
                                      fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                      color=[255,255,255],
                                      fontScale=2)                    
                    
                    if len(detections[view])>0:
                        for p in detections[view]:
                            '''
                            img = cv2.circle(img, (int(p[0]), int(p[1])), radius=radius, 
                                             color=[255,0,0], thickness=4, lineType=8, shift=0)
                            '''
                            img = cv2.rectangle(img, (int(p[0]-radius),int(p[1]-radius)), 
                                                (int(p[0]+radius),int(p[1]+radius)), 
                                                color=[255,0,0], thickness=3, lineType=8, shift=0)
                        
                    if len(d3d)>0:    
                        K = np.array(calibration[view]['K'])
                        R = np.array(calibration[view]['R'])
                        rvec = cv2.Rodrigues(R)[0]
                        t = np.array(calibration[view]['t'])
                        dist = np.array(calibration[view]['dist'])                         

                        proj = cv2.projectPoints(d3d, rvec, t, K, dist)[0].reshape(-1,2)
                        for p,c in zip(proj,colors):
                            img = cv2.circle(img, (int(p[0]), int(p[1])), radius=int(radius-radius*0.2), 
                                             color=c, thickness=-1, lineType=8, shift=0) 
                            
                    utils.mkdir(os.path.join(output_path, "visuals"))
                    imageio.imwrite(os.path.join(output_path, "visuals", "img_{}_{}.jpg".format(i, view)), img)
                    
    utils.json_write(os.path.join(output_path, "detections_3d_maxd{}_n{}_noned{}_{}.json".format(max_dist,
                                                                                                 n_candidates,
                                                                                                 dist_none,
                                                                                                 len(all_detections))),
                                  detections_3d)
    
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
                        help='JSON file containing the detections')
    parser.add_argument("--calibration", "-c", type=str, required=True, default="calibration.json",
                        help='JSON file containing the relative poses of each pair of view')
    parser.add_argument("--max_dist", "-md", type=float, required=False, default=10.0,
                        help='A detection is a candidate if the distance to the epiline is <=max_dist') 
    parser.add_argument("--n_candidates", "-n", type=int, required=False, default=4,
                        help='Maximum number of candidates') 
    parser.add_argument("--dist_none", "-dn", type=float, required=False, default=20.0,
                        help='Distance for edges conecting detections and artifically added false negatives') 
    parser.add_argument("--verbose", "-v", type=int, required=False, default=0) 
    parser.add_argument("--output_path", "-o", type=str, required=False, default='.') 
    
    parser.add_argument("--dump_visuals", "-dv", default=False, const=True, action='store_const',
                        help='Saves images for visualisation') 
    parser.add_argument("--n_visuals", "-nv", type=int, required=False, default=1)
    parser.add_argument("--filename_images", "-fi", type=str, required=False, default='filenames.json')
    
    args = parser.parse_args()

    main(**vars(args))

# python find_matches.py --detections detections.json --calibration calibration.json --verbose 1 --max_dist 10.0 --dist_none 20.0 --dump_visuals --filename_images filenames.json