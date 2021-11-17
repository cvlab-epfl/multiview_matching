import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import cv2

from multiview_matching import utils
import multiview_matching as mm
import multiview_calib as mc

import time

def generate_points(calibration, N=10, sigma_noise=10):
    
    views = list(calibration.keys())

    points_3d = []
    for _ in range(N):
        x = np.random.uniform(-4,4)
        y = np.random.uniform(-4,4)
        z = np.random.uniform(0, 2.5)
        points_3d.append((x,y,z))
    points_3d = np.array(points_3d)

    positions = {}
    for view in views:
        K = np.array(calibration[view]['K'])
        R = np.array(calibration[view]['R'])
        t = np.array(calibration[view]['t'])
        dist = np.array(calibration[view]['dist'])

        proj, mask = mc.project_points(points_3d, K, R, t, dist, image_shape=(1024, 1920))
        noise = np.random.randn(*proj.shape)*sigma_noise

        positions[view] = proj+noise
        
    return positions

def find_correspondences(calibration, positions):
    
    views = list(positions.keys())
    
    detections = {}
    for view in views:
        detections[view] = [mm.Detection2D(view=view, index=i, position=position) 
                            for i,position in enumerate(positions[view])]

    g = mm.build_graph(detections, views, calibration, max_dist=25.0, n_candidates=3, distance_none=50,
                       verbose=0)

    cliques, costs = mm.solve_ilp(g, views, weight_f=lambda d: np.exp(-d**2/20**2), verbose=0)

    detections_3d = mm.triangulate_cliques(cliques, calibration)
    
    return detections_3d

if __name__=='__main__':
    
    calibration_ = utils.json_read("calibration.json")
    calibration = {}
    for view,x in calibration_.items():
        calibration[view] = x 
    if True:
        for view,x in calibration_.items():
            x['t'] = [float(y+np.random.randn()/2) for y in x['t']]
            calibration[view+"_1"] = x

    print("N views max", len(calibration))

    views = list(calibration.keys())
    
    
    times = {}
    for n_views in [3, 4, 5, 6, 7, 8]:

        calib = {view:calibration[view] for view in views[:n_views]}

        times[n_views] = {}
        for n_objs in [1, 3, 6, 9, 12, 15, 20, 30]:

            start_time = time.time()

            n_trials = 50

            times[n_views][n_objs] = []
            for _ in range(n_trials):

                positions = generate_points(calib, N=n_objs, sigma_noise=10)

                detections_3d = find_correspondences(calib, positions)

            elapsed_time = time.time()-start_time
            times[n_views][n_objs] = elapsed_time/n_trials

            print(n_views, n_objs, "Elapsed time {:.3f}".format(times[n_views][n_objs]))
            
            
    plt.figure(figsize=(5,3))
    for n_views,t in times.items():
        x = list(t.keys())
        y = list(t.values())

        plt.semilogy(x,y, '.-', label='{} views'.format(n_views), markersize=8, lw=2)

    ax = plt.gca()
    yticks = [0.05, 0.1, 0.5, 1, 5, 10]
    ax.set_yticks(yticks)
    ax.set_yticklabels(["{:.2f}".format(y) for y in yticks])
    plt.xlabel('Number of objects')
    plt.ylabel('Time [s]')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("matching_performance.jpg", dpi=200)