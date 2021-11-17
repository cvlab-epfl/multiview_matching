#----------------------------------------------------------------------------
# Created By  : Leonardo Citraro leonardo.citraro@epfl.ch
# Date: 2021
# --------------------------------------------------------------------------
import numpy as np
import cv2

from .twoview_geometry import fundamental_from_poses, compute_epilines, distance_point_line

def _draw_line(img, line, color=(255,0,0), linewidth=10):
    w = img.shape[1]
    x0,y0 = map(int, [0, -line[2]/line[1] ])
    x1,y1 = map(int, [w, -(line[2]+line[0]*w)/line[1] ])
    return cv2.line(img.copy(), (x0,y0), (x1,y1), tuple(color), linewidth)

def draw_epilines(img1_undistorted, img2_undistorted, 
                  pts1_undistorted, pts2_undistorted, 
                  F, mask=None, linewidth=10, markersize=10, scale=1):
    
    img1_, img2_ = img1_undistorted.copy(), img2_undistorted.copy()
    
    pts1_ = np.float64(pts1_undistorted).reshape(-1,2)
    pts2_ = np.float64(pts2_undistorted).reshape(-1,2)
    
    _F = np.float32(F)
    
    s = 1
    scale = np.array([[s,0,1], 
                      [0,s,1], 
                      [0,0,1]])    
    
    if mask is not None:
        pts1_ = pts1_[np.bool_(mask.ravel())]
        pts2_ = pts2_[np.bool_(mask.ravel())]
    
    lines2 = cv2.computeCorrespondEpilines(pts1_[:,None], 1, _F).reshape(-1,3)
    for pt1, l2 in zip(pts1_, lines2): 
        color = tuple(int(x) for x in np.random.randint(0, 255, 3))
        
        img2_ = _draw_line(img2_, l2, color, linewidth)
        
        img1_ = cv2.circle(img1_, tuple(int(x) for x in pt1), radius=markersize, color=color, thickness=-1)
        img1_ = cv2.circle(img1_, tuple(int(x) for x in pt1), radius=markersize//2, color=(0,0,0), thickness=-1)        
        
    lines1 = cv2.computeCorrespondEpilines(pts2_[:,None], 2, _F).reshape(-1,3)
    for pt2, l1 in zip(pts2_, lines1): 
        color = tuple(int(x) for x in np.random.randint(0, 255, 3))
        
        img1_ = _draw_line(img1_, l1, color, linewidth)

        img2_ = cv2.circle(img2_, tuple(int(x) for x in pt2), radius=markersize, color=color, thickness=-1)
        img2_ = cv2.circle(img2_, tuple(int(x) for x in pt2), radius=markersize//2, color=(0,0,0), thickness=-1)         
        
    return img1_, img2_

def visualise_epilines_pair(image1, image2, 
                            points1, points2,
                            calibration1, calibration2,
                            linewidth=2, markersize=20):

    K1 = np.float64(calibration1['K'])
    dist1 = np.float64(calibration1['dist'])
    R1 = np.float64(calibration1['R'])
    t1 = np.float64(calibration1['t'])
    
    K2 = np.float64(calibration2['K'])    
    dist2 = np.float64(calibration2['dist'])
    R2 = np.float64(calibration2['R'])
    t2 = np.float64(calibration2['t'])

    img1_undist = cv2.undistort(image1.copy(), K1, dist1, None, K1)
    img2_undist = cv2.undistort(image2.copy(), K2, dist2, None, K2)

    pts1_undist = cv2.undistortPoints(np.reshape(points1,(-1,1,2)), K1, dist1, P=K1).reshape(-1,2)
    pts2_undist = cv2.undistortPoints(np.reshape(points2,(-1,1,2)), K2, dist2, P=K2).reshape(-1,2)   

    F = fundamental_from_poses(K1, R1, t1, K2, R2, t2)

    idx = np.arange(min(pts1_undist.shape[0],pts2_undist.shape[0]))
    np.random.shuffle(idx)
    img1_, img2_ = draw_epilines(img1_undist, img2_undist, pts1_undist[idx[:50]], pts2_undist[idx[:50]],
                                 F, None, linewidth=linewidth, markersize=markersize)

    hmin = np.minimum(img1_.shape[0], img2_.shape[0])
    return np.hstack([img1_[:hmin],img2_[:hmin]])