#----------------------------------------------------------------------------
# Created By  : Leonardo Citraro leonardo.citraro@epfl.ch
# Date: 2021
# --------------------------------------------------------------------------
import numpy as np
import cv2

def essential_to_fundamental(E, K1, K2):
    F = np.dot(np.linalg.inv(K2.T), np.dot(E, np.linalg.inv(K1)))
    F = F/F[2,2]   
    return F

def fundamental_to_essential(F, K1, K2):
    E = np.dot(K2.T, np.dot(F, K1))    
    E = E/E[2,2]
    return E

def compute_right_epipole(F):
    U, S, V = np.linalg.svd(F)
    e = V[-1]
    return e / e[2]

def compute_left_epipole(F):
    return compute_right_epipole(F.T)

def essential_from_relative_pose(Rd, td):
    tx = np.array([[0, -td[2], td[1]],
                   [td[2], 0, -td[0]],
                   [-td[1], td[0], 0]])

    E = np.dot(tx, Rd)  
    return np.float_(E)

def fundamental_from_relative_pose(Rd, td, K1, K2):
    E = essential_from_relative_pose(Rd, td)
    F = essential_to_fundamental(E, K1, K2)
    return F

def relative_pose(R1, t1, R2, t2):
    Rd = np.dot(R2, R1.T)
    td = t2-np.dot(Rd, t1)    
    return Rd, td

def essential_from_poses(K1, R1, t1, K2, R2, t2):
    Rd, td = relative_pose(R1, t1, R2, t2)
    E = essential_from_relative_pose(Rd, td)
    return E

def fundamental_from_poses(K1, R1, t1, K2, R2, t2):
    Rd, td = relative_pose(R1, t1, R2, t2)
    E = essential_from_relative_pose(Rd, td)
    F = essential_to_fundamental(E, K1, K2)
    return F

def compute_epilines(pts1_undistorted, pts2_undistorted, F):
    
    if pts1_undistorted is None:
        lines2 = []
    else:
        pts1_ = np.float64(pts1_undistorted).reshape(-1,2)
        lines2 = cv2.computeCorrespondEpilines(pts1_[:,None], 1, F).reshape(-1,3)

    if pts2_undistorted is None:
        lines1 = []
    else:        
        pts2_ = np.float64(pts2_undistorted).reshape(-1,2)
        lines1 = cv2.computeCorrespondEpilines(pts2_[:,None], 2, F).reshape(-1,3)
    
    return lines1, lines2

def distance_point_line(p, line):
    return np.abs(line[0]*p[0]+line[1]*p[1]+line[2])/np.sqrt(line[0]**2+line[1]**2)