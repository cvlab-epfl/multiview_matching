#----------------------------------------------------------------------------
# Created By  : Leonardo Citraro leonardo.citraro@epfl.ch
# Date: 2021
# --------------------------------------------------------------------------
import os
import sys
import json
import re
import os
import ast
import cv2
import glob
import pickle
import numpy as np

__all__ = ["json_read", "json_write", "pickle_read", "pickle_write", 
           "mkdir", "sort_nicely", "find_files", "invert_Rt",
           "draw_points", "draw_rectangles", "dict_keys_to_string",
           "dict_keys_from_literal_string", "matching_hungarian"]

def json_read(filename):
    try:
        with open(os.path.abspath(filename)) as f:    
            data = json.load(f)
        return data
    except:
        raise ValueError("Unable to read JSON {}".format(filename))
        
def json_write(filename, data):
    try:
        directory = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.abspath(filename), 'w') as f:
            json.dump(data, f, indent=2)
    except:
        raise ValueError("Unable to write JSON {}".format(filename))   
        
def pickle_read(filename):
    with open(filename, "rb") as f:    
        data = pickle.load(f)
    return data

def pickle_write(filename, data):
    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)        

def mkdir(directory):
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)

def find_files(file_or_folder, hint=None, recursive=False):
    # make sure to use ** in file_or_folder when using recusive
    # ie find_files("folder/**", "*.json", recursive=True)
    import os
    import glob
    if hint is not None:
        file_or_folder = os.path.join(file_or_folder, hint)
    filenames = [f for f in glob.glob(file_or_folder, recursive=recursive)]
    filenames = sort_nicely(filenames)    
    filename_files = []
    for filename in filenames:
        if os.path.isfile(filename):
            filename_files.append(filename)                 
    return filename_files

def dict_keys_to_string(d):
    return {str(key):value for key,value in d.items()}

def dict_keys_from_literal_string(d):
    new_d = {}
    for key,value in d.items():
        if isinstance(key, str):
            try:
                new_key = ast.literal_eval(key)
            except:
                new_key = key
        else:
            new_key = key
        new_d[new_key] = value
    return new_d

def invert_Rt(R, t):
    Ri = R.T
    ti = np.dot(-Ri, t)
    return Ri, ti

def matching_hungarian(pts1, pts2, max_dist=0.5):
    from munkres import Munkres
    
    _pts1 = np.array(pts1)
    _pts2 = np.array(pts2)  
    
    assert _pts1.shape[1]==_pts2.shape[1]

    n1 = _pts1.shape[0]    
    n2 = _pts2.shape[0]
    
    n_max = max(n1, n2)
    
    if n_max==0:
        return [], []

    # building the cost matrix based on the distance between 
    # detections and ground-truth positions
    matrix = np.ones((n_max, n_max))*9999999
    for i in range(n1):    
        for j in range(n2):

            # euclidan distance
            dist = np.sqrt(np.sum((p1-p2)**2 for p1, p2 in zip(_pts1[i], _pts2[j])))
            
            if dist <= max_dist:
                matrix[i,j] = dist

    indexes = Munkres().compute(matrix.copy())
   
    matches1 = []
    matches2 = []
    distances = []
    for i, j in indexes:
        value = matrix[i][j]
        if value <= max_dist:
            matches1.append(i)
            matches2.append(j)
            distances.append(value)

    return matches1, matches2, distances

def undistort_points(points, K, distCoeffs, norm_coord=False, newcameramtx=None):
    points_ = np.reshape(points, (-1,1,2))
    if newcameramtx is None:
        newcameramtx = K
    points_ = cv2.undistortPoints(np.float32(points_), K, distCoeffs, P=newcameramtx, R=None)
    points_ = np.reshape(points_, (-1,2))
    return points_