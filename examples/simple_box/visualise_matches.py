import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import imageio
import numpy as np
import itertools

from multiview_matching.utils import utils

results = utils.json_read("detections_3d_n4_dmax10.0_dnone20.0_sigma10_1.json")

positions_3d, cliques, costs = results[0]

views = list(set(itertools.chain(*[list(c.keys()) for c in cliques])))

filenames = {'cam0':'frame_0001.jpg',
             'cam1':'frame_0003.jpg',
             'cam2':'frame_0010.jpg'}

colors = [[0,255,0],
          [100,100,255], [255,255,0],
          [0,255,255], [255,0,255],
          [255,255,255], [0,0,0],
          [128,128,128], [50,128,50]]+[np.random.randint(0,255,3).tolist() for _ in range(30)]

for view in views:
    
    img = imageio.imread(filenames[view])
    
    plt.figure(figsize=(10,4))
    plt.title(view)
    plt.imshow(img)
    
    for clique, color in zip(cliques, colors):
        if view in clique:
            p = clique[view]
        plt.plot(*p, '.', color=tuple(x/255. for x in color), markersize=20)
        
    plt.savefig("matches_{}.jpg".format(view), dpi=200)