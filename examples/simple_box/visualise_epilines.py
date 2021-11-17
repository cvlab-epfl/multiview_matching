import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import imageio

from multiview_matching.utils import utils
from multiview_matching.utils import vis

detections = utils.json_read("detections.json")

calibration = utils.json_read("calibration.json")

filenames = {'cam0':'frame_0001.jpg',
             'cam1':'frame_0003.jpg',
             'cam2':'frame_0010.jpg'}

view1 = 'cam0'
view2 = 'cam1'

img1 = imageio.imread(filenames[view1])
img2 = imageio.imread(filenames[view2])

points1 = detections[0][view1]
points2 = detections[0][view2]

image = vis.visualise_epilines_pair(img1, img2, points1, points2,
                                    calibration[view1], calibration[view2],
                                    linewidth=10, markersize=30)

plt.figure(figsize=(10,4))
plt.imshow(image)
plt.show()
plt.savefig("epilines_{}_{}.jpg".format(view1, view2), dpi=200)