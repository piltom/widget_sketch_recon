
from skimage import io, filters, measure
from skimage import transform as tf
import matplotlib.pyplot as plt
import numpy as np

NUM_DISCR_ANGLES=7

img1=io.imread('pushbutton/tile002_p.png')

img2 = np.array(img1)

hspace, angles, dists=tf.hough_line(img1)
hspace, angles, dists = tf.hough_line_peaks(hspace, angles, dists)
print(angles*180/np.pi)
print(dists)
print("------")
props=measure.regionprops(img1)
print(props[0].centroid)
centroid=(int(props[0].centroid[0]),int(props[0].centroid[1]))
vcross=0
hcross=0
prevpix=img1[0][centroid[1]]>128
for y in range(1,len(img1)):
    curpix=img1[y][centroid[1]]>128
    img2[y][centroid[1]]=255
    if prevpix!=curpix:
        vcross+=1
    prevpix=curpix
prevpix=img1[centroid[0]][0]>128
for x in range(1,len(img1[0])):
    curpix=img1[centroid[0]][x]>128
    img2[centroid[0]][x]=255
    if prevpix!=curpix:
        hcross+=1
    prevpix=curpix
print(vcross)
print(hcross)
discrete_angles = np.linspace(-np.pi/2,np.pi/2, num=NUM_DISCR_ANGLES)
angle_count = np.zeros(NUM_DISCR_ANGLES)
angle_dist_sum = np.zeros(NUM_DISCR_ANGLES)
for angle,dist in zip(angles, dists):
    dif=np.abs(discrete_angles-angle)
    idx=np.argmin(dif)
    angle_count[idx]+=1
    angle_dist_sum[idx]+=abs(dist)
#print(angle_count)
#print(angle_dist_sum)
fig, ax = plt.subplots()
ax.imshow(img2, cmap=plt.cm.gray)
ax.axis((0, 256, 256, 0))
plt.show()
