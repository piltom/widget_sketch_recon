import pandas as pd
import numpy as np
from skimage import io, filters
from skimage import transform as tf
from sklearn import preprocessing
import os

dataset_df = pd.DataFrame()
NUM_DISCR_ANGLES = 5 #number of possible discrete angles
discrete_angles = np.linspace(-np.pi/2,np.pi/2, num=NUM_DISCR_ANGLES)
folders = ['img/checkbox','img/pushbutton', 'img/input','img/radiobutton', 'img/roundimage', 'img/slider', 'img/sqimage', 'img/switch', 'img/text']
for folder in folders:
    target_name=folder.split('/')[-1]
    for dirname, _, filenames in os.walk(folder):
        for filename in filenames:
            img1=io.imread(os.path.join(dirname, filename))
            hspace, angles, dists=tf.hough_line(img1)
            hspace, angles, dists = tf.hough_line_peaks(hspace, angles, dists)
            angle_count = np.zeros(NUM_DISCR_ANGLES)
            angle_dist_sum = np.zeros(NUM_DISCR_ANGLES)
            for angle,dist in zip(angles, dists):
                dif=np.abs(discrete_angles-angle)
                idx=np.argmin(dif)
                angle_count[idx]+=1
                angle_dist_sum[idx]+=abs(dist)
            dataset_df=dataset_df.append(pd.DataFrame([[filename, target_name, *angle_count, *angle_dist_sum]]), ignore_index=True)

dataset_df.rename({0:"filename",1:"target"},axis='columns',inplace=True)
scaler = preprocessing.MinMaxScaler()
dataset_df[dataset_df.columns[2:]] = scaler.fit_transform(dataset_df[dataset_df.columns[2:]])
dataset_df.to_pickle("dataset_img_hough5_doublefullsize.pkl")
