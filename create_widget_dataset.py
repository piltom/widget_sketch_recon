import pandas as pd
import numpy as np
from skimage import io
from skimage.feature import ORB
import os

descriptor_extractor = ORB(n_keypoints=200)
dataset_df = pd.DataFrame()

folders = ['img/checkbox','img/pushbutton', 'img/input','img/radiobutton', 'img/roundimage', 'img/slider', 'img/sqimage', 'img/switch', 'img/text']
for folder in folders:
    target_name=folder.split('/')[-1]
    for dirname, _, filenames in os.walk(folder):
        for filename in filenames:
            img1=io.imread(os.path.join(dirname, filename))
            descriptor_extractor.detect_and_extract(img1)
            keypoints1 = descriptor_extractor.keypoints
            descriptors1 = descriptor_extractor.descriptors
            descriptors_flat=np.array(descriptors1).flatten()
            dataset_df=dataset_df.append(pd.DataFrame([[filename, target_name, *descriptors_flat]]), ignore_index=True)
dataset_df.rename({0:"filename",1:"target"},axis='columns',inplace=True)
dataset_df.to_pickle("dataset_img_ORB_200.pkl")
