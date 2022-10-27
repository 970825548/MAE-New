#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image,ImageFilter,ImageDraw,ImageEnhance
import  random
from tqdm import tqdm
import numpy as np
import sys
from tifffile import TIFF
import tifffile as tiff
import os
import json
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, Model, Sequential
import cv2
isprs_dir = root=r'F:/train_data/Semantic_segmentation/Postdam_t5'


# In[2]:


def predict(img):
    feature = tf.cast(img, tf.float32)
    feature = tf.divide(img, 255.)
    x = tf.divide(tf.subtract(feature, rgb_mean), rgb_std)
    x = tf.expand_dims(feature, axis=0)
    print(x.shape)
    return new_model.predict(x)


# In[3]:


def label2image(pred):
    colormap = np.array(COLORMAP, dtype='float32')
#     colormap = tuple(colormap)
    x = colormap[tf.argmax(pred, axis=-1)]
#     x = tup(x)
    print(pred[0, 160, 200])
    return x


# In[4]:


def predict_images(root=isprs_dir):
    output_dir = r'E:\train_data\Semantic_segmentation\Postdam_t\pre_label'
    plt.figure(figsize=(20,10))
    rows = 2
    cols = 5
    txt_fname = '%s/tifs/%s'%(
    root,'test_data.txt')
    with open(txt_fname,'r') as f:
        image_names = f.read().split()
        pred_images,pred_labels = [None] * len(image_names), [None] * len(image_names)
        images,ndsms = [None] * len(image_names), [None] * len(image_names)
    pred_images = []
    for i,fname in enumerate(image_names):
        img_name = '%s\\srcs\\srcs_train\\%s' % (root, fname)
        ndsm_name = '%s\\ndsms\\ndsms_train\\%s' % (root, fname)

        
        images[i] = tiff.imread(img_name)
        ndsms[i] = tiff.imread(ndsm_name)
        test_images[i] = np.concatenate((images[i], ndsms[i][:,:,None]), axis=2)
        test_images[i] = tf.cast(test_images[i],tf.float32)
        pred_image = predict(test_images[i])
        pred_image = label2image(pred_image)

        pred_images.append(pred_image)
        tiff.imsave('%s.tif'%i, pred_image)
        pred_labels[i] = label_indices(pred_images[i],colormap2label)
        
    return pred_images,pred_labels

