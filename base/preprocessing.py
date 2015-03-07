'''
Created on Mar 5, 2015

@author: Daniel
'''
from skimage.color.colorconv import rgb2gray
from skimage.exposure._adapthist import equalize_adapthist
from skimage.transform._warps import resize
from skimage.feature._hog import hog
from skimage.transform.pyramids import pyramid_gaussian
from skimage.io._io import imread
import numpy as np
from sklearn.feature_extraction.image import extract_patches
import matplotlib.pyplot as plt
from sklearn.cluster.k_means_ import KMeans
from skimage.segmentation.slic_superpixels import slic
from skimage.measure._regionprops import regionprops
from skimage.util.shape import view_as_windows
from sklearn.externals.joblib.parallel import Parallel, delayed
import tempfile
import os
from sklearn.externals import joblib
import shutil

def run(img):
    rows = 88
    cols = 64
    # Resize image to standard size
    img = resize(img, (rows, cols))
    
    
    # Equalize histogram
    img = equalize_adapthist(img, ntiles_x=8,
                             ntiles_y=11)
    
    
    # Histogram of Oriented Gradients
    img = rgb2gray(img)
    
    fd, hog_image = hog(img, orientations=8,
                        pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1),
                        visualise=True,
                        normalise=True)
    
#         _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
#         ax1.axis('off')
#         ax1.imshow(img, cmap=plt.cm.gray)
#         
#         hog_image_rescaled = rescale_intensity(hog_image,
#                                                in_range=(0, 0.02))
#         
#         ax2.axis('off')
#         ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
#         plt.show()
    
    return fd

def find_center(origin, window_size):
    return window_size / 2.0 + origin


def parallel_predict(current_patch,
                     supervised,
                     y,
                     x,
                     window_size,
                     step):
    features = run(current_patch)
    predicted_class = supervised.classifier.predict(features)
    return predicted_class[0]

def find_keys(seq):
    keys = {}
    for e in seq:
        keys[e] = 1
    return keys.keys()  

def search_numbers(img, supervised, dataset):
    step = 10
    img = imread(img)
    segments_slic = slic(img,
                         n_segments=10,
                         sigma=0.5,
                         convert2lab=True,
                         enforce_connectivity=False,
                         slic_zero=True)
    regions_prop = regionprops(segments_slic)
    window_sizes = list()
    for region_prop in regions_prop:
        bbox = np.array(region_prop.bbox)
        selected_region = img[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
        features = run(selected_region)
        predicted_class = supervised.classifier.predict(features)
        if dataset.classes[predicted_class] is not dataset.classes[-1]:
            # Desired window found
            window_size = (bbox[2] - bbox[0],
                           bbox[3] - bbox[1])
            window_sizes.append(window_size)
            
    window_sizes = np.array(window_sizes)
    window_size = np.mean(window_sizes, axis=0)
    window_size = window_size.astype(np.int16)
    
    # View as window
    windows = extract_patches(img, (window_size[0],
                                    window_size[1],
                                    3),
                              step)
    
    (y_index, x_index) = windows.shape[0:2]
    
    features = Parallel(n_jobs=-1
                        )(delayed(run)(windows[y, x, 0, :])
                          for y in xrange(y_index)
                            for x in xrange(x_index))
    
    centers = Parallel(n_jobs=-1
                       )(delayed(find_center)(np.array([y * step,
                                                        x * step]),
                                              window_size)
                         for y in xrange(y_index)
                            for x in xrange(x_index))
                        
    features = np.array(features)
    centers = np.array(centers)
    labels = supervised.classifier.predict(features)
    
    # Remove data from background
    indexes_background = np.where(labels != len(dataset.classes) - 1)[0]
    features = features[indexes_background]
    centers = centers[indexes_background]
    labels = labels[indexes_background]
    
    unsupervised = KMeans(n_clusters=6, n_jobs=-1)
    unsupervised.fit(centers)
    clusters = unsupervised.fit_predict(centers)
    label_per_cluster = list()
    for id_cluster in xrange(unsupervised.n_clusters):
        indexes = np.where(clusters == id_cluster)
        labels_per_cluster = labels[indexes]
        unique_elements = np.unique(labels_per_cluster)
        min_element = 0
        max_element = np.max(unique_elements)
        counted_labels = np.bincount(labels_per_cluster)
        generated_labels = np.arange(min_element,
                                     max_element + 1)
        amax = np.argmax(counted_labels)
        label_per_cluster.append(generated_labels[amax])
            
    label_per_cluster = np.array(label_per_cluster)
    
    # Order cluster by x
    x_centers = unsupervised.cluster_centers_[:, 1]
    ordered_index = np.argsort(x_centers)
    label_per_cluster = label_per_cluster[ordered_index]
    
    numbers = ''
    for l in label_per_cluster:
        numbers+=dataset.classes[l]
        
    return numbers
        
    
                
                