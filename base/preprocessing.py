'''
Created on Mar 5, 2015

@author: Daniel
'''
from skimage.color.colorconv import rgb2gray
from skimage.exposure._adapthist import equalize_adapthist
from skimage.transform._warps import resize
import matplotlib.pyplot as plt
from skimage.feature._hog import hog
from skimage.exposure.exposure import rescale_intensity
from skimage.transform.pyramids import pyramid_gaussian
from skimage.io._io import imread
from skimage.util.shape import view_as_windows
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d, extract_patches
import matplotlib.pyplot as plt
from sklearn.cluster.k_means_ import KMeans
from skimage.filters.rank.generic import median
from skimage.morphology.selem import disk

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
    img = median(img, disk(5))
    
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
                     dataset):
    features = run(current_patch)
    predicted_class = supervised.classifier.predict(features)
    class_name = dataset.classes[predicted_class]
    return class_name

def find_keys(seq):
    keys = {}
    for e in seq:
        keys[e] = 1
    return keys.keys()  

def search_numbers(img, supervised, dataset):
    rows = 88
    cols = 64
    step = 1
    img = imread(img)
    pyramid = tuple(pyramid_gaussian(img, downscale=2))
    centers = None
    labels = list()
    count = 0
    letters = list()
    for current_image in reversed(pyramid):
        (img_rows, img_cols, _) = current_image.shape
        if img_rows < rows or img_cols < cols:
            continue
        factor = img.shape[0] / img_rows
        windows = extract_patches(current_image,
                                  (rows, cols, 3),
                                  step)
        (y_index, x_index) = windows.shape[0:2]
        for y in xrange(y_index):
            for x in xrange(x_index):
                current_patch = windows[y, x, 0, :]
                features = run(current_patch)
                predicted_class = supervised.classifier.predict(features)
                class_name = dataset.classes[predicted_class]
                if class_name is not dataset.classes[-1]:
                    # Character detected
                    # Find center
                    center = find_center(np.array([y * step, x * step]),
                                         np.array([rows, cols]))
                    center = center * factor
                    centers = [center] if centers is None \
                              else np.append(centers, [center], axis=0)
                    labels.append(class_name)
                    
        
        count += 1
        if count == 1:
            break
    labels = np.array(labels)
    unsupervised = KMeans(n_clusters=6, n_jobs=6)
    unsupervised.fit(centers)
    labels_cluster = unsupervised.fit_predict(centers)
    for id_cluster in xrange(6):
        indexes = np.where(labels_cluster == id_cluster)
        labels_per_cluster = labels[indexes]
        keys = find_keys(labels_per_cluster)
        number = 0
        letter_result = ''
        for key in keys:
            counted = len(np.where(labels_per_cluster == key)[0])
            if counted > number:
                letter_result = key
                number = counted
        letters.append((letter_result, unsupervised.cluster_centers_[id_cluster][1]))
    return sorted(letters, key=lambda center: center[1])
                
                