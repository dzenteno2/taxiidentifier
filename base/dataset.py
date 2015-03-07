'''
Created on Mar 5, 2015

@author: Daniel
'''
import os
import glob
from skimage.io._io import imread
import numpy as np
from sklearn.externals.joblib.parallel import Parallel, delayed
from base import preprocessing
from copy import copy

class Dataset(object):
    '''
    classdocs
    '''


    def __init__(self, options=None):
        '''
        Constructor
        '''
        self.classes = ['0', '1', '2', '3', '4',
                        '5', '6', '7', '8', '9',
                        'Background']
        
        self.char_directory = '/Users/Daniel/Downloads/English/Img/GoodImg/Bmp/'
        self.char_directory_prefix = 'Sample'
        self.char_extension = '.png'
        self.background_directory = '/Users/Daniel/Downloads/iccv09Data/images'
        self.background_extension = '.jpg'
        
        
    def get_train_data(self):
        training_data = None
        labels = None
        for id_class, current_class in enumerate(self.classes):
            img_search_path = None
            if current_class is not self.classes[-1]:
                img_search_path = os.path.join(self.char_directory, self.char_directory_prefix)
                img_search_path += '{:03d}'.format(id_class + 1)
                img_search_path = os.path.join(img_search_path, '*' + self.char_extension)
            else:
                img_search_path = os.path.join(self.background_directory,
                                               '*'+self.background_extension)
            img_files = glob.glob(img_search_path)
            current_training_data = Parallel(n_jobs=6
                                     )(delayed(preprocessing.run)(imread(img_file))
                                       for img_file in img_files)
            training_data = current_training_data if training_data is None \
                            else np.append(training_data, current_training_data, axis=0)
                            
            num_elements = len(current_training_data)
            current_labels = np.ones((num_elements), dtype=np.int16) * id_class
            labels = current_labels if labels is None \
                     else np.append(labels, current_labels, axis=0)
                
        return (training_data, labels)
                    
                    

                    