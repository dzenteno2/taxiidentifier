'''
Created on Mar 6, 2015

@author: Daniel
'''
from base.dataset import Dataset
from sklearn.ensemble.forest import RandomForestClassifier
import os
from sklearn.externals import joblib
from scipy.ndimage.io import imread
from base import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.linear_model.logistic import LogisticRegression

class Supervised(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.directory = '/Users/Daniel/Documents/HackCDMX/model'
        self.classifier = None
        
    def fit(self):
        dataset = Dataset()
        (training_data, labels) = dataset.get_train_data()
        rf = RandomForestClassifier(n_estimators=100, n_jobs=6)
        #lr = LogisticRegression(C=6000)
        classifier = Pipeline(steps=[('rf', rf)])
        classifier.fit(training_data, labels)
        f = os.path.join(self.directory, 'random_forest')
        joblib.dump(classifier, f)
        
    def load(self):
        f = os.path.join(self.directory, 'random_forest')
        self.classifier = joblib.load(f)
        
    def apply(self, img_path):
        dataset = Dataset()
        img = imread(img_path)
        features = preprocessing.run(img)
        predicted_label = self.classifier.predict(features)
        
        return dataset.classes[predicted_label]
        
        