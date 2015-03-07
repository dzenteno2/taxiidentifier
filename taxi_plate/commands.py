'''
Created on Mar 5, 2015

@author: Daniel
'''
from supervised_learning.Supervised import Supervised
from argparse import ArgumentParser
from base import preprocessing
from base.dataset import Dataset


modes = ('train', 'predict')

def train(argv):
    
    # Train model
    supervised = Supervised()
    supervised.fit()
    
def predict(argv):
    parser = ArgumentParser()
    parser.add_argument('image', metavar='IMG',
                        help='The image to process')
    opts = parser.parse_args(argv)
    
    supervised = Supervised()
    supervised.load()
    dataset = Dataset()
    letters = preprocessing.search_numbers(opts.image,
                                 supervised,
                                 dataset)
    
    
    
    print letters
    
    print letters[:, 0]