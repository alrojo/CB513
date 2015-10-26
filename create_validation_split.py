# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:00:27 2015

@author: alexander
"""
import numpy as np 
import cPickle as pickle
import sklearn
import sklearn.cross_validation
import theano

import utils

## From data
print("Loading train data ...")
X_in = utils.load_gz('data/cullpdb+profile_6133_filtered.npy.gz')
X = np.reshape(X_in,(5534,700,57))
del X_in
X = X[:,0:100,:]
labels = X[:,:,22:30]
mask = X[:,:,30]

a = np.arange(0,21)
b = np.arange(35,56)
c = np.hstack((a,b))
X = X[:,:,c]


# getting meta
num_seqs = np.size(X,0)
seqlen = np.size(X,1)
d = np.size(X,2)
num_classes = 8

#### REMAKING LABELS ####
X = X.astype(theano.config.floatX)
mask = mask.astype(theano.config.floatX)
# Dummy -> concat
vals = np.arange(0,8)
labels_new = np.zeros((num_seqs,seqlen))
for i in xrange(np.size(labels,axis=0)):
    labels_new[i,:] = np.dot(labels[i,:,:], vals)
labels_new = labels_new.astype('int32')
labels = labels_new

## The split
TARGET_PATH = "data/split.pkl"

split = sklearn.cross_validation.StratifiedShuffleSplit(data.labels, n_iter=1, test_size=256, random_state=np.random.RandomState(42))
indices_train, indices_valid = iter(split).next()

with open(TARGET_PATH, 'w') as f:
    pickle.dump({
        'indices_train': indices_train,
        'indices_valid': indices_valid,
    }, f, pickle.HIGHEST_PROTOCOL)

print "Split stored in %s" % TARGET_PATH