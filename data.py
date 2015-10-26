import numpy as np
import theano

import utils

##### TRAIN DATA #####
print("Loading train data ...")
X_in = utils.load_gz('data/cullpdb+profile_6133_filtered.npy.gz')
X = np.reshape(X_in,(5534,700,57))
del X_in
X = X[:,:,:]
labels = X[:,:,22:30]
mask = X[:,:,30] * -1 + 1

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

print("Loading splits ...")
##### SPLITS #####
# getting splits (cannot run before splits are made)
#split = np.load("data/split.pkl")

seq_names = np.arange(0,num_seqs)
#np.random.shuffle(seq_names)

X_train = X[seq_names[0:5278]]
X_valid = X[seq_names[5278:5534]]
labels_train = labels[seq_names[0:5278]]
labels_valid = labels[seq_names[5278:5534]]
mask_train = mask[seq_names[0:5278]]
mask_valid = mask[seq_names[5278:5534]]
num_seq_train = np.size(X_train,0)
num_seq_valid = np.size(X_valid,0)
#del split
print("Loading test data ...")
##### TEST DATA #####
X_test_in = utils.load_gz('data/cb513+profile_split1.npy.gz')
X_test = np.reshape(X_test_in,(514,700,57))
del X_test_in
X_test = X_test[:,:,:].astype(theano.config.floatX)
labels_test = X_test[:,:,22:30].astype('int32')
mask_test = X_test[:,:,30].astype(theano.config.floatX) * -1 + 1

a = np.arange(0,21)
b = np.arange(35,56)
c = np.hstack((a,b))
X_test = X_test[:,:,c]
# Setting X
num_seq_test = np.size(X_test,0)
del a, b, c

## DUMMY -> CONCAT ##
vals = np.arange(0,8)
labels_new = np.zeros((num_seq_test,seqlen))
for i in xrange(np.size(labels_test,axis=0)):
    labels_new[i,:] = np.dot(labels_test[i,:,:], vals)
labels_new = labels_new.astype('int32')
labels_test = labels_new

### ADDING BATCH PADDING ###

X_add = np.zeros((126,seqlen,d))
label_add = np.zeros((126,seqlen))
mask_add = np.zeros((126,seqlen))

X_test = np.concatenate((X_test,X_add), axis=0).astype(theano.config.floatX)
labels_test = np.concatenate((labels_test, label_add), axis=0).astype('int32')
mask_test = np.concatenate((mask_test, mask_add), axis=0).astype(theano.config.floatX)
