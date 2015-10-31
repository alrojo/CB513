# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 22:51:13 2015

@author: s145706
"""
import sys
import numpy as np

if not (len(sys.argv) == 2):
    sys.exit("Usage: python debug_metadata.py <metadata_path>")

metadata_path = sys.argv[1]

print "Loading metadata file %s" % metadata_path
metadata = np.load(metadata_path)
acc_eval_valid = metadata['accuracy_eval_valid']
acc_train = metadata['accuracy_train']
acc_eval_test = metadata['accuracy_eval_test']

print "Valid acc"
i = 1
for valid, train, test in zip(acc_eval_valid, acc_train, acc_eval_test):
    print "%d: %.5f %.5f %.5f" %(i,train, valid, test)
    i = i+1
