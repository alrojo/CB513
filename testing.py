# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:51:42 2015

@author: s145706
"""

import numpy as np
import theano
import theano.tensor as T
import lasagne as nn
import sys
from datetime import datetime, timedelta
import importlib
import time
import cPickle as pickle
print ("loading data ...")
import data
print ("loading data completed ...")
import utils

if theano.config.device != "gpu":
    with open(("metadata/alextestNO.pkl"), 'w') as f:
                    pickle.dump({
                            'config_name': "john",
                            }, f, pickle.HIGHEST_PROTOCOL)
else:
        with open(("metadata/alextestYES.pkl"), 'w') as f:
                    pickle.dump({
                            'config_name': "john",
                            }, f, pickle.HIGHEST_PROTOCOL)