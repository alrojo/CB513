# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('PDF')
import numpy as np

metadata_path = "metadata/dump_T26-20151211-105721-380.pkl"
metadata = np.load(metadata_path)

acc_eval_valid = metadata['accuracy_eval_valid']
acc_train = metadata['accuracy_train']
acc_eval_test = metadata['accuracy_eval_test']

matplotlib.pyplot.plot(acc_train, '-', acc_eval_valid, 'r', acc_eval_test, 'g',  linewidth=2.0)

matplotlib.pyplot.savefig('myfig')