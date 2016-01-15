# RNNProteins
World record on cb513 dataset, available at:
http://www.princeton.edu/~jzthree/datasets/ICML2014/

use train.py when data is unzipped in data folder

## Best network elaborated
https://github.com/alrojo/RNNProteins/blob/master/configurations/avg1.py
1. InputLayer
2. 3x ConvLayer(InputLayer, filter_size=3-5-7) + Batch Normalization
4. DenseLayer1([ConcatLayer, InputLayer]) + Batch Normalization
5. LSTMLayerF(DenseLayer1, Forward)
6. LSTMLayerB([DenseLayer1, LSTMLayerF], Backward)
7. DenseLayer2([LSTMLayerF, LSTMLayerB])
8. OutLayer(DenseLayer2)
