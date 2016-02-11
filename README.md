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
7. DenseLayer2([LSTMLayerF, LSTMLayerB], dropout=0.5)
8. OutLayer(DenseLayer2)

Gradients are further normalized and probabilities cutted. RMSProps is used and L2=0.0001

## Project elaboration: Start Juli 2015 - still ongoing

This project is a continuation of Søren Sønderby's (github.com/skaae) previous results on CB513: http://arxiv.org/abs/1412.7828, supervised under Ole Winther (cogsys.imm.dtu.dk/staff/winther/).

My project was to recreate Søren's results and test: Convolutional layers across time, L2, "vertical" links (feeding forward LSTM to backwards LSTM), batchnormalization, different optimizers etc.

It took me approximately 3 months (with grid search of 200-300 models) before I managed to achieve similar results to Søren (apperently a DropoutLayer in the DenseLayer before the first LSTM messed with the model performance, which is why it took so long to get Søren's results).

After achieving similar performance I started applying the various "new" techniques to my neural network. It took another 100-150 models gridsearching various combination which led to the model in "Best network elaborated". Notice the use of skip layer and that the convolutions are all on the input.

I would like to test BN-RNN by baidu (http://arxiv.org/abs/1512.02595) and used Batch Normalization after the LSTM when lasagne starts to support masked Batch Norms.

The article will be submitted as a Methods paper to Nature together with some other research from Ole Winther's lab. My first draft is in the article folder.

Next up: Make a 10 model average and finish the ImageNet12 article like drawings of the final network.
