# RNNProteins
World record on cb513 dataset using cullpdb+profile\_6133\_filtered (68.9% on Q8 with best single performing model), available at:
http://www.princeton.edu/~jzthree/datasets/ICML2014/

By Alexander Rosenberg Johansen

previous best single model results: 68.3% Q8 by: [Deep CNF](http://www.nature.com/articles/srep18962)

## Reproducing results

### Installation
Please refer to [lasagne's](https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne-on-Ubuntu-14.04) for installation and setup of GPU environment on an Ubuntu 14.04 machine.


### Getting repository
Go to desired folder for repo and type in terminal.
>> git clone https://github.com/alrojo/RNNProteins.git

### Training models

Use train.py when data is unzipped in data folder, i.e.
>> python train.py baseline_L2

## Best network elaborated
https://github.com/alrojo/RNNProteins/blob/master/configurations/avg1.py

1. InputLayer
2. 3x ConvLayer(InputLayer, filter_size=3-5-7) + Batch Normalization
4. DenseLayer1([ConcatLayer, InputLayer]) + Batch Normalization
5. LSTMLayerF(DenseLayer1, Forward)
6. LSTMLayerB([DenseLayer1, LSTMLayerF], Backward)
7. DenseLayer2([LSTMLayerF, LSTMLayerB], dropout=0.5)
8. OutLayer(DenseLayer2)

Gradients are further normalized if too large and probabilities cutted. RMSProps is used and L2=0.0001

## Project elaboration: Start Juli 2015 - still ongoing

This project is a continuation of Søren Sønderby's (github.com/skaae) previous results on CB513: http://arxiv.org/abs/1412.7828, supervised under Ole Winther (cogsys.imm.dtu.dk/staff/winther/).

My project was to recreate Søren's results and test: Convolutional layers across time, L2, "vertical" links (feeding forward LSTM to backwards LSTM), batchnormalization, different optimizers etc.

It took me approximately 3 months (with grid search of 200-300 models) before I managed to achieve similar validation results to Søren (apperently a DropoutLayer in the DenseLayer before the first LSTM messed with the model performance, which is why it took so long to get Søren's results).

After achieving similar performance I started applying the various "new" techniques to my neural network. It took another 100-150 models gridsearching various combination which led to the model in "Best network elaborated".
Running final test (which will be reported in a methods paper later) led to a performance increase by 1.5% compared to Sørens results. Notice the use of skip layer and that the three convolutional layers are all on the input.

Other possible routes for improved results could be found using batch normalized RNNs by baidu (http://arxiv.org/abs/1512.02595) and used Batch Normalization after the LSTM when lasagne starts to support masked Batch Norms.

The article will be submitted as a Methods paper to Nature together with some other research from Ole Winther's lab. My first draft is in the article folder.

Next up: Make a 10 model average and finish the ImageNet12 article like drawings of the final network.
