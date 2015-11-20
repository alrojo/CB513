import lasagne
import parmesan
import theano.tensor as T

#validate_every = 40
start_saving_at = 0
save_every = 20
#write_every_batch = 10

epochs = 300
batch_size = 64
N_L1 = 200
N_LSTM_F = 400
N_LSTM_B = 400
N_L2 = 200
n_inputs = 42
num_classes = 8
seq_len = 700
optimizer = "rmsprop"
lambda_reg = 0.0001
cut_grad = 20

learning_rate_schedule = {
    0: 0.001,
    150: 0.0001,
    175: 0.00001,
}

def batchnormlayer(l, num_units, nonlinearity, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.)):
    l = lasagne.layers.DenseLayer(l, num_units=num_units, W=W, b=b, nonlinearity=None)
    l = parmesan.layers.NormalizeLayer(l)
    l = parmesan.layers.ScaleAndShiftLayer(l)
    l = lasagne.layers.NonlinearityLayer(l,nonlinearity=nonlinearity)
    return l

def build_model():
    # 1. Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, seq_len, n_inputs))
    batchsize, _, _ = T.shape(l_in.input_var)
    # 2. First Dense Layer    
    l_reshape_a = lasagne.layers.ReshapeLayer(
        l_in, (batchsize*seq_len,n_inputs))
    l_1_batchnorm = batchnormlayer(l=l_reshape_a, num_units=N_L1, nonlinearity=lasagne.nonlinearities.rectify)
    l_reshape_b = lasagne.layers.ReshapeLayer(
        l_1_batchnorm, (batchsize, seq_len, N_L1))
    # 3. LSTM Layers
    l_forward = lasagne.layers.LSTMLayer(l_reshape_b, N_LSTM_F)
    l_backward = lasagne.layers.LSTMLayer(l_reshape_b, N_LSTM_B, backwards=True)
    #Concat layer
    l_sum = lasagne.layers.ConcatLayer(incomings=[l_forward, l_backward], axis=2)
    # 4. Second Dense Layer
    l_reshape_c = lasagne.layers.ReshapeLayer(
        l_sum, (batchsize*seq_len, N_LSTM_F+N_LSTM_B))
    l_2_batchnorm = batchnormlayer(l=l_reshape_c, num_units=N_L2, nonlinearity=lasagne.nonlinearities.rectify)
    # 5. Output Layer
    l_recurrent_out = lasagne.layers.DenseLayer(
        l_2_batchnorm, num_units=num_classes, nonlinearity=lasagne.nonlinearities.softmax)

    # Now, reshape the output back to the RNN format
    l_out = lasagne.layers.ReshapeLayer(
        l_recurrent_out, (batchsize, seq_len, num_classes))

    return l_in, l_out
