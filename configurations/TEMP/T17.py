import lasagne
import parmesan

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
optimizer = "adagrad"
lambda_reg = 0.0001
cut_grad = 20
batch_norm = True

learning_rate_schedule = {
    0: 0.011,
    150: 0.005,
    175: 0.0025,
}

if batch_norm:
    def DLayer(l, num_units, nonlinearity, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.)):
        l = lasagne.layers.DenseLayer(l, num_units=num_units, W=W, b=b, nonlinearity=None)
        l = parmesan.layers.NormalizeLayer(l)
        l = parmesan.layers.ScaleAndShiftLayer(l)
        l = lasagne.layers.NonlinearityLayer(l,nonlinearity=nonlinearity)
        return l
else:
    def DLayer(l, num_units, nonlinearity, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.)):
        l = lasagne.layers.DenseLayer(l, num_units=num_units, W=W, b=b, nonlinearity=None)
        l = lasagne.layers.NonlinearityLayer(l,nonlinearity=nonlinearity)
        return l

def build_model():
    # 1. Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, seq_len, n_inputs))
    # 2. First Dense Layer    
    l_reshape_a = lasagne.layers.ReshapeLayer(
        l_in, (batch_size*seq_len,n_inputs))
    l_1 = DLayer(l=l_reshape_a, num_units=N_L1, nonlinearity=lasagne.nonlinearities.rectify)
#    l_1_batchnorm = lasagne.layers.DenseLayer(l_reshape_a, num_units=N_L1)
    l_reshape_b = lasagne.layers.ReshapeLayer(
        l_1, (batch_size, seq_len, N_L1))
    # 3. LSTM Layers
    l_forward = lasagne.layers.LSTMLayer(l_reshape_b, N_LSTM_F)
    l_backward = lasagne.layers.LSTMLayer(l_reshape_b, N_LSTM_B, backwards=True)
    #Concat layer
    l_sum = lasagne.layers.ConcatLayer(incomings=[l_forward, l_backward], axis=2)
    # 4. Second Dense Layer
    l_reshape_c = lasagne.layers.ReshapeLayer(
        l_sum, (batch_size*seq_len, N_LSTM_F+N_LSTM_B))
    l_2 = DLayer(l=l_reshape_c, num_units=N_L2, nonlinearity=lasagne.nonlinearities.rectify)
#    l_2_batchnorm = lasagne.layers.DenseLayer(l_reshape_c, num_units=N_L2)
    # 5. Output Layer
    l_recurrent_out = lasagne.layers.DenseLayer(
        l_2, num_units=num_classes, nonlinearity=lasagne.nonlinearities.softmax)

    # Now, reshape the output back to the RNN format
    l_out = lasagne.layers.ReshapeLayer(
        l_recurrent_out, (batch_size, seq_len, num_classes))

    return l_in, l_out
