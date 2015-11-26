import lasagne

# protvec LSTM

#validate_every = 40
start_saving_at = 0
save_every = 20
#write_every_batch = 10

batch_norm = False
epochs = 400
batch_size = 64
N_LSTM_F = 200
N_LSTM_B = 200
n_inputs = 142
num_classes = 8
seq_len = 700
optimizer = "rmsprop"
lambda_reg = 0.0
cut_grad = 20

learning_rate_schedule = {
    0: 0.001,
}

def build_model():
    # 1. Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, seq_len, n_inputs))
    # 2. Bi-LSTM
    l_forward = lasagne.layers.LSTMLayer(l_in, N_LSTM_F)
    l_backward = lasagne.layers.LSTMLayer(l_in, N_LSTM_B, backwards=True)
    l_bi = lasagne.layers.ConcatLayer(incomings=[l_forward, l_backward], axis=2)
    # 3. Outlayer
    l_reshape = lasagne.layers.ReshapeLayer(
        l_bi, (batch_size*seq_len, N_LSTM_F+N_LSTM_B))
    l_recurrent_out = lasagne.layers.DenseLayer(
        l_reshape, num_units=num_classes, nonlinearity=lasagne.nonlinearities.softmax)
    l_out = lasagne.layers.ReshapeLayer(
        l_recurrent_out, (batch_size, seq_len, num_classes))

    return l_in, l_out