import lasagne

#validate_every = 40
start_saving_at = 0
save_every = 5
#write_every_batch = 10

epochs = 300
batch_size = 64
N_L1 = 200
N_LSTM_F = 200
N_LSTM_B = 200
N_L2 = 200
n_inputs = 42
num_classes = 8
seq_len = 700
optimizer = "adagrad"
lambda_reg = 0.0001
cut_grad = 20

learning_rate_schedule = {
    0: 0.011,
    250: 0.005,
    275: 0.0025,
}

def build_model():
    # 1. Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, seq_len, n_inputs))
    l_dim_a = lasagne.layers.DimshuffleLayer(
	l_in, (0,2,1))
    l_conv_a = lasagne.layers.Conv1DLayer(
	incoming=l_dim_a, num_filters=16, pad='same',
	filter_size=3, stride=1, nonlinearity=lasagne.nonlinearities.rectify)
    l_dim_b = lasagne.layers.DimshuffleLayer(
	l_conv_a, (0,2,1))
    # 2. First Dense Layer    
    l_reshape_a = lasagne.layers.ReshapeLayer(
        l_dim_b, (batch_size*seq_len, 16))
    l_1 = lasagne.layers.DenseLayer(
        l_reshape_a, num_units=N_L1, nonlinearity=lasagne.nonlinearities.rectify)
    l_reshape_b = lasagne.layers.ReshapeLayer(
        l_1, (batch_size, seq_len, N_L1))
#    batch_size, seq_len, _ = l_in.input_var.shape
    # 3. LSTM Layers
    l_forward = lasagne.layers.LSTMLayer(l_reshape_b, N_LSTM_F)
#    l_vertical = lasagne.layers.ConcatLayer([l_in,l_forward], axis=2)
#    l_sum = lasagne.layers.ConcatLayer([l_in, l_forward],axis=-1)
    l_backward = lasagne.layers.LSTMLayer(l_reshape_b, N_LSTM_B, backwards=True)
    
#    out = lasagne.layers.get_output(l_sum, sym_x)
#    out.eval({sym_x: })
    #Concat layer
    l_sum = lasagne.layers.ConcatLayer(incomings=[l_forward, l_backward], axis=2)
    # 4. Second Dense Layer
    l_reshape_b = lasagne.layers.ReshapeLayer(
        l_sum, (batch_size*seq_len, N_LSTM_F+N_LSTM_B))
    # Our output layer is a simple dense connection, with 1 output unit
    l_2 = lasagne.layers.DenseLayer(
	lasagne.layers.dropout(l_reshape_b, p=0.5), num_units=N_L2, nonlinearity=lasagne.nonlinearities.rectify)
    # 5. Output Layer
    l_recurrent_out = lasagne.layers.DenseLayer(
        l_2, num_units=num_classes, nonlinearity=lasagne.nonlinearities.softmax)

    # Now, reshape the output back to the RNN format
    l_out = lasagne.layers.ReshapeLayer(
        l_recurrent_out, (batch_size, seq_len, num_classes))

    return l_in, l_out
