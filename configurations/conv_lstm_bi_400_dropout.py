import lasagne

#validate_every = 40
start_saving_at = 0
#write_every_batch = 10

epochs = 200
batch_size = 128
N_HIDDEN = 400
n_inputs = 42
num_classes = 8
seq_len = 700

conv_a_filter_size = 3
conv_a_num_filters = 16

learning_rate_schedule = {
    0: 0.001,
    150: 0.0005,
    400: 0.00025,
}

def build_model():
    l_in = lasagne.layers.InputLayer(shape=(None, seq_len, n_inputs))
#    batch_size, seq_len, _ = l_in.input_var.shape
    l_dim_a = lasagne.layers.DimshuffleLayer(
        l_in, (0,2,1))
    l_conv_a = lasagne.layers.Conv1DLayer(
        incoming=l_dim_a, num_filters=conv_a_num_filters, border_mode='same',
        filter_size=conv_a_filter_size, stride=1, nonlinearity=lasagne.nonlinearities.rectify)
    l_dim_b = lasagne.layers.DimshuffleLayer(
        l_conv_a, (0,2,1))
    l_forward = lasagne.layers.LSTMLayer(l_dim_b, N_HIDDEN)
#    l_vertical = lasagne.layers.ConcatLayer([l_in,l_forward], axis=2)
#    l_sum = lasagne.layers.ConcatLayer([l_in, l_forward],axis=-1)
    l_backward = lasagne.layers.LSTMLayer(l_dim_b, N_HIDDEN, backwards=True)
    
#    out = lasagne.layers.get_output(l_sum, sym_x)
#    out.eval({sym_x: })
    l_sum = lasagne.layers.ElemwiseSumLayer([l_forward, l_backward])
    l_reshape = lasagne.layers.ReshapeLayer(
        l_sum, (batch_size*seq_len, N_HIDDEN))
    # Our output layer is a simple dense connection, with 1 output unit
    l_recurrent_out = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(l_reshape, p=0.5), num_units=num_classes, nonlinearity=lasagne.nonlinearities.softmax)

    # Now, reshape the output back to the RNN format
    l_out = lasagne.layers.ReshapeLayer(
        l_recurrent_out, (batch_size, seq_len, num_classes))

    return l_in, l_out