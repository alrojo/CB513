import lasagne

#validate_every = 40
start_saving_at = 0
#write_every_batch = 10

epochs = 200
batch_size = 128
N_HIDDEN = 300
n_inputs = 42
num_classes = 8
seq_len = 700


learning_rate_schedule = {
    0: 0.001,
    150: 0.0005,
    400: 0.00025,
}

def build_model():
    l_in = lasagne.layers.InputLayer(shape=(None, seq_len, n_inputs))
    l_forward = lasagne.layers.LSTMLayer(l_in, N_HIDDEN)
    l_vertical = lasagne.layers.ConcatLayer([l_in,l_forward], axis=2)
    l_backward = lasagne.layers.LSTMLayer(l_vertical, N_HIDDEN, backwards=True)
    
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
