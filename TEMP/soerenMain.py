'''
Recurrent network example.  Trains a bidirectional vanilla RNN to output the
sum of two numbers in a sequence of random numbers sampled uniformly from
[0, 1] based on a separate marker sequence.
'''

from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne


# Min/max sequence length
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 20
# Number of training sequences in each batch
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100





def main():
    sym_y = T.imatrix('target_output')
    sym_mask = T.matrix('mask')
    sym_x = T.tensor3()

    num_epochs = 20
    batch_size = 10
    num_classes = 3

    n_samples = 1000
    n_inputs = 10
    seq_len = 50

    X = np.random.random((n_samples, seq_len, n_inputs)).astype('float32')
    y = X.sum(axis=-1)
    y = y.flatten()
    for i, y_i in enumerate(y):
        if y_i < 2:
            y[i] = 0
        elif y_i < 4:
            y[i] = 1
        else:
            y[i] = 2

    y = y.reshape((n_samples, seq_len))

    # shift
    y = np.pad(y, mode='constant', pad_width=((0,0), (5,0)))
    y = y[:, :-5]
    y = y.astype('int32')



    print("Building network ...")
    l_in = lasagne.layers.InputLayer(shape=(batch_size, seq_len, n_inputs))
    l_forward = lasagne.layers.LSTMLayer(l_in, N_HIDDEN)
    l_backward = lasagne.layers.LSTMLayer(l_in, N_HIDDEN,backwards=True)
    l_sumz = lasagne.layers.ElemwiseSumLayer([l_forward, l_backward])
#    l_sumz = lasagne.layers.ConcatLayer([l_forward,l_backward], axis=-1)
#    print("testing ...")
#    out = lasagne.layers.get_output(l_sumz, sym_x)
#    testvar = np.ones((batch_size,seq_len,n_inputs)).astype('float32')
#    john = out.eval({sym_x: testvar})
#    print(john.shape)
#    print("done testing ...")
    
   

#    l_sum = lasagne.layers.ConcatLayer([l_forward, l_backward])
    l_reshape = lasagne.layers.ReshapeLayer(
        l_sumz, (batch_size*seq_len, N_HIDDEN))
    # Our output layer is a simple dense connection, with 1 output unit
    l_recurrent_out = lasagne.layers.DenseLayer(
        l_reshape, num_units=num_classes, nonlinearity=lasagne.nonlinearities.softmax)

    # Now, reshape the output back to the RNN format
    l_out = lasagne.layers.ReshapeLayer(
        l_recurrent_out, (batch_size, seq_len, num_classes))

    # lasagne.layers.get_output produces a variable for the output of the net
    out_train = lasagne.layers.get_output(
        l_out, sym_x, mask=sym_mask, deterministic=False)

    out_eval = lasagne.layers.get_output(
        l_out, sym_x, mask=sym_mask, deterministic=True)

    probs_flat = out_train.reshape((-1, num_classes))

    cost = T.nnet.categorical_crossentropy(probs_flat, sym_y.flatten())
    lambda_reg = 0.0005
    params = lasagne.layers.get_all_params(l_out,regularizable=True)
    reg_term = sum(T.sum(p**2) for p in params)
    cost = T.sum(cost*sym_mask.flatten()) / T.sum(sym_mask) + lambda_reg * reg_term

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out)
    # Compute SGD updates for training
    print("Computing updates ...")
    all_grads = T.grad(cost, all_params)
    updates, norm_calc = lasagne.updates.total_norm_constraint(all_grads, max_norm=20, return_norm=True)

    updates = lasagne.updates.rmsprop(updates, all_params, LEARNING_RATE)
    # Theano functions for training and computing cost
    print("Compiling functions ...")

    # Use this for training (see deterministic = False above)
    train = theano.function(
        [sym_x, sym_y, sym_mask], [cost, out_train, norm_calc], updates=updates)

    # use this for eval (deterministic = True + no updates)
    eval = theano.function([sym_x, sym_mask], out_eval)



    num_batches = n_samples // batch_size

    for epoch in range(num_epochs):
        curcost = 0
	for i in range(num_batches):
            idx = range(i*batch_size, (i+1)*batch_size)
            x_batch = X[idx]
            y_batch = y[idx]
            mask_batch = np.ones_like(y_batch).astype('float32')  # dont do this in your code!!!!!
            train_cost, out, norm = train(x_batch, y_batch, mask_batch)
            print(norm)
            curcost = curcost+train_cost
        curcost = curcost/num_batches    
	print(curcost)
            

if __name__ == '__main__':
    main()