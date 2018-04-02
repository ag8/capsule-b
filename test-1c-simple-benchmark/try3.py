import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from myutils import cfg

# understanding the shape of of the squash nonlinearity
# scale = np.linspace(-5,5,100)
# nonlin = np.square(scale) / (1+ np.square(scale))
# squash = nonlin * (scale / np.abs(scale))
# plt.plot(scale,nonlin)
# plt.plot(scale,squash,'r--')
# plt.show()



# compute the output of the capsule | weight W_ij, activations of a bottom capsule u_i, and coupling coefficients c_ij
# u^_j|i = W_ij * u_i
# s_j = sum_i(c_ij u^_j|i)
# v_j = squash(s_j)
# output the activation vector of a capsule

# let's say I have u_i and W_ij - this is a normal neuron (with an unusual nonlinearity)
# now c_i magick


# c_ij = exp(b_ij) / (sum_k (exp(b_ik))
# computing the coupling coefficient -- how much is the message from neuron i to j correlates with the output of j
# a_ij = u^_j|i * v_j


# ROUTING:
# intialize b_i with 0s
# for r iterations
#     c_i <- softmax(b_i)
#     compute output v_j
#     b_ij <- b_ij + a_ij


# what does routing do ?? -- every bottom layer routes the stuff to the other neuron above that will be activated anyway
# 1. does it have a stable state
# 2. is it stable state single
# 3. does it ever converge to a stable state

# in FC layer, if there is a single neuron that is always activated -- everyone will just send their weights to that neuron

# what is a learnable prior here ?
# is the thing stable -- given a different b it could converge to something else


#def routing(u_hat,r,l):
#    pass


def routing(input, W, b_IJ):
    ''' The routing algorithm.

    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
    Returns:
        A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''

    # W: [num_caps_i, num_caps_j, len_u_i, len_v_j]
    print "running routing algorithm with args:",input, W, b_IJ

    # Eq.2, calc u_hat
    # do tiling for input and W before matmul
    # input => [batch_size, 1152, 10, 8, 1]
    # W => [batch_size, 1152, 10, 8, 16]
    input = tf.tile(input, [1, 1, 10, 1, 1])
    print "tiled input to :",input
    W = tf.tile(W, [cfg.batch_size, 1, 1, 1, 1])
    assert input.get_shape() == [cfg.batch_size, 1152, 10, 8, 1]

    # in last 2 dims:
    # [8, 16].T x [8, 1] => [16, 1] => [batch_size, 1152, 10, 16, 1]
    # tf.tile, 3 iter, 1080ti, 128 batch size: 6min/epoch
    u_hat = tf.matmul(W, input, transpose_a=True)
    print "u_hat:",u_hat
    assert u_hat.get_shape() == [cfg.batch_size, 1152, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(cfg.iter_routing):
        print "in dynamic routing iteration",r_iter
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [1, 1152, 10, 1, 1]
            c_IJ = tf.nn.softmax(b_IJ, dim=2)
            print "c_IJ", c_IJ
            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == cfg.iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                assert s_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                assert v_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
            elif r_iter < cfg.iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                print "tf * c_IJ u_hat_stopped, sj:", c_IJ,u_hat_stopped,s_J
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                print "s_J",s_J
                v_J = squash(s_J)
                print "v_J",v_J
                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 10, 1152, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, 1152, 1, 1, 1])
                u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
                print "u_produce_v = uhat x v_J_tiled", u_produce_v,u_hat_stopped,v_J_tiled
                assert u_produce_v.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]
                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v
    return v_J

