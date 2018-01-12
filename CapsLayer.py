import tensorflow as tf
import numpy as np



def capsule(input_,num_caps,caps_dims,batch_size,init_sigma = 0.3,name="capsule",bridge_input=True): 
    if bridge_input:
        input_ = bridge(input_,num_caps,name = name+"_bridge")
    
    input_caps = input_[1]
    input_n_dims = input_[2]


    W_init = tf.random_normal(shape=(1, input_caps, num_caps, caps_dims, input_n_dims ),
        stddev=init_sigma, dtype=tf.float32, name=name + "W_init")


    #Wij
    W = tf.Variable(W_init, name=name+"_W")


    W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name=name + "_W_tiled")


    #U hat
    caps_predicted = tf.matmul(W_tiled, input_[0],
                                        name=name + "_predicted")


    #Bij 
    raw_weights_init = tf.random_normal(shape=(1, input_caps, num_caps, 1, 1),
        stddev=init_sigma, dtype=tf.float32, name=name + "_raw_weights")


    raw_weights_var = tf.Variable(raw_weights_init, name=name+"_raw_weights_var")

    raw_weights = tf.tile(raw_weights_var, [batch_size, 1, 1, 1, 1], name=name + "_raw_weights_tiled")
    #Cij
    routing_weights = tf.nn.softmax(raw_weights, dim=2, name=name + "_routing_weights")


    #Sj
    weighted_predictions = tf.multiply(routing_weights, caps_predicted,
                                           name=name + "_weighted_predictions")

    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                                 name=name + "_weighted_sum")

    #Vj
    caps_output_round_1 = squash(weighted_sum, axis=-2,
                                  name=name + "_caps_output_round_1")

    caps_output_round_1_tiled = tf.tile(
        caps_output_round_1, [1, input_caps, 1, 1, 1],
        name=name + "_caps_output_round_1_tiled")

    #Aij
    agreement = tf.matmul(caps_predicted, caps_output_round_1_tiled,
                          transpose_a=True, name=name + "_agreement")

    raw_weights_round_2 = tf.add(agreement, raw_weights,
                                 name=name + "_raw_weights_round_2")


    routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                            dim=2,
                                            name=name + "_routing_weights_round_2")
    weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                               caps_predicted,
                                               name=name + "_weighted_predictions_round_2")


    weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                         axis=1, keep_dims=True,
                                         name=name + "_weighted_sum_round_2")



    caps_output_round_2 = squash(weighted_sum_round_2,
                                  axis=-2,
                                  name=name+"_output_round_2")


    caps_output = caps_output_round_2
    return caps_output,num_caps,caps_dims



def conv_to_caps(input_,num_maps,caps_dims,name="conv",kernel_size=4,strides=2,activation=None):

    kachow = tf.layers.conv2d(input_, name=name, filters = num_maps*caps_dims,
                         kernel_size = kernel_size,strides = strides, padding = "valid",activation=activation)

    conv_n_caps = num_maps * kachow.get_shape().as_list()[1]  * kachow.get_shape().as_list()[2] 
    
    kachow = [kachow,conv_n_caps,caps_dims]
    return kachow

def caps_to_conv(input_):
    return tf.squeeze(input_[0],axis=1)

def bridge(input_,output_caps,name="bridge"):
        #change input of one capsule to fit into another capsule
        caps_raw = tf.reshape(input_[0], [-1, input_[1], input_[2]],
                       name=name+"_caps_raw")
        caps_output_expanded = tf.expand_dims(caps_raw, -1,
                                       name=name+"_caps_output_expanded")

        caps_output_tile = tf.expand_dims(caps_output_expanded, 2,
                                       name=name+"_caps_output_tile")

        caps_output_tiled = tf.tile(caps_output_tile, [1, 1, output_caps, 1, 1],
                                     name=name+"_caps_bridge")
        return caps_output_tiled,input_[1],input_[2]


def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)

def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector

def margin_loss(caps,num_caps,y,m_plus = 0.9,m_minus = 0.1,lambda_ = 0.5):
    T = tf.one_hot(y, depth=num_caps, name="T")
 
    
    caps_output_norm = safe_norm(caps, axis=-2, keep_dims=True,
                                  name="caps_output_norm")

    present_error_raw = tf.square(tf.maximum(0., m_plus - caps_output_norm),
                                  name="present_error_raw")
    present_error = tf.reshape(present_error_raw, shape=(-1, num_caps),
                               name="present_error")
    absent_error_raw = tf.square(tf.maximum(0., caps_output_norm - m_minus),
                                 name="absent_error_raw")
    absent_error = tf.reshape(absent_error_raw, shape=(-1, num_caps),
                              name="absent_error")
    print(absent_error)

    L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
               name="L")

    margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
    return margin_loss

def log_loss(output,num_caps,y):
    T = tf.one_hot(y, depth=num_caps, name="T")
    
    caps_output_norm = tf.squeeze(safe_norm(output, axis=-2, keep_dims=True,
                              name="caps_output_norm"),axis=[1,3,4])

    
    return tf.losses.log_loss(caps_output_norm,T)
