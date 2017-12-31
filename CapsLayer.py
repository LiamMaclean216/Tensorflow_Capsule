import tensorflow as tf
import numpy as np


class Capsule:
    def __init__(self,num_caps,caps_dims,name):
        self.num_caps = num_caps
        self.caps_dims = caps_dims
        self.name = name
    
    def capsule(self,input_,input_caps,input_n_dims,batch_size,with_routing = True,init_sigma = 0.3): 
        self.init_sigma = init_sigma

        W_init = tf.random_normal(shape=(1, input_caps, self.num_caps, self.caps_dims, input_n_dims ),
            stddev=self.init_sigma, dtype=tf.float32, name=self.name + "W_init")


        #Wij
        W = tf.Variable(W_init, name=self.name+"_W")


        W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name=self.name + "_W_tiled")

        print(W_tiled)
        print(input_)

        
        #U hat
        caps_predicted = tf.matmul(W_tiled, input_,
                                            name=self.name + "_predicted")

        print(caps_predicted)

        #Bij 

        raw_weights_init = tf.ones([1, input_caps, self.num_caps, 1, 1],
                       dtype=np.float32, name="raw_weights")

        raw_weights_var = tf.Variable(raw_weights_init, name=self.name+"_raw_weights_var")

        raw_weights = tf.tile(raw_weights_var, [batch_size, 1, 1, 1, 1], name=self.name + "_raw_weights_tiled")
        #Cij
        routing_weights = tf.nn.softmax(raw_weights, dim=2, name=self.name + "_routing_weights")

        print(routing_weights)

        #Sj
        weighted_predictions = tf.multiply(routing_weights, caps_predicted,
                                               name=self.name + "_weighted_predictions")

        print(weighted_predictions)
        weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                                     name=self.name + "_weighted_sum")
        
        print(weighted_sum)
        #Vj
        caps_output_round_1 = squash(weighted_sum, axis=-2,
                                      name=self.name + "_caps_output_round_1")
        print(caps_output_round_1)

        caps_output_round_1_tiled = tf.tile(
            caps_output_round_1, [1, input_caps, 1, 1, 1],
            name=self.name + "_caps_output_round_1_tiled")
        print(caps_output_round_1_tiled)

        #Aij
        agreement = tf.matmul(caps_predicted, caps_output_round_1_tiled,
                              transpose_a=True, name=self.name + "_agreement")
        print(agreement)
        
        raw_weights_round_2 = tf.add(agreement, raw_weights,
                                     name=self.name + "_raw_weights_round_2")


        routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                                dim=2,
                                                name=self.name + "_routing_weights_round_2")
        weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                                   caps_predicted,
                                                   name=self.name + "_weighted_predictions_round_2")

        
        weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                             axis=1, keep_dims=True,
                                             name=self.name + "_weighted_sum_round_2")
        
        
        print(weighted_predictions_round_2)

        caps_output_round_2 = squash(weighted_sum_round_2,
                                      axis=-2,
                                      name=self.name+"_output_round_2")
        
        
        caps_output = caps_output_round_2
        return caps_output


    #input_ capsule data
    def bridge(self,input_,output_):
        #change input of one capsule to fit into another capsule
        caps_raw = tf.reshape(input_, [-1, self.num_caps, self.caps_dims],
                       name="caps2_raw")
        caps_output_expanded = tf.expand_dims(caps_raw, -1,
                                       name=self.name + "_output_expanded")

        caps_output_tile = tf.expand_dims(caps_output_expanded, 2,
                                       name=self.name + "_output_tile")

        caps_output_tiled = tf.tile(caps_output_tile, [1, 1, output_.num_caps, 1, 1],
                                     name=self.name + "_bridge")
        return caps_output_tiled
    
    
def bridge(input_,num_caps,caps_dims,output_):
        #change input of one capsule to fit into another capsule
        caps_raw = tf.reshape(input_, [-1, num_caps, caps_dims],
                       name="caps_raw")
        caps_output_expanded = tf.expand_dims(caps_raw, -1,
                                       name="caps_output_expanded")

        caps_output_tile = tf.expand_dims(caps_output_expanded, 2,
                                       name="caps_output_tile")

        caps_output_tiled = tf.tile(caps_output_tile, [1, 1, output_.num_caps, 1, 1],
                                     name="caps_bridge")
        return caps_output_tiled


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
