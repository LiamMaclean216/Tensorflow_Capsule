
# coding: utf-8

# In[1]:


from __future__ import division, print_function, unicode_literals
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import json
tf.reset_default_graph()

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/")


def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector
    


# In[2]:


input_dim = [28,28]

X = tf.placeholder(shape=[None, input_dim[0],input_dim[1], 1], dtype=tf.float32, name="X")


# In[3]:


def capsule(input_,input_caps,input_n_dims,ouput_n_caps,ouput_n_dims,name):
    init_sigma = 0.01
 
    W_init = tf.random_normal(
    shape=(1, input_caps, ouput_n_caps, ouput_n_dims, input_n_dims),
    stddev=init_sigma, dtype=tf.float32, name=(str(name) + "_W_init"))
    
    W = tf.Variable(W_init, name="W")

    batch_size = tf.shape(X)[0]
    W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name=name + "_W_tiled")

    print(W_tiled)
    print(input_)

    caps2_predicted = tf.matmul(W_tiled, input_,
                                name=name + "_predicted")
    caps2_predicted

    raw_weights = tf.zeros([batch_size, input_caps, ouput_n_caps, 1, 1],
                           dtype=np.float32, name=name + "_raw_weights")

    routing_weights = tf.nn.softmax(raw_weights, dim=2, name=name + "_routing_weights")

    weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                       name=name + "_weighted_predictions")
    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                                 name=name + "_weighted_sum")

    caps2_output_round_1 = squash(weighted_sum, axis=-2,
                                  name=name + "_caps2_output_round_1")

    caps2_output_round_1


    caps2_output_round_1_tiled = tf.tile(
        caps2_output_round_1, [1, input_caps, 1, 1, 1],
        name=name + "_caps2_output_round_1_tiled")

    agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                          transpose_a=True, name=name + "_agreement")

    raw_weights_round_2 = tf.add(raw_weights, agreement,
                                 name=name + "_raw_weights_round_2")


    routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                            dim=2,
                                            name=name + "_routing_weights_round_2")
    weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                               caps2_predicted,
                                               name=name + "_weighted_predictions_round_2")
    weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                         axis=1, keep_dims=True,
                                         name=name + "_weighted_sum_round_2")
    caps2_output_round_2 = squash(weighted_sum_round_2,
                                  axis=-2,
                                  name=name+"_output_round_2")

    caps2_output = caps2_output_round_2
    return caps2_output



# In[4]:


caps1_n_maps = 8
caps1_n_dims = 8

num_classes =10


caps2_n_dims = 8
caps2_n_caps = 32

caps3_n_caps = num_classes
caps3_n_dims = 8


# In[5]:


conv1 = tf.layers.conv2d(X, name="conv1", filters = 32, 
                         kernel_size = 9,strides = 1,padding = "valid", activation = tf.nn.relu)
conv2 = tf.layers.conv2d(conv1, name="conv2", filters = caps1_n_maps * caps1_n_dims,
                         kernel_size = 9,strides = 2, padding = "valid",activation = tf.nn.relu)

caps1_n_caps = caps1_n_maps * conv2.get_shape().as_list()[1]  * conv2.get_shape().as_list()[2]  # 1152 primary capsules


# In[6]:


caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims],
                       name="caps1_raw")
print(caps1_raw)
caps1_output = squash(caps1_raw, name="caps1_output")


caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                       name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                   name="caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                             name="caps1_output_tiled")
#1152, 10, 16, 1
print(caps1_output_tiled)

print(" ")



# In[7]:


print("caps2")
caps2_output = capsule(caps1_output_tiled,caps1_n_caps,caps1_n_dims,caps2_n_caps,caps2_n_dims,name = "caps2")
print(" ")
print(caps2_output)

caps2_raw = tf.reshape(caps2_output, [-1, caps2_n_caps, caps2_n_dims],
                       name="caps2_raw")
caps2_output_expanded = tf.expand_dims(caps2_raw, -1,
                                       name="caps2_output_expanded")

caps2_output_tile = tf.expand_dims(caps2_output_expanded, 2,
                                   name="caps2_output_tile")

caps2_output_tiled = tf.tile(caps2_output_tile, [1, 1, caps3_n_caps, 1, 1],
                             name="caps2_output_tiled")
print(caps2_output_tiled)

print("")
print("caps3")

caps3_output = capsule(caps2_output_tiled,caps2_n_caps,caps2_n_dims,caps3_n_caps,caps3_n_dims,name = "caps3")

output = caps3_output
print("")

print(output)


# In[8]:


def condition(input, counter):
    return tf.less(counter, 100)

def loop_body(input, counter):
    output = tf.add(input, tf.square(counter))
    return output, tf.add(counter, 1)

with tf.name_scope("compute_sum_of_squares"):
    counter = tf.constant(1)
    sum_of_squares = tf.constant(0)

    result = tf.while_loop(condition, loop_body, [sum_of_squares, counter])#,swap_memory=True)
    

with tf.Session() as sess:
    print(sess.run(result))


# In[9]:


def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)
    
y_proba = safe_norm(output, axis=-2, name="y_proba")
y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")

y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")


# In[10]:


m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5


# In[11]:


def loss(caps,num_caps):
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
    L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
               name="L")

    margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
    return margin_loss
margin_loss = tf.add(loss(output,num_classes) , loss(caps2_output,caps2_n_caps))
#margin_loss =loss(caps2_output,caps2_n_caps)


# In[12]:


mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                               name="mask_with_labels")
reconstruction_targets = tf.cond(mask_with_labels, # condition
                                 lambda: y,        # if True
                                 lambda: y_pred,   # if False
                                 name="reconstruction_targets")

reconstruction_mask = tf.one_hot(reconstruction_targets,
                                 depth=num_classes,
                                 name="reconstructone_hotone_hotion_mask")
reconstruction_mask_reshaped = tf.reshape(
    reconstruction_mask, [-1, 1, num_classes, 1, 1],
    name="reconstruction_mask_reshaped")
print(reconstruction_mask_reshaped)

caps_output_masked = tf.multiply(
    output, reconstruction_mask_reshaped,
    name="caps_output_masked")


decoder_input = tf.reshape(caps_output_masked,
                           [-1, caps3_n_caps * caps3_n_dims],
                           name="decoder_input")


# In[13]:


n_hidden1 = 512
n_hidden2 = 1024
n_output = input_dim[0] * input_dim[1]

with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                              activation=tf.nn.relu,
                              name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2,
                              activation=tf.nn.relu,
                              name="hidden2")
    decoder_output = tf.layers.dense(hidden2, n_output,
                                     activation=tf.nn.sigmoid,
                                     name="decoder_output")


# In[14]:


X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_output,
                               name="squared_difference")
reconstruction_loss = tf.reduce_sum(squared_difference,
                                    name="reconstruction_loss")

alpha = 0.0005

loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")

correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")


init = tf.global_variables_initializer()
saver = tf.train.Saver()


# In[16]:


print("Starting training MNIST")
n_epochs = 10
batch_size = 50
restore_checkpoint = True

n_iterations_per_epoch = mnist.train.num_examples // batch_size

n_iterations_validation = 100#mnist.train.num_examples // batch_size


best_loss_val = np.infty
checkpoint_path = "./my_capsule_network"



with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()
    predicted = ""
    for epoch in range(n_epochs):

        for iteration in range(1, n_iterations_per_epoch + 1):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
           
            _, loss_train,pred,real = sess.run(
                [training_op, loss,y_pred,y],
                feed_dict={X: X_batch.reshape([-1, input_dim[0], input_dim[1], 1]),
                           y: y_batch,
                           mask_with_labels: True})
            if(iteration % 50 == 0):
                predicted = "  predicted : {}  real : {}".format(pred[0:5],real[0:5])
                
            print(("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}"+predicted).format(
                      iteration, n_iterations_per_epoch,
                      iteration * 100 / n_iterations_per_epoch,
                      loss_train),
                  end="")

        loss_vals = []
        acc_vals = []
        for iteration in range(1, n_iterations_validation + 1):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
               
            loss_val, acc_val = sess.run(
                    [loss, accuracy],
                    feed_dict={X: X_batch.reshape([-1, input_dim[0], input_dim[1], 1]),
                               y: y_batch})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                      iteration, n_iterations_validation,
                      iteration * 100 / n_iterations_validation),
                  end=" " * 10)
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
            epoch + 1, acc_val * 100, loss_val,
            " (improved)" if loss_val < best_loss_val else ""))

        # And save the model if it improved:
        if loss_val < best_loss_val:
            save_path = saver.save(sess, checkpoint_path)
            best_loss_val = loss_val


# In[ ]:


n_samples = 10

sample_images = mnist.test.images[:n_samples].reshape([-1, input_dim[0], input_dim[1], 1])

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    caps2_output_value, decoder_output_value, y_pred_value = sess.run(
            [caps2_output, decoder_output, y_pred],
            feed_dict={X: sample_images,
                       y: np.array([], dtype=np.int64)})



# In[ ]:



sample_images = sample_images.reshape(-1, input_dim[0], input_dim[1])
reconstructions = decoder_output_value.reshape([-1, input_dim[0], input_dim[1]])

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.imshow(sample_images[index], cmap="binary")
    plt.title("Label:" + str(mnist.test.labels[index]))
    plt.axis("off")

plt.show()

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.title("Predicted:" + str(y_pred_value[index]))
    plt.imshow(reconstructions[index], cmap="binary")
    plt.axis("off")
    
plt.show()

sample_images = sample_images.reshape(-1, input_dim[0], input_dim[1])
reconstructions = decoder_output_value.reshape([-1, input_dim[0], input_dim[1]])

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.imshow(sample_images[index], cmap="binary")
    plt.title("Label:" + str(mnist.test.labels[index]))
    plt.axis("off")

plt.show()

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.title("Predicted:" + str(y_pred_value[index]))
    plt.imshow(reconstructions[index], cmap="binary")
    plt.axis("off")
    
plt.show()

