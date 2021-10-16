import tensorflow as tf
from weights import Weights
import numpy as np


@tf.function
def conv2d_relu(input,weight,stride,padding = 'VALID'):
    out = tf.nn.conv2d(input,weight,strides=[1,stride,stride,1],padding=padding)
    out = tf.nn.relu(out)
    return out

@tf.function
def model(x,trainable_params):
    x = tf.cast( x , dtype=tf.float32 )

    #conv1
    x = conv2d_relu(x,trainable_params[0],stride=1,padding='VALID')
    x = tf.nn.max_pool2d(x,ksize=[1,2,2,1],padding='VALID',strides=[1,2,2,1])

    #conv2
    x = conv2d_relu(x,trainable_params[1],stride=1,padding='VALID')
    x = tf.nn.max_pool2d(x,ksize=[1,2,2,1],padding='VALID',strides=[1,2,2,1])

    #flatten
    x = tf.reshape(x,shape=(x.shape[0],-1))

    #classifier
    x = tf.matmul(x,trainable_params[2])
    x = tf.nn.relu(x)
    x = tf.nn.dropout(x,rate = 0.3)
    x = tf.matmul(x,trainable_params[3])
    x = tf.nn.softmax(x)
    return x

# if __name__ == '__main__':
#     weight_configs = [
#         [5,5,3,32],
#         [3,3,32,64],
#         [1600,1024],
#         [1024,10]
#     ]
#     init_method = tf.initializers.HeNormal()
#     w = Weights(weight_configs,init_method)
#     x = np.random.randn(32,28,28,3)
#     print(model(x,w.get_trainable_params()).shape)