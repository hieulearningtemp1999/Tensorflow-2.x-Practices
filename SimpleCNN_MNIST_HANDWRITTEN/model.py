import numpy as np
import tensorflow as tf
from tensorflow.keras import Model,layers

class SimpleCNN(Model):
    def __init__(self,num_classes=10):
        super(SimpleCNN,self).__init__()
        self.conv_1 = layers.Conv2D(filters = 32,kernel_size=5,strides=(1,1),padding='valid',use_bias=False)
        self.bn_1 = layers.BatchNormalization()
        self.relu_1 = layers.ReLU()
        self.pool_1 = layers.MaxPool2D(2,strides=2)

        self.conv_2 = layers.Conv2D(filters=64,kernel_size=3,strides=(1,1),padding='valid',use_bias=False)
        self.bn_2 = layers.BatchNormalization()
        self.relu_2 = layers.ReLU()
        self.pool_2 = layers.MaxPool2D(2,strides=2)

        self.flatten = layers.Flatten()

        self.fc1 = layers.Dense(1024)
        self.dropout = layers.Dropout(rate=0.4)

        self.out = layers.Dense(num_classes)
    
    def call(self,x,is_training = False):
        x = tf.reshape(x,[-1,28,28,1])
        x = self.conv_1(x) # (batchsize,24,24,32)
        x = self.bn_1(x,training=is_training) 
        x = self.relu_1(x)
        x = self.pool_1(x) #(batchsize,12,12,32)
        x = self.conv_2(x) #(batchsize,10,10,64)
        x = self.bn_2(x,training=is_training)
        x = self.relu_2(x)
        x = self.pool_2(x) #(batchsize,5,5,64)
        x = self.flatten(x)
        x = self.fc1(x) #(batchsize,1024)
        x = self.dropout(x,training=is_training) #if process is validation, self.dropout is not activated
        x = self.out(x) #(batchsize,num_classes)
        if is_training:
            x = tf.nn.softmax(x)
        return x

# if __name__ == '__main__':
#     net = SimpleCNN()
#     x = np.random.rand(64,28,28)
#     print(net(x).shape)