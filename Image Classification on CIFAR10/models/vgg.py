from re import L
from numpy.lib.arraysetops import isin
import tensorflow as tf
from tensorflow.keras import Model,layers
from tensorflow.python.keras.layers.pooling import MaxPool2D
import numpy as np
import keras
from tensorflow.keras import regularizers

vgg_config = {
    'vgg11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'vgg19':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M'],
}

class ConvBlock(layers.Layer):
    def __init__(self,f,k=3,s=1):
        """[summary]

        Args:
            f ([int]): [filters]
            k (int, optional): [kernel_size]. Defaults to 3.
            s (int, optional): [stride]. Defaults to 1.
        """
        super(ConvBlock,self).__init__()
        self.conv = layers.Conv2D(filters = f,kernel_size=k,strides=(s,s),padding='same',kernel_regularizer=regularizers.l2(0.0005))
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
    
    def call(self,x,is_training = True):
        x = self.conv(x)
        x = self.bn(x,training=is_training)
        x = self.relu(x)
        return x

class FeatureLayers(layers.Layer):
    def __init__(self,vgg_cfg):
        """[summary]

        Args:
            vgg_cfg ([vgg_config element]): [get from vgg_config]
        """
        super(FeatureLayers,self).__init__()
        self.layers = []
        for cfg in vgg_cfg:
            if cfg == 'M':
                self.layers.append(layers.MaxPool2D(2,strides=2))
            
            else:
                self.layers.append(ConvBlock(f=int(cfg),k=3,s=1))
    
    def call(self,x,is_training):
        for layer in self.layers:
            if isinstance(layer,ConvBlock):
                x = layer(x,is_training = is_training)
            
            else:
                x = layer(x)
        
        return x
                

class Classifier(layers.Layer):
    def __init__(self,num_classes=10):
        super(Classifier,self).__init__()
        self.fc_1 = layers.Dense(512,kernel_regularizer=regularizers.l2(0.0005))
        self.relu_1 = layers.ReLU()
        self.dropout_1 = layers.Dropout(0.2)
        self.fc_2 = layers.Dense(4096,kernel_regularizer=regularizers.l2(0.0005))
        self.relu_2 = layers.ReLU()
        self.dropout_2 = layers.Dropout(0.2)
        self.fc_3 = layers.Dense(num_classes)
        self.softmax = layers.Activation('softmax')
    
    def call(self,x,is_training = True):
        x = self.fc_1(x)
        x = self.relu_1(x)
        x = self.dropout_1(x,training = is_training)
        # x = self.fc_2(x)
        # x = self.relu_2(x)
        # x = self.dropout_2(x,training=is_training)
        x = self.fc_3(x)
        x = tf.nn.softmax(x)
        return x

class VGG(Model):
    def __init__(self,vgg_cfg,num_classes=10):
        """[summary]

        Args:
            vgg_cfg ([vgg_config element]): [get from vgg_config]
            num_classes (int, optional): [num labels to classify]. Defaults to 10.
        """
        super(VGG,self).__init__()
        self.feature_layers = FeatureLayers(vgg_cfg)
        self.flatten = layers.Flatten()
        self.classifier = Classifier(num_classes)
    
    def call(self,x,is_training=True):
        x = self.feature_layers(x,is_training = is_training)
        x = self.flatten(x)
        x = self.classifier(x,is_training = is_training)
        return x
# class ConvBNRelu(tf.keras.Model):
#     def __init__(self, filters, kernel_size=3, strides=1, padding='SAME', weight_decay=0.0005, rate=0.4, drop=True):
#         super(ConvBNRelu, self).__init__()
#         self.drop = drop
#         self.conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
#                                         padding=padding, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
#         self.batchnorm = tf.keras.layers.BatchNormalization()
#         self.dropOut = keras.layers.Dropout(rate=rate)

#     def call(self, inputs, training=False):
#         layer = self.conv(inputs)
#         layer = tf.nn.relu(layer)
#         layer = self.batchnorm(layer)
#         if self.drop:
#             layer = self.dropOut(layer)

#         return layer
# class VGG16Model(tf.keras.Model):
#     def __init__(self):
#         super(VGG16Model, self).__init__()
#         self.conv1 = ConvBNRelu(filters=64, kernel_size=[3, 3], rate=0.3)
#         self.conv2 = ConvBNRelu(filters=64, kernel_size=[3, 3], drop=False)
#         self.maxPooling1 = keras.layers.MaxPooling2D(pool_size=(2, 2))
#         self.conv3 = ConvBNRelu(filters=128, kernel_size=[3, 3])
#         self.conv4 = ConvBNRelu(filters=128, kernel_size=[3, 3], drop=False)
#         self.maxPooling2 = keras.layers.MaxPooling2D(pool_size=(2, 2))
#         self.conv5 = ConvBNRelu(filters=256, kernel_size=[3, 3])
#         self.conv6 = ConvBNRelu(filters=256, kernel_size=[3, 3])
#         self.conv7 = ConvBNRelu(filters=256, kernel_size=[3, 3], drop=False)
#         self.maxPooling3 = keras.layers.MaxPooling2D(pool_size=(2, 2))
#         self.conv11 = ConvBNRelu(filters=512, kernel_size=[3, 3])
#         self.conv12 = ConvBNRelu(filters=512, kernel_size=[3, 3])
#         self.conv13 = ConvBNRelu(filters=512, kernel_size=[3, 3], drop=False)
#         self.maxPooling5 = keras.layers.MaxPooling2D(pool_size=(2, 2))
#         self.conv14 = ConvBNRelu(filters=512, kernel_size=[3, 3])
#         self.conv15 = ConvBNRelu(filters=512, kernel_size=[3, 3])
#         self.conv16 = ConvBNRelu(filters=512, kernel_size=[3, 3], drop=False)
#         self.maxPooling6 = keras.layers.MaxPooling2D(pool_size=(2, 2))
#         self.flat = keras.layers.Flatten()
#         self.dropOut = keras.layers.Dropout(rate=0.5)
#         self.dense1 = keras.layers.Dense(units=512,
#                                          activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))
#         self.batchnorm = tf.keras.layers.BatchNormalization()
#         self.dense2 = keras.layers.Dense(units=10)
#         self.softmax = keras.layers.Activation('softmax')

#     def call(self, inputs, training=False):
#         net = self.conv1(inputs)
#         net = self.conv2(net)
#         net = self.maxPooling1(net)
#         net = self.conv3(net)
#         net = self.conv4(net)
#         net = self.maxPooling2(net)
#         net = self.conv5(net)
#         net = self.conv6(net)
#         net = self.conv7(net)
#         net = self.maxPooling3(net)
#         net = self.conv11(net)
#         net = self.conv12(net)
#         net = self.conv13(net)
#         net = self.maxPooling5(net)
#         net = self.conv14(net)
#         net = self.conv15(net)
#         net = self.conv16(net)
#         net = self.maxPooling6(net)
#         net = self.dropOut(net)
#         net = self.flat(net)
#         net = self.dense1(net)
#         net = self.batchnorm(net)
#         net = self.dropOut(net)
#         net = self.dense2(net)
#         net = self.softmax(net)
#         return net
def VGG11():
    return VGG(vgg_config['vgg11'])

def VGG13():
    return VGG(vgg_config['vgg13'])

def VGG16():
    return VGG(vgg_config['vgg16'])

def VGG19():
    return VGG(vgg_config['vgg19'])
                          
# if __name__ == '__main__':
#     x = np.random.randn(64,64,64,3)
#     vgg = VGG(vgg_config['vgg19'])
#     print(vgg(x,is_training = True).shape)