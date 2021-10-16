from re import L
from numpy.lib.arraysetops import isin
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.python.keras.layers.pooling import MaxPool2D
import numpy as np
import keras
from tensorflow.keras import regularizers

vgg_config = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class ConvBlock(layers.Layer):
    def __init__(self, f, k=3, s=1):
        """[summary]
        Args:
            f ([int]): [filters]
            k (int, optional): [kernel_size]. Defaults to 3.
            s (int, optional): [stride]. Defaults to 1.
        """
        super(ConvBlock, self).__init__()
        self.conv = layers.Conv2D(filters=f, kernel_size=k, strides=(s, s), padding='same',
                                  kernel_regularizer=regularizers.l2(0.0005))
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x, is_training=True):
        x = self.conv(x)
        x = self.bn(x, training=is_training)
        x = self.relu(x)
        return x


class FeatureLayers(layers.Layer):
    def __init__(self, vgg_cfg):
        """[summary]
        Args:
            vgg_cfg ([vgg_config element]): [get from vgg_config]
        """
        super(FeatureLayers, self).__init__()
        self.layers = []
        for cfg in vgg_cfg:
            if cfg == 'M':
                self.layers.append(layers.MaxPool2D(2, strides=2))

            else:
                self.layers.append(ConvBlock(f=int(cfg), k=3, s=1))

    def call(self, x, is_training):
        for layer in self.layers:
            if isinstance(layer, ConvBlock):
                x = layer(x, is_training=is_training)

            else:
                x = layer(x)

        return x

class Classifier(layers.Layer):
    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        self.fc_1 = layers.Dense(4096, kernel_regularizer=regularizers.l2(0.0005))
        self.relu_1 = layers.ReLU()
        self.dropout_1 = layers.Dropout(0.2)
        self.fc_2 = layers.Dense(4096, kernel_regularizer=regularizers.l2(0.0005))
        self.relu_2 = layers.ReLU()
        self.dropout_2 = layers.Dropout(0.2)
        self.fc_3 = layers.Dense(num_classes)
        self.softmax = layers.Activation('softmax')

    def call(self, x, is_training=True):
        x = self.fc_1(x)
        x = self.relu_1(x)
        x = self.dropout_1(x, training=is_training)
        x = self.fc_2(x)
        x = self.relu_2(x)
        x = self.dropout_2(x, training=is_training)
        x = self.fc_3(x)
        x = tf.nn.softmax(x)
        return x

class VGG(Model):
    def __init__(self, vgg_cfg, num_classes=10):
        """[summary]
        Args:
            vgg_cfg ([vgg_config element]): [get from vgg_config]
            num_classes (int, optional): [num labels to classify]. Defaults to 10.
        """
        super(VGG, self).__init__()
        self.feature_layers = FeatureLayers(vgg_cfg)
        self.flatten = layers.Flatten()
        self.classifier = Classifier(num_classes)

    def call(self, x, is_training=True):
        x = self.feature_layers(x, is_training=is_training)
        x = self.flatten(x)
        x = self.classifier(x, is_training=is_training)
        return x

def VGG11():
    return VGG(vgg_config['vgg11'])

def VGG13():
    return VGG(vgg_config['vgg13'])


def VGG16():
    return VGG(vgg_config['vgg16'])

def VGG19():
    return VGG(vgg_config['vgg19'])

# if __name__ == '__main__':
#     x = np.random.randn(64,32,32,3)
#     vgg = VGG11()
#     print(vgg(x).shape)
#     vgg.summary()