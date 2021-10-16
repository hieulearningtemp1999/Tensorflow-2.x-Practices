from tensorflow.keras import layers
from tensorflow.keras import regularizers

class conv_bn_relu(layers.Layer):
    def __init__(self,out_c,k=3,s=1,p='same',use_relu=True,use_bn=True):
        """
        :param out_c: out_channels
        :param k: kernel_size
        :param s: stride
        :param p : padding ('same' or 'valid')
        :param use_relu: if true use relu
        :param use_bn: if true use batchnorm
        """
        super(conv_bn_relu,self).__init__()
        if p != 'same' and p != 'valid':
            raise ValueError("Padding should be 'same' or 'valid'")
        self.conv = layers.Conv2D(filters=out_c,kernel_size=k,strides=(s,s),padding=p,
                                  kernel_regularizer=regularizers.l2(0.005))
        self.relu = None
        self.bn = None

        if use_bn:
            self.bn = layers.BatchNormalization()

        if use_relu:
            self.relu = layers.ReLU()

    def call(self,x):
        out = self.conv(x)
        if self.bn:
            out = self.bn(out)
        if self.relu:
            out = self.relu(out)

        return out


class conv_bn_lrelu(layers.Layer):
    def __init__(self,out_c,k=3,s=1,p='same',use_lrelu=True,alpha=0.1,use_bn=True):
        """
        :param out_c: out_channels
        :param k: kernel_size
        :param s: stride
        :param p : padding ('same' or 'valid')
        :param use_relu: if true use relu
        :param use_bn: if true use batchnorm
        """
        super(conv_bn_lrelu,self).__init__()
        if p != 'same' and p != 'valid':
            raise ValueError("Padding should be 'same' or 'valid'")
        self.conv = layers.Conv2D(filters=out_c,kernel_size=k,strides=(s,s),padding=p,
                                  kernel_regularizer=regularizers.l2(0.005))
        self.l_relu = None
        self.bn = None

        if use_bn:
            self.bn = layers.BatchNormalization()

        if use_lrelu:
            self.l_relu = layers.LeakyReLU(alpha)

    def call(self,x):
        out = self.conv(x)
        if self.bn:
            out = self.bn(out)
        if self.l_relu:
            out = self.l_relu(out)

        return out

# if __name__ == '__main__':
#     import tensorflow as tf
#     import numpy as np
#     x = tf.zeros((1,224,224,3))
#     #x = np.random.randn(1,224,224,3)
#     model = conv_bn_leakyrelu(64,3,1)
#     print(model(x).shape)