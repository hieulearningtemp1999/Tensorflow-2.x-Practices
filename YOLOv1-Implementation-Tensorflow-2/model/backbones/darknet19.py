import tensorflow as tf
from tensorflow.keras import Model,layers
from .components import conv_bn_lrelu

class Darknet19(Model):
    def __init__(self):
        super(Darknet19,self).__init__()
        self.conv_1 = conv_bn_lrelu(64,3,1)
        self.pool_1 = layers.MaxPool2D()

        self.conv_2 = conv_bn_lrelu(64,3,1)
        self.pool_2 = layers.MaxPool2D()

        self.conv_3 = conv_bn_lrelu(128,3,1)
        self.conv_4 = conv_bn_lrelu(64,1,1)
        self.conv_5 = conv_bn_lrelu(128,3,1)
        self.pool_3 = layers.MaxPool2D()

        self.conv_6 = conv_bn_lrelu(256,3,1)
        self.conv_7 = conv_bn_lrelu(128,1,1)
        self.conv_8 = conv_bn_lrelu(256,3,1)
        self.pool_4 = layers.MaxPool2D()

        self.conv_9 = conv_bn_lrelu(512,3,1)
        self.conv_10 = conv_bn_lrelu(256,1,1)
        self.conv_11 = conv_bn_lrelu(512,3,1)
        self.conv_12 = conv_bn_lrelu(256,1,1)
        self.conv_13 = conv_bn_lrelu(512,3,1)
        self.pool_5 = layers.MaxPool2D()

        self.conv_14 = conv_bn_lrelu(1024, 3, 1)
        self.conv_15 = conv_bn_lrelu(512, 1, 1)
        self.conv_16 = conv_bn_lrelu(1024, 3, 1)
        self.conv_17 = conv_bn_lrelu(512, 1, 1)
        self.conv_18 = conv_bn_lrelu(1024, 3, 1)

    def call(self,x):
        out = self.conv_1(x)
        out = self.pool_1(out)

        out = self.conv_2(out)
        out = self.pool_2(out)

        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.conv_5(out)
        out = self.pool_3(out)

        out = self.conv_6(out)
        out = self.conv_7(out)
        out = self.conv_8(out)
        out = self.pool_4(out)

        out = self.conv_9(out)
        out = self.conv_10(out)
        out = self.conv_11(out)
        out = self.conv_12(out)
        out = self.conv_13(out)
        out = self.pool_5(out)

        out = self.conv_14(out)
        out = self.conv_15(out)
        out = self.conv_16(out)
        out = self.conv_17(out)
        out = self.conv_18(out)

        return out

if __name__ == '__main__':
    model = Darknet19()
    x = x = tf.zeros((1,224,224,3))
    y = model(x)
    print(y.shape)
    backbone = tf.keras.applications.vgg16.VGG16(
        include_top=False,
        weights = None,

    )
    print(backbone(x).shape)
    backbone.summary()
