import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers, regularizers

resnet_configs = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3],
    'resnet152': [3, 8, 36, 3]
}


class ResBlock1(layers.Layer):
    def __init__(self, in_c, out_c, s):
        """[summary]

        Args:
            in_c ([int]): [in_channels]
            out_c ([int]): [out_channels]
            s ([int]): [stride]

        ResBlock1 is used for resnet18 and resnet34
        """
        super(ResBlock1, self).__init__()
        self.conv_1 = layers.Conv2D(out_c, kernel_size=3, strides=(1, 1), padding='same',
                                    kernel_regularizer=regularizers.l2(0.0005))
        self.bn_1 = layers.BatchNormalization()
        self.conv_2 = layers.Conv2D(out_c, kernel_size=3, strides=(s, s), padding='same',
                                    kernel_regularizer=regularizers.l2(0.0005))
        self.bn_2 = layers.BatchNormalization()
        self.residual = None
        if s != 1 or in_c != out_c:
            self.residual = Sequential([
                layers.Conv2D(out_c, kernel_size=1, strides=(s, s), padding='same',
                              kernel_regularizer=regularizers.l2(0.0005)),
                layers.BatchNormalization()
            ])
        self.relu = layers.ReLU()

    def call(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        if self.residual == None:
            res = x

        else:
            res = self.residual(x)

        out = out + res
        out = self.relu(out)
        return out


class ResBlock2(layers.Layer):
    def __init__(self, in_c, out_c, s):
        """[summary]

        Args:
            in_c ([int]): [in_channels]
            out_c ([int]): [out_channels]
            s ([int]): [stride]
        """
        super(ResBlock2, self).__init__()
        self.conv_1 = layers.Conv2D(out_c // 4, kernel_size=1, strides=(1, 1), padding='same',
                                    kernel_regularizer=regularizers.l2(0.0005))
        self.bn_1 = layers.BatchNormalization()
        self.conv_2 = layers.Conv2D(out_c // 4, kernel_size=3, strides=(s, s), padding='same',
                                    kernel_regularizer=regularizers.l2(0.0005))
        self.bn_2 = layers.BatchNormalization()
        self.conv_3 = layers.Conv2D(out_c, kernel_size=1, strides=(1, 1), padding='same',
                                    kernel_regularizer=regularizers.l2(0.0005))
        self.bn_3 = layers.BatchNormalization()
        self.residual = None
        if s != 1 or in_c != out_c:
            self.residual = Sequential([
                layers.Conv2D(out_c, kernel_size=1, strides=(s, s), padding='same',
                              kernel_regularizer=regularizers.l2(0.0005)),
                layers.BatchNormalization()
            ])
        self.relu = layers.ReLU()

    def call(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.conv_3(out)
        out = self.bn_3(out)
        if self.residual == None:
            res = x

        else:
            res = self.residual(x)

        out = out + res
        out = self.relu(out)
        return out


class ResNet(Model):
    def __init__(self, block, configs, num_classes=10):
        """[summary]
        block : Type of block : ResBlock1 / ResBlock2
        configs : resnet_config
        """
        super(ResNet, self).__init__()
        self.block = block
        self.configs = configs
        self.in_c = 64
        if block == ResBlock1:
            self.out_c = [64, 128, 256, 512]
        elif block == ResBlock2:
            self.out_c = [256, 512, 1024, 2048]

        self.preconv = layers.Conv2D(self.in_c, kernel_size=3, strides=(1, 1), padding='same',
                                     kernel_regularizer=regularizers.l2(0.0005))
        self.bn = layers.BatchNormalization()
        self.res_layers = self.make_res_layers()
        self.avg_pool = layers.AveragePooling2D(pool_size=(2, 2))
        self.flatten = layers.Flatten()
        # self.fc_1 = layers.Dense(256,kernel_regularizer=regularizers.l2(0.0005))
        # self.dropout = layers.Dropout(0.3)
        self.fc_2 = layers.Dense(num_classes, kernel_regularizer=regularizers.l2(0.0005))
        self.softmax = layers.Activation('softmax')

    def make_res_layers(self):
        res_layers = Sequential()
        for i, repeat_time in enumerate(self.configs):
            res_blocks = Sequential()
            in_c = self.in_c
            for j in range(repeat_time):
                if j != repeat_time - 1:
                    res_blocks.add(self.block(in_c, self.out_c[i], s=1))
                    in_c = self.out_c[i]
                else:
                    res_blocks.add(self.block(in_c, self.out_c[i], s=2))
                    in_c = self.out_c[i]
            res_layers.add(res_blocks)
        return res_layers

    def call(self, x):
        out = self.preconv(x)
        out = self.bn(out)
        out = self.res_layers(out)
        out = self.avg_pool(out)
        out = self.flatten(out)
        # out = self.fc_1(out)
        # out = self.dropout(out)
        out = self.fc_2(out)
        out = self.softmax(out)
        return out


def ResNet18():
    return ResNet(ResBlock1, resnet_configs['resnet18'])


def ResNet34():
    return ResNet(ResBlock1, resnet_configs['resnet34'])


def ResNet50():
    return ResNet(ResBlock2, resnet_configs['resnet50'])


def ResNet101():
    return ResNet(ResBlock2, resnet_configs['resnet101'])


def ResNet152():
    return ResNet(ResBlock2, resnet_configs['resnet152'])

# if __name__ == '__main__':
#     x = np.random.randn(64,32,32,3)
#     res_1_1 = ResBlock2(3,64,2)
#     print(res_1_1(x).shape)
#     resnet = ResNet(ResBlock2,resnet_configs['resnet50'])
#     print(resnet(x).shape)
