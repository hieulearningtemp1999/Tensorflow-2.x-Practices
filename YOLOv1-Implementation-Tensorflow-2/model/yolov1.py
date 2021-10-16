import tensorflow as tf
from tensorflow.keras import Model,Sequential,layers,regularizers
from .backbones import darknet19
from tensorflow.keras import initializers

class YOLOv1(Model):
    def __init__(self,backbone,num_boxes=2,num_classes=20,grid_size=7):
        super(YOLOv1,self).__init__()
        self.backbone = backbone
        self.B = num_boxes
        self.S = grid_size
        self.C = num_classes
        self.conv_layers = Sequential([
            layers.Conv2D(filters=512,kernel_size=1,strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer=initializers.HeNormal()),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),
            # layers.Conv2D(filters=13,kernel_size=1,strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(0.0005)),
            # layers.BatchNormalization(),
            # layers.LeakyReLU(0.1)
        ])

        self.fc_layers  = Sequential([
            layers.Flatten(),
            layers.Dense(512,kernel_regularizer=regularizers.l2(0.0005)),
            layers.Dropout(0.5),
            layers.Dense(self.S*self.S*(5*self.B+self.C)),
        ])

    def call(self,x):
        x = tf.cast(x,tf.float32)
        out = self.backbone(x)
        out = self.conv_layers(out)
        out = self.fc_layers(out)
        out = tf.reshape(out,[out.shape[0],self.S,self.S,5*self.B+self.C])
        return out

# if __name__ == '__main__':
#     backbone = darknet19.Darknet19()
#     yolov1 = YOLOv1(backbone)
#     x = tf.zeros((1,224,224,3))
#     print(yolov1(x).shape)

