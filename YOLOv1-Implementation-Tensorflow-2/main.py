import argparse
import json
import os
import tensorflow as tf
from datasets import geo_shape
from model import yolov1
from model.backbones import darknet19,vgg
from sklearn.model_selection import train_test_split
import train
import test

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_image_dir",type=str,help="Image Directory")
    parser.add_argument("-j","--json_path",type=str,help="Path to json annotation")
    parser.add_argument("-b","--batchsize",type=int,default=64,help="batchsize")
    parser.add_argument("-e","--epochs",type=int,default=100,help="epochs")
    parser.add_argument("-lr","--learning_rate",type=float,default=0.001,help="initial learning rate")
    return parser.parse_args()

if __name__ == '__main__':
    args = arguments()
    img_root = args.input_image_dir
    json_path = args.json_path
    batchsize = args.batchsize
    epochs = args.epochs
    lr = args.learning_rate
    device = tf.test.gpu_device_name()
    cell_size = 7
    num_classes = 3
    img_size = 224
    box_per_cell = 2

    with tf.device(device):
        #Get dataset
        train_dataset,test_dataset = geo_shape.get_dataset(json_path,img_root,test_size=0.2,img_size=img_size,cell_size=cell_size,num_classes=num_classes)

        #Get loader
        train_loader = geo_shape.get_loader(train_dataset,batchsize)
        test_loader = geo_shape.get_loader(test_dataset,batchsize)

        # #Get backbone
        # backbone = darknet19.Darknet19()
        backbone = vgg.VGG16()

        # #Get model
        model = yolov1.YOLOv1(backbone,num_classes=num_classes)

        # #Get optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

        best_test_iou = 0
        for epoch in range(epochs):
            optimizer.lr = lr*0.5**((epoch+1)//20)
            print("Epoch : {}/{}".format(epoch+1,epochs))
            print("Learning rate : ",optimizer._decayed_lr('float32').numpy())
            train_loss,train_map,train_iou = train.run(train_loader,model,optimizer,S=cell_size,B=box_per_cell,C=num_classes)
            test_loss,test_map,test_iou = test.run(test_loader,model,S=cell_size,B=box_per_cell,C=num_classes)
            print("Train loss : {}, Test loss : {}".format(train_loss,test_loss))
            print("Train mAP : {}, Test mAP :{}".format(train_map,test_map))
            print("Train IOU : {}, Test IOU : {}".format(train_iou,test_iou))
            if best_test_iou < test_iou:
                best_test_iou = test_iou
                model.save_weights("./output_model_folder/best_iou/")
            
            print("Best Test iou : ",best_test_iou)