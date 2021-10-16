import tensorflow as tf
import json
from tqdm import tqdm
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from .preprocess import preprocess

classes = {
    'circle':0, 'triangle':1,  'rectangle':2
}

# def parse_json(json_path,img_root,img_size=224,cell_size=7,num_classes=3):
#     """
#     Params:
#     json_path : path to json annotation
#     img_root : dir contains images
#     return : X (num_images,image_size,image_size,3), Y (num_images,cell_size,cell_size,num_classes+5)
#     """
#     labels = json.load(open(json_path))
#     num_images = len(labels)
#     X = np.zeros((num_images,img_size,img_size,3),dtype=np.uint8)
#     Y = np.zeros((num_images,cell_size,cell_size,num_classes+5))
#     for idx,label in tqdm(enumerate(labels)):
#         img = cv2.imread(os.path.join(img_root,str(idx)+".png"))
#         X[idx] = img
#         for box in label['boxes']:
#             x1 = box['x1']
#             y1 = box['y1']
#             x2 = box['x2']
#             y2 = box['y2']
#             #onehot
#             one_hot_list = [0]*len(classes)
#             one_hot_list[classes[box['class']]] = 1
#             x_center = (x1+x2)/2.
#             y_center = (y1+y2)/2.
#             w = np.abs(x2-x1)
#             h = np.abs(y2-y1)
#             x_idx,y_idx = int(x_center/img_size*cell_size),int(y_center/img_size*cell_size)
#             Y[idx,x_idx,y_idx] = *one_hot_list,x_center,y_center,w,h,1
        
#     return X,Y

def get_dataset(json_path,img_root,test_size=0.2,img_size=224,cell_size=7,num_classes=3):
    labels = json.load(open(json_path))
    num_images = len(labels)
    img_paths = []
    Y = np.zeros((num_images,cell_size,cell_size,num_classes+5))
    # print(Y.shape)
    for idx,label in tqdm(enumerate(labels)):
        img_path = os.path.join(img_root,str(idx)+".png")
        img_paths.append(img_path)
        for box in label['boxes']:
            x1 = box['x1']
            y1 = box['y1']
            x2 = box['x2']
            y2 = box['y2']
            #one hot 
            one_hot_list = [0]*len(classes)
            one_hot_list[classes[box['class']]] = 1
            x_center = (x1+x2)/2.
            y_center = (y1+y2)/2.
            w = np.abs(x2-x1)
            h = np.abs(y2-y1)
            x_idx,y_idx = int(x_center/img_size*cell_size),int(y_center/img_size*cell_size)
            Y[idx,x_idx,y_idx] = 1,x_center,y_center,w,h,*one_hot_list
    
    #Split data to train and test
    img_train_paths,img_test_paths,Y_train,Y_test = train_test_split(img_paths,Y,test_size=test_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((img_train_paths,Y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((img_test_paths,Y_test))
    return train_dataset,test_dataset
    

def get_loader(dataset,batchsize=64):
    dataloader = (
        dataset
        .map(preprocess,num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batchsize)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataloader

# if __name__ == '__main__':
#     json_path = "D:\\Deep Learning Repos\\YOLOv1-Implementation-Tensorflow-2\\datasets\\labels.json"
#     img_root = "D:\\Deep Learning Repos\\YOLOv1-Implementation-Tensorflow-2\\datasets\\train"
#     # x,y,z,t = get_dataset(json_path,img_root)
#     # print(z.shape)
#     # print(t.shape)