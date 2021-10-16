import cv2
import tensorflow as tf
import numpy as np

def preprocess_func(image_path,label_matrix):
    image_path = image_path.numpy().decode('utf-8')
    img = cv2.imread(image_path)
    img = np.array(img,np.float32)
    img_normalized = img/255.
    return tf.cast(img_normalized,tf.float32),tf.cast(label_matrix,tf.float32)

def preprocess(image_path,label_matrix):
    img_normalized, label_matrix = tf.py_function(preprocess_func,[image_path,label_matrix],[tf.float32,tf.float32])
    return img_normalized,label_matrix
