import os
import argparse
import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
from tensorflow.keras import layers,optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from get_model import get_model
import train
import test
def normalize(x,mean,std):
    x = (x-mean)/(std+1e-7)
    return x
    
def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model_type",type=str,help="Model type")
    parser.add_argument("-b","--batchsize",type=int,default=128,help="Batch size")
    parser.add_argument("-lr","--learning_rate",type=float,default=0.001,help="Learning rate")
    parser.add_argument("-e","--epochs",type=int,default=300,help="Epochs")
    parser.add_argument("-o","--output_folder",type=str,default="./model_output_folder",help="Output folder to save model files")
    return parser.parse_args()
    
if __name__ == '__main__':
    args = arguments()
    model_type = args.model_type
    batch_size = args.batchsize
    lr = args.learning_rate
    epochs = args.epochs
        
    device = tf.test.gpu_device_name()
    print("Device : ",device)
    with tf.device(device):
        #Get dataset
        (x_train,y_train),(x_test,y_test) = cifar10.load_data()
            
        num_train = x_train.shape[0]
        num_test = x_test.shape[0]
        print("Train sample number : ",x_train.shape[0])
        print("Test sample number : ",x_test.shape[0])
            
        #convert label to one-hot coding
        y_train = to_categorical(y_train,num_classes = 10)
        y_test = to_categorical(y_test,num_classes = 10)
    
        #Get mean and std for normalization
        mean = np.mean(x_train,axis=(0,1,2,3))
        std = np.std(x_train,axis=(0,1,2,3))
        x_train = normalize(x_train,mean,std)
        x_test = normalize(x_test,mean,std)
        train_datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        train_dataset = train_datagen.flow(x_train, y_train, batch_size=batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
        test_dataset = test_dataset.batch(batch_size)
        # (std, mean, and principal components if ZCA whitening is applied).
            
        #Data augmentation
        # trainAug = tf.keras.Sequential([
        #     layers.experimental.preprocessing.Resizing(height=32,width=32),
        #     layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        #     layers.experimental.preprocessing.RandomRotation(0.2),
        #     #layers.experimental.preprocessing.Normalization(mean=[0.4914, 0.4822, 0.4465],variance=[0.2023**2, 0.1994**2, 0.2010**2])
        # ])
            
        # testAug = tf.keras.Sequential([
        #     layers.experimental.preprocessing.Resizing(height=32,width=32),
        #     #layers.experimental.preprocessing.Normalization(mean=[0.4914, 0.4822, 0.4465],variance=[0.2023**2, 0.1994**2, 0.2010**2])
        # ])
            
        # #Get train and test dataset with batch
        # train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
        # #train_dataset = train_dataset.batch(batch_size)
        # train_dataset = (
        #     train_dataset
        #     .shuffle(batch_size*100)
        #     .batch(batch_size)
        #     .map(lambda x,y: (trainAug(x),y),num_parallel_calls=tf.data.AUTOTUNE)
        #     # .map(lambda x,y: (normalize(x,mean,std),y),num_parallel_calls=tf.data.AUTOTUNE)
        # )
        # test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
        # # test_dataset = test_dataset.batch(batch_size)
        # test_dataset = (
        #     test_dataset
        #     .batch(batch_size)
        #     .map(lambda x,y: (testAug(x),y),num_parallel_calls=tf.data.AUTOTUNE)
        #     # .map(lambda x,y: (normalize(x,mean,std),y),num_parallel_calls=tf.data.AUTOTUNE)
        # )
            
        #Get model
        model = get_model(model_type)
            
        #Get optimizer and learning rate scheduler
        num_steps = num_train/batch_size
        learning_rate_fn = optimizers.schedules.PiecewiseConstantDecay(
            [(epochs/3)*num_steps,(epochs*2./3)*num_steps],
            [lr,lr*0.1,lr*0.01]
        )
        optimizer = optimizers = optimizers.SGD(learning_rate = learning_rate_fn,decay=1e-6, momentum=0.9, nesterov=True)
    
        #To save hist
        train_loss_hist = []
        train_acc_hist = []
        test_loss_hist = []
        test_acc_hist = []
        best_acc = 0
        steps_per_epoch = (num_train)//batch_size+1
    
        for epoch in range(epochs):
            # optimizer.lr = lr*(0.5**(epoch//20))
            print("Epochs {}/{}, Learning rate : {}".format(epoch+1,epochs,optimizer._decayed_lr('float32').numpy()))
            #train  
            train_loss,train_acc = train.run(train_dataset,model,optimizer,steps_per_epoch)
            train_loss_hist.append(train_loss)
            train_acc_hist.append(train_acc)
            
            #test
            test_loss,test_acc = test.run(test_dataset,model)
            test_loss_hist.append(test_loss)
            test_acc_hist.append(test_acc)
            
            #save best model       
            if best_acc < test_acc/num_test*100.:
                best_acc = test_acc/num_test*100.
                
            print("Train Loss : {},Train Acc : {}, Test Loss : {}, Test Acc : {}, Best Acc : {}".format(train_loss,train_acc/num_train*100.,test_loss,test_acc/num_test*100.,best_acc))
            #     break
            
        
        