import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from model import SimpleCNN
import train
import test
import train_with_tf_function
import test_with_tf_function

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b","--batch_size",type=int,default=64,help="Batch size")
    parser.add_argument("-lr","--learning_rate",type=float,default=0.001,help="Learning Rate")
    parser.add_argument("-e","--epochs",type=int,default=10,help="epochs")
    parser.add_argument("-tf","--tf_function",type=bool,default=False,help="use tf function to speed up , True if needed,False if not needed")
    return parser.parse_args()



if __name__ == '__main__':
    args = arguments()
    device = tf.test.gpu_device_name()
    print("DEVICE : ",device)
    with tf.device(device):
        #Get dataset
        (x_train,y_train),(x_test,y_test) = mnist.load_data()
        x_train = np.array(x_train,np.float32)
        x_test = np.array(x_test,np.float32)

        #Normalize
        x_train = x_train/255.0
        x_test = x_test/255.0

        #dataset with batchsize
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
        train_dataset = train_dataset.batch(args.batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
        test_dataset = test_dataset.batch(args.batch_size)

        #Get model
        model = SimpleCNN(num_classes=10)

        #Get optimizer
        optimizer = tf.optimizers.Adam(args.learning_rate)
        run_time_epoch = []
        print("TRAINING...")
        print(args.tf_function)
        for epoch in range(args.epochs):
            start_time = time.time()
            if args.tf_function:
                train_loss = train_with_tf_function.train(train_dataset,model,optimizer)
                test_loss,test_acc = test_with_tf_function.test(test_dataset,model)
            else:
                train_loss = train.train(train_dataset,model,optimizer)
                test_loss,test_acc = test.test(test_dataset,model)

            end_time = time.time()
            print("Epoch : {}/{} , test_loss : {}, test_acc : {}".format(epoch+1,args.epochs,test_loss,test_acc))
            print("Run time : {} seconds \n".format(end_time-start_time))
            run_time_epoch.append(end_time-start_time)
        
        run_time_epoch = np.array(run_time_epoch)
        
        if args.tf_function:
            np.save("time_with_tf_function.npy",run_time_epoch)
        else:
            np.save("time_without_tf_function.npy",run_time_epoch)
        