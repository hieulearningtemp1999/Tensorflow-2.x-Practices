import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, optimizers
from get_model import get_model
import train, test


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, help="Model Type")
    parser.add_argument("-b", "--batchsize", type=int, default=64, help="batch size")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Epochs")
    parser.add_argument("-o", "--output_model_folder", type=str, default="./output_model_folder",
                        help="Folder to save .h5 models")
    parser.add_argument("-oh", "--output_training_hist_folder", type=str, default="./training_hists",
                        help="Folder to save training loss and accuracy histories (train loss,train acc ,test loss ,test acc) of each model")
    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    model_type = args.model_type
    batchsize = args.batchsize
    learning_rate = args.learning_rate
    epochs = args.epochs
    output_model_folder = args.output_model_folder
    output_training_hist_folder = args.output_training_hist_folder

    device = tf.test.gpu_device_name()
    with tf.device(device):
        if device:
            print("Device : ", device)
        else:
            print("Device : CPU ")
        # Get dataset
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        nums_train = x_train.shape[0]
        nums_test = x_test.shape[0]
        print("Training samples : ", nums_train)
        print("Testing samples :", nums_test)

        # one-hot coding
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)

        # Get mean and std of training dataset for normalization
        mean = np.mean(x_train, axis=(0, 1, 2, 3))
        std = np.std(x_train, axis=(0, 1, 2, 3))
        x_train = (x_train - mean) / (std + 1e-7)
        x_test = (x_test - mean) / (std + 1e-7)

        # Get data augmentation
        train_aug = tf.keras.Sequential([
            layers.experimental.preprocessing.Resizing(32, 32),
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.04),
            layers.experimental.preprocessing.RandomTranslation(0.2, 0.2),
            layers.experimental.preprocessing.RandomZoom(0.2, 0.2)

        ])

        test_aug = layers.experimental.preprocessing.Resizing(32, 32)

        # Get train and test dataset with batchsize and data augmentation
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = (
            train_dataset
                .batch(batchsize)
                .map(lambda x, y: (train_aug(x), y), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE)
        )

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = (
            test_dataset
                .batch(batchsize)
                .map(lambda x, y: (test_aug(x), y), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE)
        )

        # Get model
        model = get_model(model_type)

        # Get optimizer
        num_steps = nums_train / batchsize
        learning_rate_fn = optimizers.schedules.PiecewiseConstantDecay(
            [(epochs / 3) * num_steps, (epochs * 2. / 3) * num_steps],
            [learning_rate, learning_rate * 0.1, learning_rate * 0.01]
        )
        optimizer = optimizers.SGD(learning_rate=learning_rate_fn, decay=1e-6, momentum=0.9, nesterov=True)

        # To save hist
        train_loss_hist = []
        train_acc_hist = []
        test_loss_hist = []
        test_acc_hist = []
        best_acc = 0

        for epoch in range(epochs):
            # optimizer.lr = learning_rate*(0.75**(epoch//30))
            print("Epoch : {}/{}, Learning Rate : {}".format(epoch + 1, epochs, optimizer._decayed_lr('float32').numpy()))

            # train
            train_loss, train_correct = train.run(train_dataset, model, optimizer)
            train_acc = train_correct / nums_train * 100.
            train_loss_hist.append(train_loss)
            train_acc_hist.append(train_acc)

            # test
            test_loss, test_correct = test.run(test_dataset, model)
            test_acc = test_correct / nums_test * 100.
            test_loss_hist.append(test_loss)
            test_acc_hist.append(test_acc)

            if best_acc < test_acc:
                best_acc = test_acc
                model.save_weights(os.path.join(output_model_folder, model_type) + "/")

            print("Train Loss : {}, Train Acc : {}".format(train_loss_hist[-1], train_acc_hist[-1]))
            print("Test Loss :{}, Test Acc: {}".format(test_loss_hist[-1], test_acc_hist[-1]))
            print("Best Acc : {} \n".format(best_acc))

        # Save hist
        np.save(os.path.join(output_training_hist_folder, "train_losses/" + model_type + ".npy"),
                np.array(train_loss_hist, dtype=np.float32))
        np.save(os.path.join(output_training_hist_folder, "train_accuracies/" + model_type + ".npy"),
                np.array(train_acc_hist, dtype=np.float32))
        np.save(os.path.join(output_training_hist_folder, "test_losses/" + model_type + ".npy"),
                np.array(test_loss_hist, dtype=np.float32))
        np.save(os.path.join(output_training_hist_folder, "test_accuracies/" + model_type + ".npy"),
                np.array(test_acc_hist, dtype=np.float32))



