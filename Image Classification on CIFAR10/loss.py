import tensorflow as tf

@tf.function
def loss(pred,target):
    return tf.keras.losses.CategoricalCrossentropy()(target,pred)