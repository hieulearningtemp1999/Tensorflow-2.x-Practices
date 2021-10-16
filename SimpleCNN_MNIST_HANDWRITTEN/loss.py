import tensorflow as tf

@tf.function
def cross_entropy(pred,target):
    target = tf.one_hot(target,depth=10)
    pred = tf.clip_by_value(pred,1e-9,1.)
    return tf.reduce_mean(-tf.reduce_sum(target*tf.math.log(pred)))