import tensorflow as tf
from loss import cross_entropy

def test(test_dataset,model):
    losses = []
    correct = 0
    totals = 0
    for data,target in test_dataset:
        pred = model(data,is_training=False)
        pred_label = tf.argmax(pred,axis=1)
        # print(((tf.cast(pred_label,tf.int64))==(tf.cast(target,tf.int64))))
        check_vector = ((tf.cast(pred_label,tf.int64))==(tf.cast(target,tf.int64)))
        correct += tf.reduce_sum(tf.cast(check_vector,tf.float32))
        totals += len(data)
        loss = cross_entropy(pred,target)
        losses.append(loss)
    
    return sum(losses)/len(losses),correct/totals * 100