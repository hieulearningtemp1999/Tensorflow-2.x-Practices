import tensorflow as tf
from loss import cross_entropy

@tf.function
def test_step(data,target,model):
    pred = model(data,is_training=False)
    pred_label = tf.argmax(pred,axis=1)
    check_vector = ((tf.cast(pred_label,tf.int64))==(tf.cast(target,tf.int64)))
    correct = tf.reduce_sum(tf.cast(check_vector,tf.float32))
    loss = cross_entropy(pred,target)
    return loss,correct

def test(test_dataset,model):
    losses = []
    correct = 0
    totals = 0
    for data,target in test_dataset:
        totals += len(data)
        loss,correct_batch = test_step(data,target,model)
        losses.append(loss)
        correct += correct_batch
        losses.append(loss)
    
    return sum(losses)/len(losses),correct/totals * 100