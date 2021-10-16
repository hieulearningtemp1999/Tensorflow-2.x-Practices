import tensorflow as tf
from tqdm import tqdm
from loss import loss

@tf.function
def train_step(model,data,target,optimizer):
    with tf.GradientTape() as tape:
        pred = model(data)
        train_loss = loss(pred,target)
    grads = tape.gradient(train_loss,model.trainable_weights)
    optimizer.apply_gradients(zip(grads,model.trainable_weights))
    pred_label = tf.argmax(pred,axis=1)
    target_label = tf.argmax(target,axis=1)
    check_equal = (tf.cast(pred_label,tf.int64)) == (tf.cast(target_label,tf.int64))
    correct = tf.reduce_sum(tf.cast(check_equal,tf.float32))
    return train_loss,correct

def run(train_dataset,model,optimizer):
    print("TRAINING...")
    train_losses = []
    corrects = 0
    loader = tqdm(train_dataset)
    for i,(data,target) in enumerate(loader):
        train_loss,correct = train_step(model,data,target,optimizer)
        corrects += correct
        train_losses.append(train_loss)
        loader.set_postfix(train_loss_batch=train_loss)
    return sum(train_losses)/len(train_losses),corrects
