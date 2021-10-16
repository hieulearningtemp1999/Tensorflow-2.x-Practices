import tensorflow as tf
from loss import cross_entropy
from tqdm import tqdm

def train(train_dataset,model,optimizer):
    losses = []
    train_loader = tqdm(train_dataset)
    for (data,target) in (train_loader):
        with tf.GradientTape() as tape:
            pred = model(data,is_training = True)
            loss = cross_entropy(pred,target)
            losses.append(loss)

        grads = tape.gradient(loss,model.trainable_weights)

        optimizer.apply_gradients(zip(grads,model.trainable_weights))

        train_loader.set_postfix(loss_batch = loss)
    print("Train loss : ",sum(losses)/len(losses))
    #tf.print("Train loss : ",sum(losses)/len(losses))
    return sum(losses)/len(losses) #return train batch losses