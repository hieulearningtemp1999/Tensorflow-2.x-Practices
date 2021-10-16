import tensorflow as tf
from loss import cross_entropy
from tqdm import tqdm

@tf.function
def train_step(data,target,model,optimizer):
    with tf.GradientTape() as tape:
            pred = model(data,is_training = True)
            loss = cross_entropy(pred,target)
    grads = tape.gradient(loss,model.trainable_weights)
    optimizer.apply_gradients(zip(grads,model.trainable_weights))
    return loss

def train(train_dataset,model,optimizer):
    losses = []
    train_loader = tqdm(train_dataset)
    for (data,target) in (train_loader):
        loss = train_step(data,target,model,optimizer)
        losses.append(loss)

        train_loader.set_postfix(loss_batch = loss)
    print("Train loss : ",sum(losses)/len(losses))
    return sum(losses)/len(losses) #retur train batch losses