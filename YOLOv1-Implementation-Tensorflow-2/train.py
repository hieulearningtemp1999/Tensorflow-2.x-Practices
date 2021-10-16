from loss import YoloLoss
import tensorflow as tf
from tqdm import tqdm
from utils import evaluate
import numpy as np

# @tf.function
def train_step(model, data, target, optimizer,S,B,C):
    with tf.GradientTape() as tape:
        pred = model(data)
        train_loss,iou_metric = YoloLoss(pred, target,S,B,C)
    grads = tape.gradient(train_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    mAP = evaluate(target, pred)
    return train_loss,mAP,iou_metric


def run(train_loader, model, optimizer,S,B,C):
    print("Training...")
    train_losses = []
    mAPS = []
    iou_metrics = []
    loader = tqdm(train_loader)
    for data, target in loader:
        train_loss,mAP,iou_metric = train_step(model, data, target, optimizer,S,B,C)
        train_losses.append(train_loss)
        mAPS.append(mAP)
        iou_metrics.append(iou_metric)
        loader.set_postfix(train_loss_batch=train_loss,train_map_batch=mAP,train_iou_batch=iou_metric)
        
        
    return np.mean(train_losses),np.mean(mAPS),np.mean(iou_metrics)
