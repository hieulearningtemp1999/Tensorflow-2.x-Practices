import numpy as np
from loss import YoloLoss
import tensorflow as tf
from tqdm import tqdm
from utils import evaluate

# @tf.function
def test_step(model,data,target,S,B,C):
    pred = model(data)
    test_loss,iou_metric = YoloLoss(pred,target,S,B,C)
    mAP = evaluate(target,pred)

    return test_loss,mAP,iou_metric

def run(test_loader,model,S,B,C):
    print("Training...")
    test_losses = []
    mAPS = []
    iou_metrics = []
    loader = tqdm(test_loader)
    for data,target in loader:
        test_loss,mAP,iou_metric = test_step(model,data,target,S,B,C)
        test_losses.append(test_loss)
        mAPS.append(mAP)
        iou_metrics.append(iou_metric)
        loader.set_postfix(test_loss_batch = test_loss,test_map_batch=mAP,test_iou_batch=iou_metric)
        

    return np.mean(test_losses),np.mean(mAPS),np.mean(iou_metrics)
