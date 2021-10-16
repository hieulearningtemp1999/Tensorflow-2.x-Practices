import tensorflow as tf
import numpy as np
from datasets import geo_shape

def get_true_normalized_boxes(raw_boxes,img_size=224):
    """
    raw_boxes : batchsize,S,S,4
    """
    box_x1 = (raw_boxes[...,0:1] - raw_boxes[...,2:3]/2.)/img_size
    box_x2 = (raw_boxes[...,0:1] + raw_boxes[...,2:3]/2.)/img_size
    box_y1 = (raw_boxes[...,1:2] - raw_boxes[...,3:4]/2.)/img_size
    box_y2 = (raw_boxes[...,1:2] + raw_boxes[...,3:4]/2.)/img_size
    # print(box_x1.shape)
    box = tf.concat([box_x1,box_y1,box_x2,box_y2],axis=-1)
    return box

def convert_label_to_bndboxes(bndboxes):
    """

    :param bndboxes: np array with shape [batchsize,S,S,4]
    :return: box_xy , box_wh (batchsize,S,S,2)
    """
    cell_height_idx = tf.range(bndboxes.shape[1])
    cell_width_idx = tf.range(bndboxes.shape[2])
    cell_height_idx = tf.tile(cell_height_idx,[bndboxes.shape[2]]) # SxS : 0 1 2 3...S 0 1 2 3.. S ....
    cell_width_idx = tf.tile(np.expand_dims(cell_width_idx,0),[bndboxes.shape[1],1]) #SxS
    cell_width_idx = (tf.transpose(cell_width_idx))
    cell_width_idx = tf.reshape(cell_width_idx,(-1)) # 0 0 0 0  . . . ... . . S S S S S
    cell_idx = tf.transpose(tf.stack([cell_height_idx,cell_width_idx]))# SxS,2 (loc_x,loc_y) : [0,0],[1,0],...[S,0],[0,1],[1,1],...[S,1],....
    cell_idx = tf.reshape(cell_idx,(1,bndboxes.shape[1],bndboxes.shape[2],2))
    box_x = (bndboxes[...,0:1]+tf.cast(cell_idx[...,0:1],tf.float32))/(bndboxes.shape[2])
    box_y = (bndboxes[...,1:2]+tf.cast(cell_idx[...,1:2],tf.float32))/(bndboxes.shape[1])
    box_w = tf.square(bndboxes[...,2:3])
    box_h = tf.square(bndboxes[...,3:])
    box_x_1 = box_x - box_w/2.
    box_x_2 = box_x + box_w/2.
    box_y_1 = box_y - box_h/2.
    box_y_2 = box_y + box_h/2.
    box = tf.concat([box_x_1,box_y_1,box_x_2,box_y_2],axis=-1)
    return box

def cal_iou(box_1,box_2,box_format=1):
    """
    function calculates iou of two bndboxes
    :param box_1: 4-D tensor with shape (batchsize,S,S,4)
    :param box_2: 4-D tensor with shape (batchsize,S,S,4)
    :param box_format : 0 : corner , 1 : center
    :return:
    """
    if box_format == 1:
        #Get box1 infos
        box_1_x = box_1[...,0:1] #batchsize,S,S,1
        box_1_y = box_1[...,1:2] #batchsize,S,S,1
        box_1_w = box_1[...,2:3] #batchsize,S,S,1
        box_1_h = box_1[...,3:4] #batchsize,S,S,1
        box_1_xmin = box_1_x - box_1_w/2 #batchsize,S,S,1
        box_1_ymin = box_1_y - box_1_h/2 #batchsize,S,S,1
        box_1_xmax = box_1_x + box_1_w/2 #batchsize,S,S,1
        box_1_ymax = box_1_y + box_1_h/2 #batchsize,S,S,1

        #Get box2 infos
        box_2_x = box_2[..., 0:1]  # batchsize,S,S,1
        box_2_y = box_2[..., 1:2]  # batchsize,S,S,1
        box_2_w = box_2[..., 2:3]  # batchsize,S,S,1
        box_2_h = box_2[..., 3:4]  # batchsize,S,S,1
        box_2_xmin = box_2_x - box_2_w / 2  # batchsize,S,S,1
        box_2_ymin = box_2_y - box_2_h / 2  # batchsize,S,S,1
        box_2_xmax = box_2_x + box_2_w / 2  # batchsize,S,S,1
        box_2_ymax = box_2_y + box_2_h / 2  # batchsize,S,S,1

    else:
        box_1_xmin = box_1[...,0:1]
        box_1_ymin = box_1[...,1:2]
        box_1_xmax = box_1[...,2:3]
        box_1_ymax = box_1[...,3:4]

        box_2_xmin = box_2[...,0:1]
        box_2_ymin = box_2[...,1:2]
        box_2_xmax = box_2[...,2:3]
        box_2_ymax = box_2[...,3:4]

    x1 = tf.math.maximum(box_1_xmin,box_2_xmin)
    y1 = tf.math.maximum(box_1_ymin,box_2_ymin)
    x2 = tf.math.minimum(box_1_xmax,box_2_xmax)
    y2 = tf.math.minimum(box_1_ymax,box_2_ymax)
    intersect = np.clip(x2-x1,0,None)*np.clip(y2-y1,0,None)
    area_1 = tf.math.abs((box_1_xmax-box_1_xmin)*(box_1_ymax-box_1_ymin))
    area_2 = tf.math.abs((box_2_xmax-box_2_xmin)*(box_2_ymax-box_2_ymin))
    return intersect/(area_1+area_2-intersect+1e-7) #batchsize,S,S,1

def yolo_output2boxes(y,img_size=224,S=7,B=2,C=3):
    classes = y[...,5*B:]
    
    classes_idx = tf.argmax(classes,axis=3)
    max_classes = np.max(classes,axis=3)
    classes_per_boxes = tf.tile(classes_idx[:,:,:,tf.newaxis,tf.newaxis],multiples=[1,1,1,B,1]) #batchsize,S,S,B,1
    max_classes_per_boxes = tf.tile(max_classes[:,:,:,tf.newaxis,tf.newaxis],multiples=[1,1,1,B,1]) #batchsize,S,S,B,1

    #Get coordinate and confident

    pred_boxes = []
    for i in range(B):
        raw_boxes = y[:,:,:,(5*i+1):(5*i+5)]
        # print("raw_boxes :",raw_boxes.shape)
        confs = y[:,:,:,(5*i):(5*i+1)] #batchsize,S,S,1
        if B==1:
            boxes_coords = get_true_normalized_boxes(raw_boxes) #batchsize,S,S,4
            # print(boxes_coords.shape)
        else:
            boxes_coords = convert_label_to_bndboxes(raw_boxes)
        # print("boxcodr : ",boxes_coords.shape)
        boxes = tf.concat((boxes_coords,confs),axis=-1)
        pred_boxes.append(tf.expand_dims(boxes,axis=3))
    pred_boxes = tf.concat(pred_boxes,axis=3) #batchisze,S,S,B,5
    probs = tf.cast(max_classes_per_boxes,tf.float32) * pred_boxes[:,:,:,:,4:]
    res = tf.concat((pred_boxes,tf.cast(classes_per_boxes,tf.float32),tf.cast(probs,tf.float32)),axis=-1) #batchsize,S,S,B,6
    res = tf.reshape(res,(res.shape[0],res.shape[1]*res.shape[2]*res.shape[3],res.shape[4])) #batchsize,S*S*B,7 (7 : x,y,w,h,conf,class,probs)
    # print(res.shape)  
    return res #batchsize,N,7 (N=S*S*B)

def decoder(bboxes,iou_threshold=0.5,score_threshold=0.5):
    bboxes = bboxes.numpy()
    x1 = bboxes[:,0] 
    x2 = bboxes[:,2] 
    y1 = bboxes[:,1] 
    y2 = bboxes[:,3]
    new_bboxes = np.zeros((bboxes.shape[0],4))
    new_bboxes[:,0] = y1
    new_bboxes[:,1] = x1
    new_bboxes[:,2] = y2
    new_bboxes[:,3] = x2
    scores = bboxes[:,4]
    keep_idxs = tf.image.non_max_suppression(new_bboxes,scores=scores,max_output_size=len(bboxes),iou_threshold=iou_threshold,score_threshold=score_threshold)
    keep_boxes = tf.gather(bboxes,keep_idxs)
    # print(keep_boxes.shape)
    return keep_boxes # (N,7)

def evaluate(target,pred,B=2,num_classes=3,iou_threshold=0.5,score_threshold=0.5):

    target_boxes = yolo_output2boxes(target, B=1)  # batchsize,S*S*1,7
    pred_boxes = yolo_output2boxes(pred, B=B)  # batchsize,S*S*B,7
    # print(target_boxes.shape,pred_boxes.shape)
    mAPs = []
    for i in range(len(target_boxes)):
        average_precisions = []
        target_boxes_decoded = decoder(target_boxes[i], iou_threshold, score_threshold)  # (N,7)
        pred_boxes_decoded = decoder(pred_boxes[i], iou_threshold, score_threshold)  # (M,7)
        for c in range(num_classes):
            dets = pred_boxes_decoded[pred_boxes_decoded[:, 5] == c] # Num_dets_perclass,7
            gts = target_boxes_decoded[target_boxes_decoded[:, 5] == c]  # num_gts_perclass,7
            if len(gts) == 0:
                continue

            # amount_bboxes = Counter([gt[5] for gt in gts])
            # for key,val in amount_bboxes.items():
            #     amount_bboxes[key] = np.zeros(val)
            # print(amount_bboxes)
            num_gts = len(gts)
            sorted(dets,key=lambda x: x[-1], reverse=True)  # sort by probs
            TP = np.zeros(shape=(len(dets)))
            FP = np.zeros(shape=(len(dets)))
            # print(TP.shape)
            for det_idx, det in enumerate(dets):
                best_iou = 0
                for idx,gt in enumerate(gts):
                    iou = cal_iou(det[:4],gt[:4],box_format=0)
                    if iou > best_iou:
                        best_gt_idx = idx
                        best_iou = iou
                print(c,best_iou)
                if best_iou > 0.25:
                    # if amount_bboxes[det[5]][best_gt_idx] == 0:
                    TP[det_idx]=1
                    # amount_bboxes[det[5]][best_gt_idx] = 1
                    # else:
                    #     FP[det_idx]=1
                else:
                    FP[det_idx] = 1
            # print(TP,FP)
            TP_cumsum = np.cumsum(TP,axis=0)
            FP_cumsum = np.cumsum(FP,axis=0)
            recalls = TP_cumsum / (num_gts+1e-6)
            precisions = TP_cumsum / (TP_cumsum+FP_cumsum+1e-6)
            precisions = np.concatenate((np.array([1]),precisions))
            recalls = np.concatenate((np.array([0]),recalls))
            average_precisions.append(np.trapz(precisions,recalls))
        # print(average_precisions)
        mAP = sum(average_precisions)/len(average_precisions)
        mAPs.append(mAP)
        break
    # print(mAPs)
    return sum(mAPs)/len(mAPs)


    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2
    json_path = "D:\\Deep Learning Repos\\YOLOv1-Implementation-Tensorflow-2\\datasets\\labels.json"
    img_root = "D:\\Deep Learning Repos\\datasets\\geo_shapes\\train"
    #Get dataset
    train_dataset,test_dataset = geo_shape.get_dataset(json_path,img_root,test_size=0.2,img_size=224,cell_size=7,num_classes=3)

    #Get loader
    train_loader = geo_shape.get_loader(train_dataset,64)
    test_loader = geo_shape.get_loader(test_dataset,64)
    for data,target in test_loader:
        mAP = evaluate(target,target,B=1)
        print(mAP)
        y_true = yolo_output2boxes(target,B=1)
        y_true_boxes_i = decoder(y_true[0])
        y_true_boxes_i[:,3:3+4]*224
        img = data[0]
        for i in range(len(y_true_boxes_i)):
            start_point = (int(y_true_boxes_i[i,0]),int(y_true_boxes_i[i,1]))
            end_point = (int(y_true_boxes_i[i,2]),int(y_true_boxes_i[i,3]))
            # start_x = int(y_true_boxes_i[i,0])
            # start_y = int(y_true_boxes_i[i,1])
            # end_x = int(y_true_boxes_i[i,2])
            # end_y = int(y_true_boxes_i[i,3])
            print(start_point,end_point)
            color = (255,255,0)
            thickness = 2
            # img = cv2.rectangle(img,(start_x,start_y),(end_x,end_y),(255,255,0),2)
        plt.imshow(img)
        
        break
    plt.show()