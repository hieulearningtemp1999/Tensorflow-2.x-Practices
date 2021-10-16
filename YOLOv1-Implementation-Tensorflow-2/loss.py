import tensorflow as tf
import numpy as np
from utils import cal_iou

# def YoloLoss(preds,targets,S,B,C):
#     """

#     :param preds: predicted label matrix : np array with shape : batchsize,S,S,C+5*B
#     :param targets: ground truth label matrix : np array with shape : batchsize,S,S,C+5
#     :param S: grid_size
#     :param B: num_boxes predicted per grid
#     :param C: num_classes
#     :return:
#     """
#     preds = tf.cast(preds,tf.float32)
#     targets = tf.cast(targets,tf.float32)
#     batchsize = preds.shape[0]
#     lambda_noobj = 0.5
#     lambda_coord = 5

#     target_class = targets[...,:C] # batchsize ,S,S,C
#     target_box = targets[...,C:C+4] # batchsize,S,S,4
#     target_obj = targets[...,C+4:] # batchsize,S,S,1 (confident score)
#     #target_box = convert_label_to_bndboxes(target_box) #(batchsize,S,S,4)

#     pred_class = preds[...,:C] #batchsize,S,S,C
#     pred_boxes = [] #bndboxes
#     pred_objs = [] #confident scores
#     ious = []
#     for i in range(B):
#         pred_box = (preds[...,(C+5*i):(C+5*i+4)]) #batchsize,S,S,4
#         pred_boxes.append(tf.expand_dims(pred_box,3))
#         pred_obj = preds[...,C+5*i+4] #batchsize,S,S
#         pred_objs.append(tf.expand_dims(pred_obj,0))
#         iou = cal_iou(pred_box,target_box)
#         ious.append(tf.expand_dims(iou,0))
#     ious = tf.concat(ious,axis=0) #B,batchsize,S,S,1
#     ious = tf.transpose(ious,(1,2,3,0,4)) #batchsize,S,S,B,1
#     ious = np.max(ious,axis=4) #batchsize,S,S,B
#     best_box = np.max(ious,axis=3,keepdims=True) #batchsize,S,S,1
#     box_mask = tf.cast(ious >= best_box,ious.dtype) #batchsize,S,S,B
#     pred_objs = tf.concat(pred_objs,axis=0)
#     pred_objs = tf.transpose(pred_objs,(1,2,3,0)) #batchsize,S,S,B

#     #get Pred xy,wh
#     pred_boxes = tf.concat(pred_boxes,axis=3) #batchsize,S,S,B,4
#     pred_xy = pred_boxes[:,:,:,:,:2] #batchsize,S,S,B,2
#     pred_wh = pred_boxes[:,:,:,:,2:4] #batchsize,S,S,B,2

#     #get target xy,wh
#     target_box = tf.expand_dims(target_box, 3)  # batchsize,S,S,1,4
#     target_xy = target_box[:, :, :, :, 0:2]  # batchsize,S,S,1,2
#     target_wh = target_box[:, :, :, :, 2:4]  # batchisze,S,S,1,2

#     """ Class Loss """
#     classLoss = target_obj*tf.square(pred_class - target_class) #batchsize,S,S,C
#     classLoss = (tf.reduce_sum(classLoss))

#     """ Obj Loss"""
#     obj_loss = box_mask * target_obj * tf.square(1-pred_objs)

#     """ NoObj Loss"""
#     noobj_loss = lambda_noobj * (1 - box_mask * target_obj) * tf.square(0-pred_objs)

#     """Confident loss"""
#     conf_loss = (tf.reduce_sum(obj_loss+noobj_loss))

#     """ Box Loss """
#     box_mask = np.expand_dims(box_mask,axis=-1)
#     target_obj = np.expand_dims(target_obj,axis=-1)
#     box_loss = lambda_coord * box_mask * target_obj * tf.square(target_xy - pred_xy) #batchsize,S,S,B,2
#     box_loss += lambda_coord * box_mask * target_obj * tf.square(np.sqrt(np.abs(target_wh))-np.sqrt(np.abs(pred_wh)))
#     box_loss = (tf.reduce_sum(box_loss))

#     """Total Loss"""
#     loss = (classLoss + box_loss + conf_loss)/batchsize
#     return tf.convert_to_tensor(loss,dtype=tf.float32)


def compute_iou(boxes1, boxes2, scope='iou'):
    """calculate ious
    Args:
      boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
      boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
    Return:
      iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    """
    # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
    boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                             boxes1[..., 1] - boxes1[..., 3] / 2.0,
                             boxes1[..., 0] + boxes1[..., 2] / 2.0,
                             boxes1[..., 1] + boxes1[..., 3] / 2.0],
                            axis=-1)

    boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                             boxes2[..., 1] - boxes2[..., 3] / 2.0,
                             boxes2[..., 0] + boxes2[..., 2] / 2.0,
                             boxes2[..., 1] + boxes2[..., 3] / 2.0],
                            axis=-1)

        # calculate the left up point & right down point
    lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
    rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

        # intersection
    intersection = tf.maximum(0.0, rd - lu)
    inter_square = intersection[..., 0] * intersection[..., 1]

        # calculate the boxs1 square and boxs2 square
    square1 = boxes1[..., 2] * boxes1[..., 3]
    square2 = boxes2[..., 2] * boxes2[..., 3]

    union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)
    
def YoloLoss(predicts, labels,S=7,B=2,C=3,img_size=224):
    """calculate loss function
    Args:
      predicts: 4-D tensor [batch_size, 7, 7, 5*nbox+n_class] 
      labels: 4-D tensor [batch_size, 7, 7, 5+n_class]
    Return:
      loss: scalar
    """
    cell_size=S
    box_per_cell = B
    offset = np.transpose(np.reshape(np.array(
            [np.arange(cell_size)] * cell_size * box_per_cell),
            (box_per_cell, cell_size, cell_size)), (1, 2, 0))
    offset = offset[None, :]
    offset = tf.constant(offset, dtype=tf.float32)
    offset_tran = tf.transpose(offset, (0, 2, 1, 3))
        
        # 2 phần tử đầu của vector dự đoán tại một ô vuông là confidence score
    predict_object = predicts[..., :box_per_cell]
        
        # 8 phần tử tiếp theo là dự đoán offset của boundary box và width height
    predict_box_offset = tf.reshape(predicts[...,box_per_cell:5*box_per_cell], (-1, cell_size, cell_size, box_per_cell, 4))
        
        # các phần tử cuối là dự đoán lớp của object
    predict_class = predicts[...,5*box_per_cell:]
        # chuyển vị trí offset về toạ độ normalize trên khoảng [0-1]
    predict_normalized_box = tf.stack(
                                    [(predict_box_offset[..., 0] + offset) / cell_size,
                                     (predict_box_offset[..., 1] + offset_tran) / cell_size,
                                     tf.square(predict_box_offset[..., 2]),
                                    tf.square(predict_box_offset[..., 3])], axis=-1)

    # lấy các nhãn tương ứng 
    true_object = labels[..., :1]
    true_box = tf.reshape(labels[..., 1:5], (-1, cell_size, cell_size, 1, 4))
        
    # để normalize tọa độ pixel về đoạn [0-1] chúng ta chia cho img_size (224)
    true_normalized_box = tf.tile(true_box, (1, 1, 1, box_per_cell, 1))/img_size
    true_class = labels[..., 5:]
        
        # tính vị trí offset từ nhãn 
    true_box_offset =  tf.stack(
                                    [true_normalized_box[..., 0] * cell_size - offset,
                                     true_normalized_box[..., 1] * cell_size - offset_tran,
                                     tf.sqrt(true_normalized_box[..., 2]),
                                     tf.sqrt(true_normalized_box[..., 3])], axis=-1)
        
        # tính iou
    predict_iou = compute_iou(true_normalized_box, predict_normalized_box)
        
        # mask chứa vị trí các ô vuông chứa object
    object_mask = tf.reduce_max(predict_iou, 3, keepdims=True)  
        
        # tính metric để monitor 
    iou_metric = tf.reduce_mean(tf.reduce_sum(object_mask, axis=[1,2,3])/tf.reduce_sum(true_object, axis=[1,2,3]))
        
    object_mask = tf.cast((predict_iou>=object_mask), tf.float32)*true_object

    noobject_mask = tf.ones_like(object_mask) - object_mask
        
        ## class loss
    # print(predict_class.shape,true_class.shape)
    class_delta = true_object*(predict_class - true_class)
    class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1,2,3]), name='class_loss')
        
        ## object loss
    object_delta = object_mask*(predict_object - predict_iou)
    object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1,2,3]), name='object_loss')
        
        ## noobject loss
    noobject_delta = noobject_mask*predict_object
    noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1,2,3]), name='noobject_loss')
        
        ## coord loss
    box_mask = tf.expand_dims(object_mask, 4)
    box_delta = box_mask*(predict_box_offset - true_box_offset)
    box_loss = tf.reduce_mean(tf.reduce_sum(tf.square(box_delta), axis=[1,2,3]), name='box_loss')
        
    loss = 0.5*class_loss + object_loss + 0.1*noobject_loss + 10*box_loss
    # print("HETLOT")
    # return loss, iou_metric, predict_object, predict_class, predict_normalized_box
    return loss,iou_metric