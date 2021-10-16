import tensorflow as tf

class Weights():
    def __init__(self,weight_configs,init_method):
        self.trainable_params = []
        self.weight_configs = weight_configs
        self.init_method = init_method
    
    def get_trainable_params(self):
        num_conv = 0
        num_fc = 0
        for cfg in self.weight_configs:
            if len(cfg) == 2:
                self.trainable_params.append(self.get_weight(self.init_method,cfg,name="fc_"+str(num_fc)))
                num_fc +=1

            elif len(cfg) == 4:
                self.trainable_params.append(self.get_weight(self.init_method,cfg,name="conv_"+str(num_conv)))
                num_conv +=1
        
        return self.trainable_params

    def get_weight(self,init_method,shape,name):
        return tf.Variable(init_method(shape),name=name,trainable=True,dtype=tf.float32)