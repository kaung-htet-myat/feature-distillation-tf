import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization
import tensorflow.keras.backend as K
from tensorflow.keras.losses import MSE

from scipy.stats import norm
import scipy
import math


def squared_distance(source, target):
    # Calculate (student_feature-teacher_feature)**2
    
    return K.square(target-source)

def distillation_loss(source, target, margin):
    target = K.maximum(target, margin)
    loss = squared_distance(source, target)
    mask = K.cast(((source > target) | (target > 0)), dtype="float32")
    
    loss = loss * mask
    return K.sum(loss)

def build_feature_connector(t_channel, s_channel):
    C = [Conv2D(t_channel, kernel_size=1, strides=1, padding='same'),
         BatchNormalization()]

    return Sequential(C)

def get_margin_from_BN(bn):
    margin = []
    std = bn.get_weights()[0] 
    mean = bn.get_weights()[1] 
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return tf.convert_to_tensor(margin, dtype=tf.float32)

class Distiller(Model):
    def __init__(self, t_net, s_net):
        super(Distiller, self).__init__()

        t_channels = t_net.get_channel_num() # [16,16*wf,32*wf,64*wf] suppose t_net is WRN
        s_channels = s_net.get_channel_num() # [16,32,64] suppose s_net is RN

        # each connector for each channel pairs is a Sequential model on its own
        # when the channel numbers are different, the connector is built for size of the student channel
        self.Connectors = [build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)]

        # teacher_bns = [last_bn_layer_of_16, last_bn_layer_of_32, last_bn_layer_of_64]
        teacher_bns = t_net.get_bn_before_relu()
        
        # shape of margins = [16,16*wf,32*wf,64*wf]
        # 1 margin value for each channel
        self.margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        
        for i, margin in enumerate(self.margins):
            # Expand the margin size to be the same as that of teacher features
            self.margins[i] = K.expand_dims(K.expand_dims(K.expand_dims(margin, axis=0), axis=1), axis=2)
            
        self.t_net = t_net
        self.s_net = s_net
        
        for l in self.t_net.layers:
            l.trainable = False

    def call(self, x):

        t_feats, t_out = self.t_net.extract_feature(x, preReLU=True)
        s_feats, s_out = self.s_net.extract_feature(x, preReLU=True)
        feat_num = len(t_feats) # (3,k,k,channel_size)

        loss_distill = 0
        for i in range(feat_num):
            #print(s_feats[i].shape)
            s_feats[i] = self.Connectors[i](s_feats[i]) # (1,k,k,teacher_channel_size)
            loss_distill += distillation_loss(s_feats[i], t_feats[i], self.margins[i] / 2 ** (feat_num - i - 1))

        return s_out, loss_distill