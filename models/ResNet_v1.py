import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Input, Flatten, Dense, Activation, AveragePooling2D)
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

SEED = 3222

class BasicBlock(Model):
    expansion = 1
    def __init__(self, num_filters, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2D(num_filters, kernel_size=3, strides=stride, padding='same', kernel_initializer=he_normal(seed=SEED), kernel_regularizer=l2(1e-4))
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(num_filters, kernel_size=3, strides=1, padding='same', kernel_initializer=he_normal(seed=SEED), kernel_regularizer=l2(1e-4))
        self.bn2 = BatchNormalization()
        self.relu = Activation('relu')
        self.downsample = downsample
        self.stride = stride
        
    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = tf.keras.layers.add([out,residual])
        # out = self.relu(out)

        return out
    
class ResNet(Model):
    def __init__(self, depth, num_classes):
        super(ResNet, self).__init__()
        self.inplanes = 16

        n = int((depth - 2) / 6)
        block = BasicBlock

        self.conv1 = Conv2D(self.inplanes, kernel_size=3, strides=1, padding='same', kernel_initializer=he_normal(seed=SEED), kernel_regularizer=l2(1e-4))
        self.bn1 = BatchNormalization()
        self.relu = Activation('relu')
        self.layer1 = self._make_layer(block, 16, n) # (32,32,16)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = AveragePooling2D(pool_size=8)
        self.flatten = Flatten()
        self.dense = Dense(num_classes, kernel_initializer=he_normal(seed=SEED))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = Sequential([
                Conv2D(planes, kernel_size=1, strides=stride, padding='same', kernel_initializer=he_normal(seed=SEED), kernel_regularizer=l2(1e-4)),
                BatchNormalization(),
            ])

        layers = []
        layers.append(block(planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(planes))

        return Sequential(layers)

    def call(self, x):

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x
    
    def get_bn_before_relu(self):
        
        bn1 = self.layer1.layers[-1].bn2
        bn2 = self.layer2.layers[-1].bn2
        bn3 = self.layer3.layers[-1].bn2

        return [bn1, bn2, bn3]
    
    def get_channel_num(self):
    
        return [16, 32, 64]
    
    def extract_feature(self, x, preReLU=False):
    
        x = self.conv1(x)
        x = self.bn1(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)

        x = self.relu(feat3)
        x = self.avgpool(x)
        x = self.flatten(x)
        out = self.dense(x)
        
        if not preReLU:
            feat1 = self.relu(feat1)
            feat2 = self.relu(feat2)
            feat3 = self.relu(feat3)

        return [feat1, feat2, feat3], out