import os
import time
import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

import distiller
import models.ResNet_v1 as RN


CP_PATH = 'data/rn44_cp/rn44_cp'
BATCH_SIZE = 128
EPOCHS = 5
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
AUTO = tf.data.experimental.AUTOTUNE
SEED = 3222

# Data Augmentation Functions
def random_shift(image):
    return  keras.preprocessing.image.random_shift(image, 0.1, 0.1,  row_axis=0, col_axis=1, channel_axis=2)

def augment(image, label):
    image = tf.numpy_function(random_shift, [image], tf.float32)
    image = tf.image.random_flip_left_right(image)
    return image, label

# Data Loading
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_ds = train_ds.shuffle(10000, seed=SEED)
train_ds = train_ds.map(augment)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTO)

test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTO)


# Model Loading
## Model Architectures can be changed here
## Only ResNetv1 variants are implemented in this codebase
t_net = RN.ResNet(depth=44, num_classes=10) # teacher model (pre-trained)
t_net.load_weights(CP_PATH)

s_net = RN.ResNet(depth=20, num_classes=10) # student model

d_net = distiller.Distiller(t_net, s_net) # distiller model

# Loss Functions for distillation training
criterion_CE = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
train_accuracy = keras.metrics.CategoricalAccuracy()
train_loss = keras.metrics.Mean()
test_accuracy = keras.metrics.CategoricalAccuracy()
test_loss = keras.metrics.Mean()

optimizer = keras.optimizers.Adam() # optimizer initialization

@tf.function
def train_step(inputs):
    images, labels = inputs

    with tf.GradientTape() as tape:
        logits, loss_distill = d_net(images, training=True)
        loss_ce = criterion_CE(labels, logits)
        loss = loss_ce+ K.sum(loss_distill)/BATCH_SIZE/1000

    gradients = tape.gradient(loss, d_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, d_net.trainable_variables))
    train_accuracy(labels, logits)
    train_loss(loss)
    
@tf.function
def test_step(inputs):
    images, labels = inputs

    logits, loss_distill = d_net(images, training=False)
    loss = criterion_CE(labels, logits)

    test_accuracy(labels, logits)
    test_loss(loss)
    
def train_with_distill(d_net, epochs):
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        test_loss.reset_states()
        test_accuracy.reset_states()

        print('\nDistillation epoch: %d' % epoch)

        #step = 0
        for data in train_ds:
            train_step(data)
            #print("batch "+str(step)+" loss: "+ str(train_loss.result()))
            #step +=1
        
        for data in test_ds:
            test_step(data)

        print(
        f"Epoch: {epoch+1} \nTrain Losses: {train_loss.result()} "
        f"Train Accuracy: {train_accuracy.result()*100} \n"
        f"Test Losses: {test_loss.result()} " 
        f"Test Accuracy: {test_accuracy.result()*100}"
        )

train_with_distill(d_net, EPOCHS)