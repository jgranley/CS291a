import tensorflow as tf
from keras import backend as K
from keras import layers
import keras
import numpy as np
from skimage.transform import resize
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pulse2percept as p2p
from pulse2percept.models import BiphasicAxonMapModel
from pulse2percept.stimuli import BiphasicPulseTrain
from pulse2percept.utils import center_image
from pulse2percept.implants import ArgusII

from .dataset_gen import *


def get_loss(model, implant, regularize=None, reg_coef=0.05):
    bundles = model.grow_axon_bundles()
    axons = model.find_closest_axon(bundles)
    axon_contrib = model.calc_axon_sensitivity(axons, pad=True).astype(np.float32)
    axon_contrib = tf.constant(axon_contrib, dtype='float32')
    x = tf.constant([implant[e].x for e in implant.electrodes], dtype='float32')
    y = tf.constant([implant[e].y for e in implant.electrodes], dtype='float32')
    rho = model.rho
    # get effect models. Need to reimplement them in tensorflow fashion
    def scale_threshold(pdur):
        return model.a1 + model.a0*pdur
    def predict_freq_amp(amp, freq):
        return model.a2*amp + model.a3*freq
    def bright(freq, amp, pdur):
        F_bright = predict_freq_amp(amp * scale_threshold(pdur), freq)
        return F_bright
    def size(freq, amp, pdur):
        min_f_size = 10**2 / (model.rho**2)
        F_size = model.a5 * amp * model.scale_threshold(pdur) + model.a6
        return tf.maximum(F_size, min_f_size)
    def streak(freq, amp, pdur):
        min_f_streak = 10**2 / (model.axlambda ** 2)
        F_streak = model.a9 - model.a7 * pdur ** model.a8
        return tf.maximum(F_streak, min_f_streak)

    def reg_none(y_pred):
        return tf.zeros((len(y_pred)))
    def reg_l1(y_pred):
        return tf.reduce_sum(tf.abs(y_pred[:, :, 1]), axis=-1) + tf.reduce_sum(tf.abs(y_pred[:, :, 0]), axis=-1)
    def reg_l2(y_pred):
        return tf.reduce_mean(y_pred[:, :, 1]**2, axis=-1)
    if regularize is None:
        reg = reg_none
    elif regularize == 'l1':
        reg = reg_l1
    elif regularize == 'l2':
        reg = reg_l2
    else:
        reg = reg_none

    def biphasic_axon_map_batched(ypred):
        bright_effects = bright(ypred[:, :, 0], 
                                ypred[:, :, 1], 
                                ypred[:, :, 2])
        # make bright effects 0 if amp is 0
        # mask = tf.cast(ypred[:, :, 0] > 0.5, 'float32')
        # bright_effects = bright_effects * mask
        size_effects = size(ypred[:, :, 0], 
                            ypred[:, :, 1], 
                            ypred[:, :, 2])
        streak_effects = streak(ypred[:, :, 0], 
                                ypred[:, :, 1], 
                                ypred[:, :, 2])
        eparams = tf.stack([bright_effects, size_effects, streak_effects], axis=2)
        d2_el = (axon_contrib[:, :, 0, None] - x)**2 + (axon_contrib[:, :, 1, None] - y)**2
        intensities = eparams[:, None, None, :, 0] * tf.math.exp(-d2_el[None, :, :, :] / (2. * rho**2 * eparams[:, :, 1])[:, None, None, :]) * (axon_contrib[None, :, :, 2, None] ** (1./eparams[:, None, None, :, 2]))
        return tf.reduce_max(tf.reduce_sum(intensities, axis=-1), axis=-1)

    # assumes model outputs same shape as ytrue
    def mse(ytrue, ypred):
        pred_imgs = biphasic_axon_map_batched(ypred)
        yt = tf.reshape(ytrue, (-1, model.grid.shape[0] * model.grid.shape[1]))
        loss = tf.reduce_mean((pred_imgs - yt)**2, axis=-1) + reg_coef * reg(ypred)
        return loss
    
    return tf.function(mse, jit_compile=True)

def get_model(implant, input_shape, num_dense=0):
    """ Makes a keras model for the model
    """
    inputs = layers.Input(shape=input_shape, dtype='float32')
    x = tf.image.flip_up_down(inputs)

    # fully convolutional
    num_filters = [100, 1000, 100]
    kernel_sizes = [5, 5, 5]
    # for idx_conv in range(3):
    #     x = layers.Conv2D(num_filters[idx_conv], kernel_sizes[idx_conv], padding='same')(x)
    #     if idx_conv <2:
    #         x = layers.MaxPool2D()(x)
    #     x = layers.Activation('relu')(x)
    
    x = layers.Flatten()(x)
    for i in range(num_dense):
        x = layers.Dense(500, activation='relu')(x)
    amps = layers.Dense(len(implant.electrodes))(x)
    amps = layers.ReLU()(amps)
    freqs = layers.Dense(len(implant.electrodes))(x)
    freqs = layers.ReLU()(freqs)
    # freqs = tf.where(amps >= 0.5, freqs, tf.zeros_like(freqs))
    pdurs = layers.Dense(len(implant.electrodes))(x)
    pdurs = layers.ReLU()(pdurs) + 1e-3
    # pdurs = tf.where(amps >= 0.5, pdurs, tf.zeros_like(pdurs))
    outputs = tf.stack([freqs, amps, pdurs], axis=-1)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def train_model(nn, model, implant, reg, reg_coef, datatype, opt, learning_rate, batch_size=32, filename="", num_dense=0):
    data_dir = "../data"
    data_path = os.path.join(data_dir, datatype, filename)
    results_folder = os.path.join("../results", datatype)

    ex = np.array([implant[e].x for e in implant.electrodes], dtype='float32')
    ey = np.array([implant[e].y for e in implant.electrodes], dtype='float32')

    targets, stims = read_h5(data_path)
    targets = targets.reshape([-1] + list(model.grid.shape) + [1])
    targets_train, targets_test, stims_train, stims_test = train_test_split(targets, stims, test_size=0.2)
    
    if opt == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif opt == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    lossfn = get_loss(model, implant, regularize=reg, reg_coef=reg_coef)

    modelname = (f"nn"
                f"_{len(implant.electrodes)}elecs"
                f"_{opt}_lr{learning_rate}"
                f"_{str(reg)}")
    log_dir = "../results/tensorboard/" + modelname + datetime.datetime.now().strftime("%m%d-%H%M%S")

    modelpath = os.path.join(results_folder, modelname)
    tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    cp = tf.keras.callbacks.ModelCheckpoint(modelpath, save_best_only=False)
    es = tf.keras.callbacks.EarlyStopping(patience=30, monitor='loss')

    nn.compile(optimizer=optimizer, loss=lossfn, metrics=[lossfn])
    hist = model.fit(x=targets_train, y=targets_train, batch_size=batch_size, epochs=1000,
                     callbacks=[tb, cp, es], validation_data=(targets_test, targets_test), validation_batch_size=32)
    
    # save_model
    json_path = os.path.join(results_folder)
    
    if os.path.exists(json_path):
        info = json.load(open(json_path))
    else:
        info = {}
    info[modelname] = {}
    info[modelname]['test_loss'] = hist['val_loss'][-1]
    info[modelname]['train_loss'] = hist['val_loss'][-1]
    info[modelname]['epochs'] = len(hist['val_loss'])
    info[modelname]['opt'] = opt
    info[modelname]['lr'] = learning_rate
    info[modelname]['reg'] = reg
    info[modelname]['reg_coef'] = reg_coef
    info[modelname]['batch_size'] = batch_size
    info[modelname]['dense_layers'] = num_dense
    info[modelname]['tensorboard_logdir'] = log_dir
    info[modelname]['rho'] = model.rho
    info[modelname]['lambda'] = model.axlambda
    info[modelname]['n_elecs'] = len(implant.electrodes)
    info[modelname]['shape'] = str(model.grid.shape)
    info[modelname]['']
    info[modelname]['']
    json.dump(info, open(json_path, 'w'))

    # plot some images
    if not os.path.exists(os.path.join(results_folder, 'predicted_images')):
        os.mkdir(os.path.join(results_folder, 'predicted_images'))
    