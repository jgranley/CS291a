import tensorflow as tf
from keras import backend as K
import numpy as np
from skimage.transform import resize

import pulse2percept as p2p
from pulse2percept.models import BiphasicAxonMapModel
from pulse2percept.stimuli import BiphasicPulseTrain
from pulse2percept.utils import center_image
from pulse2percept.implants import ArgusII


def get_loss(model, implant, regularize=None, reg_coef=0.05):
    axon_contrib = tf.constant(model.axon_contrib, dtype='float32')
    x = tf.constant([implant[e].x for e in implant.electrodes])
    y = tf.constant([implant[e].y for e in implant.electrodes])
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

    # def reg_none(y_pred):
    #     return tf.zeros((len(y_pred)))
    # def reg_l1(y_pred):
    #     return tf.reduce_sum(tf.abs(y_pred), axis=-1) 
    # def reg_l2(y_pred):
    #     return tf.reduce_mean(y_pred**2, axis=-1)
    # if regularize is None:
    #     reg = reg_none
    # elif regularize == 'l1':
    #     reg = reg_l1
    # elif regularize == 'l2':
    #     reg = reg_l2

    def biphasic_axon_map_batched(ypred):
        bright_effects = bright(ypred[:, :, 0], 
                                ypred[:, :, 1], 
                                ypred[:, :, 2])
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
        return tf.reduce_mean((tf.reshape(pred_imgs, ytrue.shape) - ytrue)**2)
    
    return mse

@tf.function
def biphasic_axon_map_batched(elec_params, x, y, axon_segments, rho):
    bright_effects = bright(elec_params[:, :, 0], 
                                      elec_params[:, :, 1], 
                                      elec_params[:, :, 2])
    size_effects = size(elec_params[:, :, 0], 
                                  elec_params[:, :, 1], 
                                  elec_params[:, :, 2])
    streak_effects = streak(elec_params[:, :, 0], 
                                      elec_params[:, :, 1], 
                                      elec_params[:, :, 2])
    eparams = tf.stack([bright_effects, size_effects, streak_effects], axis=2)
    # print(eparams)

    d2_el = (axon_segments[:, :, 0, None] - x)**2 + (axon_segments[:, :, 1, None] - y)**2
    intensities = eparams[:, None, None, :, 0] * tf.math.exp(-d2_el[None, :, :, :] / (2. * rho**2 * eparams[:, :, 1])[:, None, None, :]) * (axon_segments[None, :, :, 2, None] ** (1./eparams[:, None, None, :, 2]))
    # return intensities
    return tf.reduce_max(tf.reduce_sum(intensities, axis=-1), axis=-1)