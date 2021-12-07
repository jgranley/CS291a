from matplotlib.colors import Normalize
import tensorflow as tf
from keras import backend as K
from keras import layers
import keras
import numpy as np
from skimage.transform import resize
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
import json
import h5py
import imageio

import pulse2percept as p2p
from pulse2percept.models import BiphasicAxonMapModel
from pulse2percept.stimuli import BiphasicPulseTrain
from pulse2percept.utils import center_image
from pulse2percept.implants import ArgusII

# import dataset_gen

# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

def get_loss(model, implant, regularize=None, reg_coef=0.05, size_norm=False, loss_fn='mse'):
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
        return tf.zeros_like(y_pred[:, 0, 0])
    def reg_l1(y_pred):
        return tf.reduce_sum(tf.abs(y_pred[:, :, 1]), axis=-1)
    def reg_l1_ampfreq(y_pred):
        return tf.reduce_sum(tf.abs(y_pred[:, :, 1]), axis=-1) + tf.reduce_sum(tf.abs(y_pred[:, :, 0]), axis=-1)
    def reg_l2(y_pred):
        return tf.reduce_sum(y_pred[:, :, 1]**2, axis=-1)
    def reg_elecs(y_pred):
        return tf.math.count_nonzero((y_pred[:, :, 1] > 0.5), axis=-1, dtype='float32')

    if regularize is None:
        regfn = reg_none
    elif regularize == 'l1':
        regfn = reg_l1
    elif regularize == 'l1_ampfreq':
        regfn = reg_l1_ampfreq
    elif regularize == 'l2':
        regfn = reg_l2
    elif regularize == 'elecs':
        regfn = reg_elecs
    else:
        regfn = reg_none

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
        loss = tf.reduce_mean((pred_imgs - yt)**2, axis=-1) 
        if size_norm: # normalize by total number of pixels
            loss /= tf.math.count_nonzero(yt, axis=-1, dtype='float32')
            loss *= tf.cast(model.grid.shape[0] * model.grid.shape[1], 'float32')
        loss += reg_coef * regfn(ypred)
        return loss

    def ms_ssim(ytrue, ypred):
        pred_imgs = biphasic_axon_map_batched(ypred)
        pred_imgs = tf.reshape(pred_imgs, (-1, model.grid.shape[0], model.grid.shape[1], 1))
        ytrue = tf.reshape(ytrue, (-1, model.grid.shape[0], model.grid.shape[1], 1))

        loss = 1 - tf.image.ssim_multiscale(ytrue, pred_imgs, 3, power_factors = (0.0448, 0.2856, 0.3001, 0.2363), filter_size=7)
        loss += reg_coef * regfn(ypred)
        return loss

    if loss_fn == 'mse':
        fn = mse
        fn.__name__ = 'mse_' + str(regularize)
        return tf.function(fn, jit_compile=True)
    elif loss_fn == 'msssim':
        fn = ms_ssim
        fn.__name__ = 'msssim_' + str(regularize)
        # cant jit msssim
        return tf.function(fn) 

def get_model(implant, input_shape, num_dense=0, force_zero=False, sigmoid=False, clip=False):
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
    if clip == 'relu':
        amps = layers.ReLU(max_value=10)(amps)
    elif clip == 'sigmoid':
        amps = layers.Activation('sigmoid')(amps) * 10.
    else:
        amps = layers.ReLU()(amps)
    if force_zero:
        amps = tf.where(amps >= 0.5, amps, tf.zeros_like(amps))
    freqs = layers.Dense(len(implant.electrodes))(x)
    if clip == 'relu':
        freqs = layers.ReLU(max_value=200)(freqs)
    elif clip == 'sigmoid':
        freqs = layers.Activation('sigmoid')(freqs) * 200.
    else:
        freqs = layers.ReLU()(freqs)
    if force_zero:
        freqs = tf.where(amps >= 0.5, freqs, tf.zeros_like(freqs))
    pdurs = layers.Dense(len(implant.electrodes))(x)
    if clip == 'relu':
        pdurs = layers.ReLU(max_value=100)(pdurs) + 1e-3
    elif clip == 'sigmoid':
        pdurs = layers.Activation('sigmoid')(pdurs) * 100. + 1e-3
    else:
        pdurs = layers.ReLU()(pdurs) + 1e-3
    if force_zero:
        pdurs = tf.where(amps >= 0.5, pdurs, tf.zeros_like(pdurs))
    outputs = tf.stack([freqs, amps, pdurs], axis=-1)
    if sigmoid:
        mask = layers.Dense(len(implant.electrodes), activation='sigmoid')(x)
        outputs = outputs * mask[:, :, None]
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def train_model(nn, model, implant, reg, targets, stims, reg_coef, datatype, opt, learning_rate, clip=False,
                batch_size=32, num_dense=0, force_zero=False, sigmoid=False, size_norm=False, fonts='all', loss_str='mse'):
    data_dir = "../data"
    results_folder = os.path.join("../results", datatype)
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    ex = np.array([implant[e].x for e in implant.electrodes], dtype='float32')
    ey = np.array([implant[e].y for e in implant.electrodes], dtype='float32')

    targets_train, targets_test, stims_train, stims_test = train_test_split(targets, stims, test_size=0.2)
    if opt == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif opt == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        # can pass in custom optimizer
        optimizer = opt
    lossfn = get_loss(model, implant, regularize=reg, reg_coef=reg_coef, size_norm=size_norm, loss_fn=loss_str)
    loss_noreg = get_loss(model, implant, size_norm=size_norm, loss_fn=loss_str)
    def loss_reg(y_true, y_pred):
        return lossfn(y_true, y_pred) - loss_noreg(y_true, y_pred)
    loss_reg.__name__ = str(reg)

    dt = datetime.datetime.now().strftime("%m%d-%H%M%S")
    modelname = (f"nn_{loss_str}"
                f"_{len(implant.electrodes)}elecs"
                f"_{opt}_lr{learning_rate}"
                f"_{str(reg)}_coef{str(reg_coef)}"
                + dt)
    log_dir = os.path.join("../results/tensorboard/", datatype, modelname)
    modelpath = os.path.join(results_folder, modelname)

    tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    cp = tf.keras.callbacks.ModelCheckpoint(modelpath, save_best_only=False)
    es = tf.keras.callbacks.EarlyStopping(patience=250, monitor='loss', restore_best_weights=True)
    nn.compile(optimizer=optimizer, loss=lossfn, metrics=[loss_noreg, loss_reg])
    hist = nn.fit(x=targets_train, y=targets_train, batch_size=batch_size, epochs=1000,
                     callbacks=[ es], validation_data=(targets_test, targets_test), validation_batch_size=batch_size)
    hist = hist.history
    
    # save_model
    print(f"done training {modelname}")
    nn.save(modelpath)
    json_path = os.path.join(results_folder, "info.json")
    if os.path.exists(json_path):
        info = json.load(open(json_path))
    else:
        info = {}
        info['1elec'] = {}
        info['msssim'] = {}
    if loss_str == 'mse':
        dict_name = '1elec' # compatibility
    elif loss_str == 'msssim':
        dict_name = 'msssim'
    info[dict_name][modelname] = {}
    info[dict_name][modelname]['test_loss'] = np.min(hist['val_loss'])
    info[dict_name][modelname]['train_loss'] = np.min(hist['loss'])
    info[dict_name][modelname]['epochs'] = len(hist['val_loss'])
    info[dict_name][modelname]['opt'] = opt
    info[dict_name][modelname]['lr'] = learning_rate
    info[dict_name][modelname]['reg'] = reg
    info[dict_name][modelname]['reg_coef'] = reg_coef
    info[dict_name][modelname]['batch_size'] = batch_size
    info[dict_name][modelname]['dense_layers'] = num_dense
    info[dict_name][modelname]['tensorboard_logdir'] = log_dir
    info[dict_name][modelname]['rho'] = model.rho
    info[dict_name][modelname]['lambda'] = model.axlambda
    info[dict_name][modelname]['n_elecs'] = 1
    info[dict_name][modelname]['shape'] = str(model.grid.shape)
    info[dict_name][modelname]['force_good_stims'] = str(force_zero)
    info[dict_name][modelname]['sigmoid'] = str(sigmoid)
    info[dict_name][modelname]['size_norm'] = str(size_norm)
    info[dict_name][modelname]['clip'] = clip
    info[dict_name][modelname]['fonts'] = fonts
    json.dump(info, open(json_path, 'w'))

    # plot some images
    if data_type == 'percepts' or data_type == 'alphabet':
        if not os.path.exists(os.path.join(results_folder, 'predicted_images')):
            os.mkdir(os.path.join(results_folder, 'predicted_images'))
        ims_per_row = 15
        rows = 2
        fig, axes = plt.subplots(nrows = rows*2, ncols=ims_per_row, figsize=(20, 20))
        fig.subplots_adjust(wspace=0, hspace=-0.75)
        # predicted first
        for i in range(rows):
            for j in range(ims_per_row):
                if j == 1:
                    plt.ylabel("Preds", fontweight="bold", fontsize=20)
                plt.sca(axes[2*i][j])
                idx = i * ims_per_row + j
                pred = nn(targets_train[idx:idx+1]).numpy()
                score = float(lossfn(targets_train[idx:idx+1], pred).numpy())
                pred_img = model._predict_spatial_jax(pred[0], ex, ey)
                plt.imshow(pred_img.reshape(model.grid.shape), cmap='gray')
                plt.annotate(f"{str(round(score, 3))}", (1, 6), color='white')
                plt.yticks([])
                plt.xticks([])
                axes[2*i][j].spines['bottom'].set_color('gray')
                axes[2*i][j].spines['top'].set_color('gray')
                axes[2*i][j].spines['right'].set_color('gray')
                axes[2*i][j].spines['left'].set_color('gray')
                axes[2*i][j].spines['bottom'].set_linewidth(2)
                axes[2*i][j].spines['top'].set_linewidth(1)
                axes[2*i][j].spines['right'].set_linewidth(2)
                axes[2*i][j].spines['left'].set_linewidth(2)
                # plt.axis(False)
        for i in range(rows):
            for j in range(ims_per_row):
                if j == 1:
                    plt.ylabel("True", fontweight="bold", fontsize=20)
                plt.sca(axes[2*i+1][j])
                idx = i * ims_per_row + j
                plt.imshow(targets_train[idx], cmap='gray')
                # plt.axis(False)
                plt.yticks([])
                plt.xticks([])
                axes[2*i+1][j].spines['bottom'].set_color('gray')
                axes[2*i+1][j].spines['top'].set_color('gray')
                axes[2*i+1][j].spines['right'].set_color('gray')
                axes[2*i+1][j].spines['left'].set_color('gray')
                axes[2*i+1][j].spines['bottom'].set_linewidth(2)
                axes[2*i+1][j].spines['top'].set_linewidth(1)
                axes[2*i+1][j].spines['right'].set_linewidth(2)
                axes[2*i+1][j].spines['left'].set_linewidth(2)
    elif data_type =='alphabet':
        pass
    plt.savefig(os.path.join(results_folder, 'predicted_images', modelname + "_" + str(round(np.min(hist['val_loss']), 3)) +".png"), bbox_inches="tight")

    return round(np.min(hist['val_loss']), 4)


def read_h5(path):
    if not os.path.exists(path):
        raise ValueError("Provided path does not exist")
    hf = h5py.File(path, 'r')
    if 'stims' not in hf.keys() or 'percepts' not in hf.keys():
        raise ValueError("H5 formatted incorrectly")
    stims = np.array(hf.get('stims'), dtype='float32')
    percepts = np.array(hf.get('percepts'), dtype='float32')
    hf.close()
    return percepts, stims

def load_alphabet(path, model, fonts=[i for i in range(31)]):
    folders = os.listdir(path)
    folders = [f for f in folders if f.isnumeric()]
    targets = []
    labels = []
    for folder in folders:
        letters = os.listdir(os.path.join(path, folder))
        for font in fonts:
            if str(font) + ".png" not in letters:
                continue
            img = imageio.imread(os.path.join(path, folder, str(font) + ".png"))
            img = resize(img, model.grid.shape, anti_aliasing=True)
            img = 1 - img # invert
            img = 2 * img # rescale
            targets.append(np.array(img, dtype='float32'))
            labels.append(int(folder))

    targets = np.array(targets, dtype='float32')
    labels = np.array(labels)
    return targets, labels

def encode(target, implant, model, mode='amp', stimrange=(0, 2), maxval=None):
    stim = []
    if maxval is None:
        maxval = np.max(target)
    for elec in implant.electrodes:
        # find location to sample
        x_dva, y_dva = model.retinotopy.ret2dva(implant.electrodes[elec].x, implant.electrodes[elec].y)
        # interpolate?
        # print(x_dva, y_dva)
        x_img = (x_dva - model.xrange[0]) / model.xystep
        y_img = (y_dva - model.yrange[0]) / model.xystep
        
        x_img = int(round(x_img, ndigits=0))
        y_img = int(round(y_img, ndigits=0))
        # image is centered differently
        # print(x_img, y_img)
        # print()
        px_intensity = target[y_img, x_img, 0]
        stim_intensity = px_intensity / maxval * (stimrange[1] - stimrange[0]) + stimrange[0]
        if stim_intensity < 0.5:
            stim_intensity = 0
            freq = 0
        else:
            freq = 20
        pulse = np.array([freq, stim_intensity, 0.45], dtype='float32')
        stim.append(pulse)
    return np.array(stim, dtype='float32')

if __name__ == "__main__":
    model = BiphasicAxonMapModel(axlambda=800, rho=200, a4=0, engine="jax", xystep=0.5, xrange=(-14, 12), yrange=(-12, 12))
    model.build()
    implant = ArgusII(rot=-30)
    #####################################################################
    #                            PERCEPTS                               #
    #####################################################################
    data_type = 'percepts'
    h5_file = 'percepts_argusii_1elec_rho200lam800_12031654.h5'
    targets, stims = read_h5(os.path.join("../data", data_type, h5_file))
    targets = targets.reshape((-1, 49, 53, 1))
    # print(targets)
    # print(targets.dtype)
    # print(targets.shape)
    # test opts / learning rates
    # best_loss = 99999
    # best_opt = ""
    # best_lr = 999
    # for opt in ['sgd', 'adam']:
    #     for lr in [0.00001, 0.00005, 0.0001, 0.001]:
    #         nn = get_model(implant, targets[0].shape)
    #         loss = train_model(nn, model, implant, None, targets, stims, 0.005, data_type, opt, lr)
    #         if loss < best_loss:
    #             best_loss = loss
    #             best_opt = opt
    #             best_lr = lr
                
    # test architectures
    # best_loss = 9999
    # best_ndense = 0
    # best_force = False
    # for n_dense in [0, 1, 3]:
    #     for force_zero in [True, False]:
    #         nn = get_model(implant, targets[0].shape, num_dense=n_dense, force_zero=force_zero)
    #         loss = train_model(nn, model, implant, None, targets, stims, 0.005, data_type, best_opt, best_lr, num_dense=n_dense, force_zero=force_zero)
    #         if loss < best_loss:
    #             best_loss = loss
    #             best_ndense = n_dense
    #             best_force = force_zero

    # test regularization / coef
    # best_loss = 9999
    # for reg in ['l1', 'l2', 'l1_ampfreq', 'elecs']:
    #     for coef in [0.005, 0.01]:
    #         for lr, opt in zip([0.0001, 0.00005], ['adam', 'adam']):
    #             nn = get_model(implant, targets[0].shape, num_dense=1, force_zero=False)
    #             loss = train_model(nn, model, implant, reg, targets, stims, coef, data_type, opt, lr, num_dense=1, force_zero=False)
    #             if loss < best_loss:
    #                 best_loss = loss
    # sig = False
    # for lr, opt in zip([ 0.00001], ['adam']):
    #     nn = get_model(implant, targets[0].shape, num_dense=1, force_zero=False, sigmoid=True)
    #     loss = train_model(nn, model, implant, None, targets, stims, 0.0, data_type, opt, lr, num_dense=1, force_zero=False, sigmoid=True)
    #     if loss < 0.07:
    #         sig = True

    # for lr, opt in zip([0.00001], ['adam']):
    #     for sig in [True, False]:
    #         nn = get_model(implant, targets[0].shape, num_dense=1, force_zero=False, sigmoid=sig)
    #         loss = train_model(nn, model, implant, 'elecs', targets, stims, 0.005 * 7.5, data_type, opt, lr, num_dense=1, force_zero=False, sigmoid=sig, size_norm=True)

    # print(best_loss)



    #####################################################################
    #                            ALPHABET                               #
    #####################################################################

    # data_type = 'alphabet'
    # letters, labels = load_alphabet("../data/alphabet", model)
    # targets = letters.reshape((-1, 49, 53, 1))

    # test opts / learning rates
    # for opt in ['sgd', 'adam']:
    #     for lr in [0.00001, 0.0001, 0.001]:
    #         nn = get_model(implant, targets[0].shape, num_dense=1)
    #         loss = train_model(nn, model, implant, None, targets, labels, 0.005, data_type, opt, lr)
    #         if loss < best_loss:
    #             best_loss = loss
    #             best_opt = opt
    #             best_lr = lr

    # for n_dense in [0, 1, 3]:
    #     nn = get_model(implant, targets[0].shape, num_dense=n_dense, force_zero=False)
    #     loss = train_model(nn, model, implant, None, targets, labels, 0.005, data_type, best_opt, best_lr, num_dense=n_dense, force_zero=False)
    #     if loss < best_loss:
    #         best_loss = loss
    #         best_ndense = n_dense

    # test regularization / coef
    # # best_loss = 9999
    # for reg in ['l1', 'l2', 'l1_ampfreq', 'elecs']:
    #     for coef in [0.05]:
    #         for lr, opt in zip([0.0001], ['adam']):
    #             nn = get_model(implant, targets[0].shape, num_dense=1, force_zero=False)
    #             loss = train_model(nn, model, implant, reg, targets, labels, coef, data_type, opt, lr, num_dense=1, force_zero=False)
                # if loss < best_loss:
                #     best_loss = loss
    # sig = False
    # # sigmoid
    # for lr, opt in zip([ 0.00001, 0.0001], ['adam', 'adam']):
    #     nn = get_model(implant, targets[0].shape, num_dense=1, force_zero=False, sigmoid=True)
    #     loss = train_model(nn, model, implant, None, targets, labels, 0.0, data_type, opt, lr, num_dense=1, force_zero=False, sigmoid=True)
    #     if loss < 0.07:
    #         sig = True

    # for lr, opt in zip([0.00001], ['adam']):
    #     for sig in [True, False]:
    #         nn = get_model(implant, targets[0].shape, num_dense=1, force_zero=False, sigmoid=False)
    #         loss = train_model(nn, model, implant, 'elecs', targets, labels, 0.005 * 7.5, data_type, opt, lr, num_dense=1, force_zero=False, sigmoid=False, size_norm=True)
   
    # nn = get_model(implant, targets[0].shape, num_dense=1, force_zero=False)
    # loss = train_model(nn, model, implant, None, targets, labels, 0.0, data_type, 'adam', 0.0002, num_dense=1, force_zero=False)

    # for clip in ['relu', 'sigmoid']:
    #     nn = get_model(implant, targets[0].shape, num_dense=1, force_zero=False, clip=clip)
    #     loss = train_model(nn, model, implant, None, targets, labels, 0.05, data_type, 'adam', 0.0001, num_dense=1, force_zero=False, clip=clip)


    # for font in range(1, 31):
    #     data_type = 'alphabet'
    #     letters, labels = load_alphabet("../data/alphabet", model, fonts=[font])
    #     targets = letters.reshape((-1, 49, 53, 1))
    #     if len(targets) == 0:
    #         continue
    #     nn = get_model(implant, targets[0].shape, num_dense=1, force_zero=False, clip='relu')
    #     loss = train_model(nn, model, implant, None, targets, labels, 0.0, data_type, 'adam', 0.0001, num_dense=1, force_zero=False, fonts=font, clip='relu', batch_size=16)

    # for font in [28, 27, 26, 25, 21, 20, 10]:
    #     data_type = 'alphabet'
    #     letters, labels = load_alphabet("../data/alphabet", model, fonts=[font])
    #     targets = letters.reshape((-1, 49, 53, 1))
    #     if len(targets) == 0:
    #         continue
    #     nn = get_model(implant, targets[0].shape, num_dense=1, force_zero=False, clip='relu')
    #     loss = train_model(nn, model, implant, 'l1', targets, labels, 0.0001, data_type, 'adam', 0.0001, num_dense=1, force_zero=False, fonts=font, clip='relu', batch_size=16)
    

    # ms-ssim
    # data_type = 'alphabet'
    # letters, labels = load_alphabet("../data/alphabet", model)
    # targets = letters.reshape((-1, 49, 53, 1))

    # lossfn = 'msssim'
    # for opt in ['sgd', 'adam']:
    #     for lr in [0.00001, 0.0001, 0.001]:
    #         nn = get_model(implant, targets[0].shape, num_dense=1)
    #         loss = train_model(nn, model, implant, None, targets, labels, 0.005, data_type, opt, lr, loss_str=lossfn)

    # new model
    model = BiphasicAxonMapModel(axlambda=1400, rho=80, a4=0, engine="jax", xystep=0.5, xrange=(-14, 12), yrange=(-12, 12))
    model.build()
    implant = ArgusII(rot=-30)
    data_type = 'alphabet'
    letters, labels = load_alphabet("../data/alphabet", model)
    targets = letters.reshape((-1, 49, 53, 1))
    nn = get_model(implant, targets[0].shape, num_dense=1, force_zero=False)
    loss = train_model(nn, model, implant, None, targets, labels, 0.0, data_type, 'adam', 0.0002, num_dense=1, force_zero=False)