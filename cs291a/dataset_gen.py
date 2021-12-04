""" Generates percepts """
import numpy as np
import random
import h5py
import os
import json
from datetime import datetime
import argparse

import pulse2percept as p2p

def rand_stim(implant, n_electrodes=1):
    maxamp = 10
    maxfreq = 200
    # randomly pick UP TO n_electrodes
    sample_elecs = random.randint(1, n_electrodes)
    elecs = random.sample([i for i in range(len(implant.electrodes))], sample_elecs)
    stim = np.zeros((len(implant.electrodes), 3), dtype='float32')
    for elec in elecs:
        amp = random.random() * (maxamp - 1) + 1
        freq = random.random() * (maxfreq - 1) + 1
        pdur = random.expovariate(1)
        while pdur > 1000 / freq / 2 or pdur < 0.01 or pdur > 100:
            pdur = random.expovariate(1)
        stim[elec] = np.array([freq, amp, pdur])
    return stim

def rand_percepts(model, implant, n_elecs=1, n_samples=10000):
    model.build()
    x = np.array([implant[e].x for e in implant.electrodes], dtype='float32')
    y = np.array([implant[e].y for e in implant.electrodes], dtype='float32')

    percepts = []
    stims = []
    for i in range(n_samples):
        stim = rand_stim(implant, n_electrodes=n_elecs)
        percept = model._predict_spatial_jax(stim, x, y).reshape(model.grid.shape)
        stims.append(stim)
        percepts.append(percept)
    percepts = np.array(percepts)
    stims = np.array(stims)

    return percepts, stims

def write_h5(percepts, stims, path):
    if os.path.exists(path):
        print("h5 exists, overwriting")
    percepts = np.array(percepts, dtype='float32')
    stims = np.array(stims, dtype='float32')
    hf = h5py.File(path, 'w')
    hf.create_dataset('stims', data=stims)
    hf.create_dataset('percepts', data=percepts)
    hf.close()

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

def get_path(model, implant, n_electrodes):
    path = (f'percepts' 
           f'_{str(type(implant)).lower().split(".")[-1][:-2]}'
           f'_{n_electrodes}elec'
           f'_rho{model.rho}lam{model.axlambda}'
           f'_{datetime.now().strftime("%m%d%H%M")}'
           f'.h5')
    return path

def save_h5(dirname, model, implant, percepts, stims, n_electrodes):
    info_json = os.path.join(dirname, "info.json")
    if os.path.exists(info_json):
        info = json.load(open(info_json))
    else:
        info = {}
    path = get_path(model, implant, n_electrodes)
    info[path] = {}
    info[path]['model'] = str(type(model))
    info[path]['implant'] = str(type(implant))
    info[path]['n_elecs'] = n_electrodes
    info[path]['rho'] = model.rho
    info[path]['axlambda'] = model.axlambda
    info[path]['xystep'] = model.xystep
    info[path]['xrange'] = str(model.xrange)
    info[path]['yrange'] = str(model.yrange)
    info[path]['size'] = len(stims)
    info[path]['subject'] = ''
    info[path]['min_ax_sensitivity'] = model.min_ax_sensitivity
    for p in ['a' + str(i) for i in range(10)]:
        info[path][p] = getattr(model, p)
    json.dump(info, open(info_json, 'w'))

    write_h5(percepts, stims, os.path.join(dirname, path))

if __name__ == '__main__':

    n_electrodes = 15
    n_samples = 6000
    model = p2p.models.BiphasicAxonMapModel(engine='jax', a4=0, rho=200, xrange=(-14, 12), yrange=(-12, 12), xystep=0.5, axlambda=800)
    model.build()
    implant = p2p.implants.ArgusII(rot=-30)
    percepts, stims = rand_percepts(model, implant, n_elecs = n_electrodes, n_samples = n_samples)

    save_h5('/home/jgranley/cs291a/data/percepts', model, implant, percepts, stims, n_electrodes)