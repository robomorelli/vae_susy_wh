#!/usr/bin/env python3
import pandas as pd
import numpy as np
import random
import os
import pickle

from numpy.random import seed

import json
import os
import shutil

import random as rn

from config import *
from utils import *


all_signal = []
signal_name = []
all_signal_dict = {}

signal_name_path = os.listdir(numpy_sig)

for i, name in enumerate(signal_name_path):

    all_signal.append(np.load(numpy_sig + name))
    if 'p0' in name:
        signal_name.append(''.join(name.split('Wh_hbb_')[1].split('p0')[0:2]))
    else:
        signal_name.append(''.join(name.split('Wh_hbb_')[1].split('p5')[0:2]))

    all_signal_dict[signal_name[-1]] = np.load(numpy_sig + name)


to_select = ['187_12','212_37','275_50','250_75','300_100',

            '400_200','500_100','550_50','550_300', '600_50',

            '650_300','750_250','800_400', '700_50', '900_50',

            ]

bkg = np.load('numpy_data/train_val_test/background.npy')
train = np.load('numpy_data/train_val_test/background_train.npy')
val = np.load('numpy_data/train_val_test/background_val.npy')


np.random.seed(42)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(12345)
random.seed(30)

for ts in to_select:
    print(ts)
    sig = all_signal_dict[ts]
    bkg = bkg

    y_bkg = np.array([0]*len(bkg)).reshape(len(bkg),1)
    y_sig = np.array([1]*len(sig)).reshape(len(sig),1)
    y_bkg_sig = np.concatenate((y_bkg, y_sig))

    bkg_sig = np.concatenate((bkg,sig))
    bkg_sig = np.concatenate((bkg_sig, y_bkg_sig), axis=1)

    # SAVE bkg_sig but without label

    df = pd.DataFrame(bkg_sig, columns=cols+['label'])

    bkg_sig_train, bkg_sig_val_test, y_train, y_val_test = train_test_split(
         df.drop('label', axis=1), df['label'],
                test_size=0.70, random_state=123)

    # SAVE bkg_sig_train

    df_val_test = pd.DataFrame(bkg_sig_val_test, columns=cols)
    df_val_test['label'] = y_val_test

    bkg_sig_val, bkg_sig_test, y_val, y_test = train_test_split(
         bkg_sig_val_test, y_val_test,
                test_size=0.5714285714285715, random_state=123)

     # SAVE bkg_sig_val and bkg_sig_val
    df_bkg_val = pd.DataFrame(bkg_sig_val, columns=cols)
    df_bkg_test = pd.DataFrame(bkg_sig_test, columns=cols)

    df_bkg_val['label'] = y_val
    df_bkg_test['label'] = y_test

    sig_val = (df_bkg_val[df_bkg_val['label']==1]).values.copy()
    sig_test = (df_bkg_test[df_bkg_test['label']==1]).values.copy()

    bkg_val = (df_bkg_val[df_bkg_val['label']==0]).values.copy()
    bkg_test = (df_bkg_test[df_bkg_test['label']==0]).values.copy()

    dir = 'numpy_data/train_val_test/model_dependent/bkg_sig_{}_30_30_40'.format(ts)

    try:
        os.makedirs(dir)
    except:
        print('already, exist, continue...')

    np.save(dir + '/background_sig_{}.npy'.format(ts), bkg_sig[:,:-1])

    np.save(dir + '/background_sig_train_{}.npy'.format(ts), bkg_sig_train)
    np.save(dir + '/background_sig_val_{}.npy'.format(ts), bkg_sig_val)
    np.save(dir + '/background_sig_test_{}.npy'.format(ts), bkg_sig_test)

    np.save(dir + '/background_val_{}.npy'.format(ts,i), bkg_val[:,:-1])
    np.save(dir + '/background_test_{}.npy'.format(ts,i), bkg_test[:,:-1])

    np.save(dir + '/sig_val_{}.npy'.format(ts,i), sig_val[:,:-1])
    np.save(dir + '/sig_test_{}.npy'.format(ts,i), sig_test[:,:-1])
