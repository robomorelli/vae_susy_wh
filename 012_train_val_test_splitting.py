#!/usr/bin/env python3
import pandas as pd
import numpy as np
import uproot
import os
import random
import argparse

from config import *
from utils import *

def main(proportions, seed, random_state):

    try:
        os.makedirs(train_val_test)
    except:
        pass

    train_val_test_split(numpy_bkg, train_val_test, proportions, seed, random_state)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='split data in train val test')

    parser.add_argument('--proportions', nargs="?", type = list, default = [0.30, 0.30, 0.40], help= 'train val test split ratio')
    parser.add_argument('--seed', nargs="?", type = list, default = 30, help= 'seed')
    parser.add_argument('--random_state', nargs="?", type = list, default = 42, help= 'random_state')

    args = parser.parse_args()

    main(args.proportions, args.seed, args.random_state)
