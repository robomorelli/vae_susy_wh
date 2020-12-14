#!/usr/bin/env python3
import numpy as np

import numpy
import pandas as pd
from pathlib import Path

import json
import os
import shutil
import argparse

from config import *



def main(analysis):
    
    if analysis == 'model_dependent':
        
        idx = 0
        
        sig_names = os.listdir(model_dep_results_multiple)
        print('the dictionaries for the following signal injection will be updated: {}'.format(sig_names))
        path_results = model_dep_results_multiple
        
        path_append =[]
        for s_n in sig_names:
            fold_names = os.listdir(path_results + '/' + s_n)
            for fold_name in fold_names:  
                fold_name_path = path_results + '/' + s_n + '/' + fold_name
                try:
                    weights_names = os.listdir(fold_name_path)
                except:
                    continue

                for i,weights_name in enumerate(weights_names):  
                    path = fold_name_path + '/' + weights_name
                    filenames = os.listdir(path)
                    names_csv = [name  for name in filenames if name.split('.')[-1]=='csv']

            #         ###CHECK HOSTNAME
            #         host = directory
                    header = True
                    total = []

                    for ix, n in enumerate(names_csv):
                        if 'summary' not in n:
                            df = pd.read_csv(path + '/' + n)

                            val_loss = df['val_loss'].min()
                            KL_loss = df['val_KL_loss'].min()
            #                 metric =  df['val_RecoNLL_metric'].min()
                            model_num = n.split('.')[0].split('_')[-1]
                            epochs = len(df)

                            total.append([val_loss,KL_loss, epochs, model_num])

                    columns = ['val_loss', 'val_KL_loss','epochs','model_num']
                    analysis = pd.DataFrame(total, columns=columns)

                    print(path, 'number', idx)
                    idx += 1
                    path_append.append(path.split('//')[1] + '/' + 'result_summary.csv')

        analysis.to_csv(path  + '/' + 'result_summary.csv')

        num = int(input("Enter a number: "))
        df = pd.read_csv(model_dep_results_multiple+path_append[num])
        print(df)
        
    elif analysis == 'model_independent':
        
        sig_names = os.listdir(model_results_multiple)
        print('the dictionaries for the following signal injection will be updated: {}'.format(sig_names))
        path_results = model_results_multiple
        fold_names = os.listdir(model_results_multiple)
        print('the dictionaries for the following models will be updated: {}'.format(fold_names))
        
        idx = 0
        path_append = []
        for fold_name in fold_names:  
            fold_name_path = model_results_multiple + '/' + fold_name
            weights_names = os.listdir(fold_name_path)

            for weights_name in weights_names:  
                path = fold_name_path + '/' + weights_name
                filenames = os.listdir(path)
                names_csv = [name  for name in filenames if name.split('.')[-1]=='csv']

        #         ###CHECK HOSTNAME
        #         host = directory
                header = True
                total = []

                for ix, n in enumerate(names_csv):
                    if 'summary' not in n:
                        df = pd.read_csv(path + '/' + n)

                        val_loss = df['val_loss'].min()
                        KL_loss = df['val_KL_loss'].min()
        #                 metric =  df['val_RecoNLL_metric'].min()
                        model_num = n.split('.')[0].split('_')[-1]
                        epochs = len(df)

                        total.append([val_loss,KL_loss, epochs, model_num])

                columns = ['val_loss', 'val_KL_loss','epochs','model_num']
                analysis = pd.DataFrame(total, columns=columns)

                print(path, 'number', idx)
                idx += 1
                path_append.append(path.split('//')[1] + '/' + 'result_summary.csv')

        analysis.to_csv(path  + '/' + 'result_summary.csv')
        
        num = int(input("Enter a number: "))
        df = pd.read_csv(model_results_multiple+path_append[num])
        print(df)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='arguments for summarize results script')

    parser.add_argument('--analysis', nargs="?", type = str, default = 'model_dependent', help= 'model_dependent or model_independent')
    args = parser.parse_args()
    
    main(args.analysis)
            