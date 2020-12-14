#!/usr/bin/env python3
import pandas as pd
import numpy as np
import uproot
import random
import os
import shutil
import argparse

from config import *

random.seed(30)

data_folder = root_folder + 'user.eschanet.allTrees_v2_0_2_signal_1Lbb_skim.root'
root = uproot.open(data_folder)

signal_id = []
count = 0
for n in root.keys():
        if 'nosys' not in str(n).lower():
            n = str(n)
            signal_id.append('_'.join(n.split('Wh_hbb_')[1].split('_')[0:2]))  
            
unique_name = np.unique(signal_id)

def main(direction, depth, additional_cuts):
    
    if direction == 'down':
        syst_path = numpy_sig_syst_down
    elif direction == 'up':
        syst_path = numpy_sig_syst_up
    
    signal_dict = {}
    for n in unique_name:
        for names in root.keys():
            if n in str(names):
                if 'nosys' not in str(names).lower():
                    if str(names).lower().split(';')[0].endswith(direction):
    #                     print(('_'.join(str(names).split('_')[5:])).split('__')[0])
                        if ('_'.join(str(names).split('_')[5:])).split('__')[0] in syst:
    #                         print(('_'.join(str(names).split('_')[5:])).split('__')[0])
                            try:
                                signal_dict[n].append(str(names).split(";")[0].split("'")[1])
                            except:
                                signal_dict[n] = [str(names).split(";")[0].split("'")[1]]  


    entrysteps=3000000
    tot = 0

    for folder in unique_name:
        dir = syst_path + '{}'.format(folder)
        if os.path.exists(dir):
    #         continue
            shutil.rmtree(dir)
            os.makedirs(dir)
        else:
            os.makedirs(dir)

        for name in signal_dict[folder]:

            events = uproot.open(data_folder)[name]
            array = events.lazyarray('met')

            print('lunghezza array', len(array))
            file_split = len(array)//entrysteps
            start_name_file = 0
            entrystart = start_name_file*entrysteps

            print(name)

            batches = events.iterate(columns_sig, entrystart=entrystart,
                                   entrysteps=entrysteps, 
                                       outputtype=pd.DataFrame)

            for ix in range(start_name_file, file_split+1):

                print(ix)    
                batch = next(batches)
                print('adding luminosity')
                batch['luminosity'] = 139000
                print(len(batch))

                batch = batch[batch['nLep_signal'].astype(int)==1]
                print('after signal {}'.format(len(batch)))

                batch = batch[batch['trigMatch_metTrig'].astype(int)==1]
                print('after trig {}'.format(len(batch)))

                batch = batch[((batch['nBJet30_MV2c10']>=1)&(batch['nBJet30_MV2c10']<4))]
                print('after bjet {}'.format(len(batch)))

                batch = batch[batch['met']>=220]         
                print('after met {}'.format(len(batch)))

                batch = batch[batch['mt']>=50]         
                print('after mt {}'.format(len(batch)))

                if depth == 'middle':
                    batch = batch[((batch['mbb']>=100)&(batch['mbb']<=140))]  
                    print('after mbb {}'.format(len(batch)))    

                    batch = batch[batch['mct2']>100] 
                    print('after mct2 {}'.format(len(batch)))

                if additional_cuts:

                    print('cutting below 0 and above 1000')
                    batch = batch[((batch['mct2']>=0)&(batch['mct2']<1000))]
                    batch = batch[((batch['mt']>=0)&(batch['mt']<1000))]
                    batch = batch[((batch['met']>=0)&(batch['met']<1000))]
                    batch = batch[((batch['mlb1']>=0)&(batch['mlb1']<1000))]
                    batch = batch[((batch['lep1Pt']>=0)&(batch['lep1Pt']<1000))]

                if len(batch) > 0:

                    batch['weight'] = batch['genWeight']*batch['eventWeight']*batch['pileupWeight']*\
                                     batch['leptonWeight']*batch['bTagWeight']*batch['jvtWeight']*batch['luminosity']

    #                 batch['weight'] = batch.apply(lambda row: row['genWeight']*row['eventWeight']*row['pileupWeight']*
    #                                      row['leptonWeight']*row['bTagWeight']*row['jvtWeight']*row['luminosity'], axis=1)

                    batch_fin = batch.iloc[:,:8]

                    batch_fin['weight'] = batch['weight']

                    batch_fin = batch_fin[['met', 'mt', 'mbb', 'mct2',
                    'mlb1','lep1Pt', 'nJet30', 'nBJet30_MV2c10', 'weight']]

                    tot = len(batch)
                    print('tot = {}'.format(tot))
                    print("\x1b[31m\"saving {}_{}""\x1b[0m".format(name,ix))
                    np.save(syst_path + '{}/{}.npy'.format(folder,name), batch_fin.values)
                    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='arguments for systematics cuts: define if up or down and if additional cuts are required')

    parser.add_argument('--direction', nargs="?", type = str, default = 'down', help= 'up or down')
    parser.add_argument('--depth', nargs="?", type = str, default = 'middle', help= 'depth of the cuts')
    parser.add_argument('--clean_data', nargs="?", type = bool, default = True, help= 'remove events with variable <0 and >1000')

    args = parser.parse_args()
    additional_cuts = args.clean_data
    
    main(args.direction, args.depth, additional_cuts)
