# vae_susy_wh

# 010_root_to_numpy.py: convert root ntuples and and apply cuts selection on numpy data. How to run:
-) python 010_root_to_numpy.py --type bkg --depth middle --clean_data True

--type: "sig" or "bkg": convert the signals or the background data
--depth: "preselection" or "middle"
--clean_data: True or False

# 011_merge_numpy.py: merge all the backroung numpy file splitted with 010 script:
-) 011_merge_numpy.py

# 012_train_val_test_splitting.py: split in a reproducible way the data in train validation and test:
-) python 012_root_to_numpy.py --proportions [0.30,0.30,0.40] --seed 30 --random_state 42

--proportions: the train_val_test split
--seed: seed for initialization
--random_state: 42 for random_state split

# 013_signal_sys: convert and cut the systematics
-) python 013_signal_sys --direction 'down' --depth 'middle' --clean_data True

--direction: 'up' or 'down' systematic effect
--depth: "preselection" or "middle"
--clean_data: True or False

# 014_signal_injection.py: inject signal the data and split the data with [0.30,0.30,0.40] proportions
signal injected in the data:
['187_12','212_37','275_50','250_75','300_100',
'400_200','500_100','550_50','550_300', '600_50',
'650_300','750_250','800_400', '700_50', '900_50']

# 020_plot_distribution
open this jupyter to check the data distribution

# 030 vae_multiple_train: run multiple training of the same vae model to select (later) the best:

to play with the parameter of the model edit the script in the very last line where this function is called:

train_vae(name_fold=name_fold, name_weights = name_weights, dictionary=components_dict, w=weights, ind_w = individual_weights
            ,  intermediate_dim=50, act_fun='relu',latent_dim=latent_dim,kernel_max_norm=500, lr=0.003,epochs=2000, weight_KL_loss=0.6,
              batch_size=200 , cols = selected_components, num_train=5 , sig_inj = [True, '650_300']
             )
             
the editable parameters of the model start from intermediate dim (begin of second line).

# 031_multiple_model_summary: script to take a summary of the different train respect to a signal injection:
# python 031_multiple_model_summary --analysis 'model_dependent'

# 032_multiple_model_summary.ipynb: jupyter version (more stable, better way)

# 040_test_results.ipynb: jupyter notebook to check proxy performances of a model

# 041_cut_optimization.ipynb: jupyter notebook to run scan threshold optimization

# 042_reco_loss_syst_storing.ipynb: jupyter notebook to save the results in root ntuple (To Test)

# 050_exclusion_region_syst: jupyter notebook to run exclusion region analysis. Thi final output is a json file to feed
in the countour_plot script









