#!/usr/bin/env python3
import os

root_folder = 'root_data/'

ray_model_results = 'model_results/ray_results/'

splitted_numpy_bkg = 'splitted_numpy_bkg/'
splitted_numpy_bkg_pre = 'splitted_numpy_bkg/preselection/'
splitted_numpy_bkg_middle = 'splitted_numpy_bkg/middle/'

splitted_numpy_sig = 'splitted_numpy_sig/'
splitted_numpy_sig_pre = 'splitted_numpy_sig/preselection/'
splitted_numpy_sig_middle = 'splitted_numpy_sig/middle/'

numpy_bkg = 'numpy_data/background/'
numpy_bkg_pre = 'numpy_data/background/preselection/'
numpy_bkg_middle = 'numpy_data/background/middle/'

numpy_sig = 'numpy_data/signal/'
numpy_sig_pre = 'numpy_data/signal/preselection/'
numpy_sig_middle = 'numpy_data/signal/middle'

numpy_sig_syst_down = 'numpy_data/signal_sys/down/'
numpy_sig_syst_up = 'numpy_data/signal_sys/up/'

numpy_sig_syst_down_pre = 'numpy_data/signal_sys/down/preselection/'
numpy_sig_syst_up_pre = 'numpy_data/signal_sys/up/preselection/'

numpy_sig_syst_down_middle = 'numpy_data/signal_sys/down/middle/'
numpy_sig_syst_up_middle = 'numpy_data/signal_sys/up/middle/'

train_val_test = 'numpy_data/train_val_test/'
train_val_test_mod_dep = 'numpy_data/train_val_test/model_dependent/'
train_val_test_mod_dep_boot_30_30_40 = 'numpy_data/train_val_test/bootstrap/model_dependent/bck_sig_30_30_40_bootstrap/'

model_results_single = 'model_results/model_independent/single_train/'
model_results_multiple = 'model_results/model_independent/multiple_train/'
model_results_bump_single = 'model_results/model_independent/bump_single_train/'
model_results_bump_multiple = 'model_results/model_independent/bump_multiple_train/'

model_dep_results_single = 'model_results/model_dependent/single_train/'
model_dep_results_multiple = 'model_results/model_dependent/multiple_train/'
model_dep_results_bump_single = 'model_results/model_dependent/bump_single_train/'
model_dep_results_bump_multiple = 'model_results/model_dependent/bump_multiple_train/'

model_dep_results_boot = 'model_results/model_dependent/boot/'

dict_results_exc_reg_mod = 'dictionary_results/exclusion_region/model_independent/'
dict_results_exc_reg_mod_dep = 'dictionary_results/exclusion_region/model_dependent/'

dict_results = 'dict_results/'
json_results = 'json_results/'

plot_results = 'plot_results/'

columns = ['met', 'mt', 'mbb', 'mct2',
           'mlb1', 'nJet30', 'lep1Pt', 'nBJet30_MV2c10', 'jet1Pt',
           'trigMatch_metTrig', 'jet2Pt',
           'jet3Pt','jet4Pt', 'nLep_signal',
           'genWeight','eventWeight', 'pileupWeight',
           'leptonWeight','bTagWeight','jvtWeight']

columns_sig = ['met', 'mt', 'mbb', 'mct2',
               'mlb1', 'nJet30', 'lep1Pt', 'nBJet30_MV2c10',
               'genWeight','eventWeight', 'pileupWeight',
               'leptonWeight','bTagWeight','jvtWeight',
               'trigMatch_metTrig', 'nLep_signal']

cols = ['met', 'mt', 'mbb', 'mct2',
        'mlb1', 'lep1Pt', 'nJet30','nBJet30_MV2c10', 'weight']


syst = ['JET_JER_EffectiveNP_1',
'JET_JER_EffectiveNP_2',
'JET_JER_EffectiveNP_3',
'JET_JER_EffectiveNP_4',
'JET_JER_EffectiveNP_5',
'JET_JER_EffectiveNP_6',
'JET_JER_EffectiveNP_7',
'JET_JER_EffectiveNP_8',
'JET_JER_EffectiveNP_9',
'JET_JER_EffectiveNP_10',
'JET_JER_EffectiveNP_11',
'JET_JER_EffectiveNP_12restTerm',
'JET_JER_DataVsMC',
'JET_BJES_Response',
'JET_EffectiveNP_Mixed1',
'JET_EffectiveNP_Mixed2',
'JET_EffectiveNP_Mixed3',
'JET_EffectiveNP_Detector1',
'JET_EffectiveNP_Detector2',
'JET_EffectiveNP_Modelling1',
'JET_EffectiveNP_Modelling2',
'JET_EffectiveNP_Modelling3',
'JET_EffectiveNP_Modelling4',
'JET_EffectiveNP_Statistical1',
'JET_EffectiveNP_Statistical2',
'JET_EffectiveNP_Statistical3',
'JET_EffectiveNP_Statistical4',
'JET_EffectiveNP_Statistical5',
'JET_EffectiveNP_Statistical6',
'JET_EtaIntercalibration_Modelling',
'JET_EtaIntercalibration_NonClosure_highE',
'JET_EtaIntercalibration_NonClosure_negEta',
'JET_EtaIntercalibration_NonClosure_posEta',
'JET_EtaIntercalibration_TotalStat',
'JET_Flavor_Composition',
'JET_Flavor_Response',
'JET_Pileup_OffsetMu',
'JET_Pileup_OffsetNPV',
'JET_Pileup_PtTerm',
'JET_Pileup_RhoTopology',
'JET_PunchThrough_MC16',
'JET_SingleParticle_HighPt',
'EG_RESOLUTION_ALL',
'EG_SCALE_ALL',
'EG_SCALE_AF2',
'EL_EFF_ID_TOTAL_1NPCOR_PLUS_UNCOR',
'EL_EFF_Reco_TOTAL_1NPCOR_PLUS_UNCOR',
'EL_EFF_Iso_TOTAL_1NPCOR_PLUS_UNCOR',
'MUON_ID',
'MUON_MS',
'MUON_SCALE',
'MUON_SAGITTA_RHO',
'MUON_SAGITTA_RESBIAS',
'MUON_EFF_RECO_STAT',
'MUON_EFF_RECO_SYS',
'MUON_EFF_RECO_STAT_LOWPT',
'MUON_EFF_RECO_SYS_LOWPT',
'MUON_EFF_ISO_STAT',
'MUON_EFF_ISO_SYS',
'MUON_EFF_BADMUON_STAT',
'MUON_EFF_BADMUON_SYS',
'MUON_EFF_TTVA_STAT',
'MUON_EFF_TTVA_SYS',
'FT_EFF_B_systematics',
'FT_EFF_C_systematics',
'FT_EFF_Light_systematics',
'FT_EFF_extrapolation',
'FT_EFF_extrapolation_from_charm',
'MET_SoftTrk_Scale',
'MET_SoftTrk_ResoPara',
'MET_SoftTrk_ResoPerp',
'jvtWeightJET_JvtEfficiency',
'pileupWeight']


if not os.path.exists(dict_results):
    os.makedirs(dict_results)

if not os.path.exists(json_results):
    os.makedirs(json_results)
