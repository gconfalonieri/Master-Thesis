import os

import models.utilities
from experiments.models_iterations import cross_validation_users_evaluate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import numpy as np
import toml
from sklearn.model_selection import KFold
import experiments.utilities
import models
import models.deep_learning_models as dl_models

config = toml.load('config.toml')

is_ordered = config['general']['is_ordered']

filename_results = config['path']['filename_results']
filename_performances_cross_corr = config['path']['filename_performances_cross_corr']
filename_performances_aggregate = config['path']['filename_performances_aggregate']
# filename_learning_rate = config['path']['filename_learning_rate']

data_source = config['general']['data_source']

n_split = config['algorithm']['n_kfold_splits']

is_filter = config['general']['is_filter']

excluded_users_list = config['general']['excluded_users_list']
validation_users_list = config['general']['validation_users_list']

random_seed_list = config['random_seed']['seed_list']

for seed in random_seed_list:
    experiments.utilities.fix_seeds_unique_cycle(seed)
    for i in range(0, len(excluded_users_list)):
        excluded_list = excluded_users_list[i]
        validation_list = validation_users_list[i]
        model = ''
        array_total = []
        array_validation = []
        if data_source == 'EYE':
            array_total = models.utilities.get_users_arrays_eye(is_ordered, excluded_list)
            array_validation = models.utilities.get_users_validation_arrays_eye(is_ordered, validation_list)
            model = models.deep_learning_models.get_target_model_eye_1()
            str_name = 'EYE_VALIDATION ' + str(i) + '_SEED_' + str(seed)
        if data_source == 'EEG':
            array_total = models.utilities.get_users_arrays_eeg(is_ordered, excluded_list)
            array_validation = models.utilities.get_users_validation_arrays_eeg(is_ordered, validation_list)
            model = models.deep_learning_models.get_target_model_eeg_1()
            str_name = 'EEG_VALIDATION ' + str(i) + '_SEED_' + str(seed)
        x_array = array_total[0]
        y_array = array_total[1]
        x_val_array = array_validation[0]
        y_val_array = array_validation[1]

        cross_validation_users_evaluate(0, model, x_array, y_array, x_val_array, y_val_array, str_name)