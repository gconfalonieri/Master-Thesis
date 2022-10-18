import toml
import experiments
import os
import experiments_alternative
from models.utilities import get_users_arrays_eeg, get_users_arrays_eye

os.environ['TF_DETERMINISTIC_OPS'] = '1'

config = toml.load('config.toml')

experiments.utilities.init_files()
experiments.utilities.fix_seeds_unique()

c = 0

excluded_list = []
excluded_users_list = config['general']['excluded_users']

data_source = config['general']['data_source']

is_ordered = config['general']['is_ordered']


for label_array in config['path']['labels_arrays']:

    label_name = experiments.utilities.get_labels_array_string(label_array)

    if data_source == 'EYE':
        array_total = get_users_arrays_eye(is_ordered, excluded_users_list)
    if data_source == 'EEG':
        array_total = get_users_arrays_eeg(is_ordered, excluded_users_list)

    x_array = array_total[0]
    y_array = array_total[1]

    for input_array in config['path']['input_arrays']:

        input_name = experiments.utilities.get_input_array_string(input_array)

        for loss_type in config['algorithm']['loss_types']:

            for optimizer_type in config['algorithm']['optimizer_types']:

                if data_source == 'EYE':
                    c = experiments.models_iterations.iterate_target_model_eye(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name, excluded_list)
                if data_source == 'EEG':
                    c = experiments.models_iterations.iterate_target_model_eeg(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name, excluded_list)
