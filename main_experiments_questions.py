import toml
import experiments.models_iterations
from experiments.utilities import get_input_array_string, get_labels_array_string

from models.utilities import get_users_arrays_eye, get_users_arrays_eeg

config = toml.load('config.toml')

experiments.utilities.init_files()
experiments.utilities.fix_seeds_unique()

c = 0

excluded_users_list = config['general']['excluded_users']

data_source = config['general']['data_source']

for label_array in config['path']['labels_arrays']:

    label_name = get_labels_array_string(label_array)

    if data_source == 'EYE':
        array_total = get_users_arrays_eye(True, excluded_users_list)
    if data_source == 'EEG':
        array_total = get_users_arrays_eeg(True, excluded_users_list)

    x_array = array_total[0]
    y_array = array_total[1]

    for input_array in config['path']['input_arrays']:

        input_name = get_input_array_string(input_array)

        for loss_type in config['algorithm']['loss_types']:

            for optimizer_type in config['algorithm']['optimizer_types']:

                for excluded_list in config['general']['excluded_media_list_mean']:

                    if data_source == 'EYE':
                        c = experiments.models_iterations.iterate_target_model_eye(c, x_array, y_array, loss_type,
                                                                                   optimizer_type, label_name,
                                                                                   input_name, excluded_list)
                    if data_source == 'EEG':
                        c = experiments.models_iterations.iterate_target_model_eeg(c, x_array, y_array, loss_type,
                                                                                   optimizer_type, label_name,
                                                                                   input_name, excluded_list)