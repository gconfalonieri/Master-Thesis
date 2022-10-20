from datetime import datetime
from numpy import mean, std
import numpy
import pandas as pd
import toml


def normalize_in_range(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def get_dict_start_seconds(user_id, data_type):
    path = ""

    if data_type == 'eye':
        path = "datasets/timestamps/eye-tracker/timestamp_eye_" + user_id.lower() + '.txt'
    elif data_type == 'eeg':
        path = "datasets/timestamps/eeg/timestamp_eeg_" + user_id.lower() + ".txt"

    file = open(path, "r")
    timestamp = file.readline()
    file.close()
    date = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")

    seconds = (date.hour * 3600) + (date.minute * 60) + date.second + (date.microsecond / 1000000)

    return seconds


config = toml.load('./config.toml')

min_norm_value = config['preprocessing']['min_normalization']
max_norm_value = config['preprocessing']['max_normalization']

for i in range(0, 53):
    user_id = 'USER_' + str(i+1)
    if i not in config['general']['not_valid_users']:
        df_eye = pd.read_csv('datasets/eye-tracker/User ' + str(i) + '_all_gaze.csv')
        df_eeg = pd.read_csv('datasets/eeg/eeg_user_' + str(i) + '.csv')
        media_names = df_eye.drop_duplicates('MEDIA_NAME', keep='last')['MEDIA_NAME']
        eeg_time_col = df_eeg[' time']
        total_power_channel = df_eeg[' totPwr']
        eeg_timestamps = []
        start_eye = get_dict_start_seconds(user_id, 'eye')
        start_eeg = get_dict_start_seconds(user_id, 'eeg')

        # Ottiengo timestamp per ogni campione EEG

        for x in eeg_time_col:
            time = x + start_eeg
            eeg_timestamps.append(time)

        # Calcolo il tempo impiegato per ogni domanda

        max_times = dict()

        for j in df_eye.index:
            max_times[df_eye['MEDIA_NAME'][j]] = int(df_eye[df_eye.columns[3]][j])

        # Calcolo il timestamp per finale di ogni domanda

        interval_bounds = dict()

        sum = start_eye

        for key in max_times:
            interval_bounds[key] = sum
            sum += max_times[key]

        # Creo un dizionario che associ ad ogni timestamp dell'eye-tracker, la domanda corrispondente

        all_media_name = dict()
        media_list = []

        for j in df_eye.index:
            index = int(df_eye[df_eye.columns[3]][j]) + interval_bounds[df_eye['MEDIA_NAME'][j]]
            all_media_name[index] = '' # Inizializzo

        for j in df_eye.index:
            index = int(df_eye[df_eye.columns[3]][j]) + interval_bounds[df_eye['MEDIA_NAME'][j]]
            all_media_name[index] = df_eye['MEDIA_NAME'][j] # Copio la domanda

        # Se il timestamp di EEG e eye-tracker coincidono, allora salvo la domanda

        for eeg_time in eeg_timestamps:
            for eye_time in all_media_name:
                if int(eeg_time) == int(eye_time):
                    media_list.append(all_media_name[eye_time])

        # Mi adatto alla lunghezza dell'eye-tracker (che effettivamente è quello che rimane in funzione
        # durante tutto il test)

        reduced_eeg_time = []
        reduced_delta = []
        reduced_alpha_1 = []
        reduced_alpha_2 = []
        reduced_beta1 = []
        reduced_beta2 = []
        reduced_gamma1 = []
        reduced_gamma2 = []
        reduced_theta = []
        reduced_totPwr = []

        for j in range(0, len(media_list)):
            reduced_eeg_time.append(round((eeg_timestamps[j] - start_eeg), 1))
            reduced_alpha_1.append(df_eeg[' Alpha1'][j])
            reduced_alpha_2.append(df_eeg[' Alpha2'][j])
            reduced_delta.append(df_eeg[' Delta'][j])
            reduced_beta1.append(df_eeg[' Beta1'][j])
            reduced_beta2.append(df_eeg[' Beta2'][j])
            reduced_gamma1.append(df_eeg[' Gamma1'][j])
            reduced_gamma2.append(df_eeg[' Gamma2'][j])
            reduced_theta.append(df_eeg[' Theta'][j])
            reduced_totPwr.append(total_power_channel[j])

        # save dataframe

        sync_dataframe = pd.DataFrame(
            columns=['time', 'media_name', 'delta', 'alpha1', 'alpha2', 'beta1', 'beta2',
                     'gamma1', 'gamma2', 'theta', 'totalPower'])

        sync_dataframe['time'] = reduced_eeg_time

        sync_dataframe['delta'] = normalize_in_range(reduced_delta, min_norm_value, max_norm_value)
        sync_dataframe['alpha1'] = normalize_in_range(reduced_alpha_1, min_norm_value, max_norm_value)
        sync_dataframe['alpha2'] = normalize_in_range(reduced_alpha_2, min_norm_value, max_norm_value)
        sync_dataframe['beta1'] = normalize_in_range(reduced_beta1, min_norm_value, max_norm_value)
        sync_dataframe['beta2'] = normalize_in_range(reduced_beta2, min_norm_value, max_norm_value)
        sync_dataframe['gamma1'] = normalize_in_range(reduced_gamma1, min_norm_value, max_norm_value)
        sync_dataframe['gamma2'] = normalize_in_range(reduced_gamma2, min_norm_value, max_norm_value)
        sync_dataframe['theta'] = normalize_in_range(reduced_theta, min_norm_value, max_norm_value)
        sync_dataframe['totalPower'] = normalize_in_range(reduced_totPwr, min_norm_value, max_norm_value)

        sync_dataframe['media_name'] = media_list

        # sync_dataframe.to_csv(config['path']['sync_prefix'] + 'sync_dataset_user_' + str(i) + '.csv', index=False)

        print(user_id + " DONE")