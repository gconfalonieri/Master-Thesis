import numpy
import toml
import pandas as pd
import numpy as np
import scipy.signal
import sklearn.utils
from sklearn.model_selection import train_test_split

import experiments

config = toml.load('config.toml')
experiments.utilities.fix_seeds_unique()


def get_max_series_len():

    max_len = 0

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['not_valid_users']:
            path = config['path']['sync_prefix_eeg'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                reduced_df = df_sync[df_sync['media_name'] == name]
                curr_len = len(reduced_df.iloc[:, 0])
                if curr_len > max_len:
                    max_len = curr_len

    return max_len


def get_max_series_len_shifted():

    max_len = 0

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['not_valid_users']:
            path = config['path']['sync_prefix_eye'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                reduced_df = df_sync[df_sync['media_name'] == name]
                for j in range(0, 15):
                    shifted = reduced_df.iloc[j::15, :]
                    curr_len = len(shifted.iloc[:, 0])
                    if curr_len > max_len:
                        max_len = curr_len
    return max_len


def get_questions_array():

    complete_x_list = []

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            path = config['path']['sync_prefix']+ 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                question_list = []
                reduced_df = df_sync[df_sync['media_name'] == name]
                for f in config['algorithm']['gaze_features']:
                    arr = np.asarray(reduced_df[f]).astype('float32')
                    question_list.append(arr)
                complete_x_list.append(question_list)

    return np.array(complete_x_list, dtype=np.ndarray)


def get_questions_oversampled_array():

    complete_x_list = []

    max_len = get_max_series_len()

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        if i not in config['general']['excluded_users']:
            path = config['path']['sync_prefix'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            for name in media_names:
                question_list = []
                reduced_df = df_sync[df_sync['media_name'] == name]
                for f in config['algorithm']['gaze_features']:
                    arr = np.asarray(reduced_df[f]).astype('float32')
                    oversampled_array = numpy.array(0)
                    if config['preprocessing']['resample_library'] == 'sklearn':
                        oversampled_array = sklearn.utils.resample(arr, n_samples=max_len, stratify=arr)
                    elif config['preprocessing']['resample_library'] == 'scipy':
                        oversampled_array = scipy.signal.resample(arr, max_len)
                    question_list.append(oversampled_array)
                complete_x_list.append(question_list)

    return np.array(complete_x_list, dtype=np.ndarray)


def get_labels_questions_array():

    complete_y_list = []

    path_labelled_df = config['path']['labelled_dataset']
    df_labelled = pd.read_csv(path_labelled_df)

    for i in df_labelled.index:
        arr = np.array(df_labelled['LABEL'][i])
        complete_y_list.append(np.expand_dims(arr, axis=(0)))

    return np.asarray(complete_y_list).astype('int')


def aggregate_questions(old_array, filter):

    new_array = []

    cnt = 0

    for x in old_array:
        cnt += 1
        if (filter and cnt not in config['general']['excluded_media']) or not filter:
            for y in x:
                new_array.append(y)
        if cnt == 24:
            cnt = 0

    return np.array(new_array).astype(np.float32)


def aggregate_users(old_array, filter):

    new_array = []

    # (41, 24, 80000)

    for user in old_array:
        cnt_question = 0
        for question in user:
            cnt_question += 1
            if (filter and cnt_question not in config['general']['excluded_media']) or not filter:
                for z in question:
                    new_array.append(z)

    return np.array(new_array).astype(np.float32)


def aggregate_users_target(old_array, filter, excluded_list):

    new_array = []

    for user in old_array:
        cnt_question = 0
        for question in user:
            cnt_question += 1
            if (filter and cnt_question not in excluded_list) or not filter:
                for z in question:
                    new_array.append(z)

    return np.array(new_array).astype(np.float32)


def aggregate_users_labels_target(old_array, filter, excluded_list):

    new_array = []

    for x in old_array:
        cnt = 0
        for y in x:
            cnt += 1
            if (filter and cnt not in excluded_list) or not filter:
                for z in y:
                    new_array.append(z)

    return np.asarray(new_array).astype('int')


def aggregate_questions_labels(old_array, filter):

    new_array = []

    cnt = 0

    for x in old_array:
        cnt += 1
        if (filter and cnt not in config['general']['excluded_media']) or not filter:
            for y in x:
                new_array.append(y)
        if cnt == 24:
            cnt = 0

    return np.asarray(new_array).astype('int')


def aggregate_users_labels(old_array, filter):

    new_array = []

    for x in old_array:
        cnt = 0
        for y in x:
            cnt += 1
            if (filter and cnt not in config['general']['excluded_media']) or not filter:
                for z in y:
                    new_array.append(z)

    return np.asarray(new_array).astype('int')


def get_users_arrays_eye(is_ordered, excluded_user_list):

    complete_x = []
    complete_y = []

    max_len = get_max_series_len_shifted()

    path_labelled_df = config['path']['labelled_dataset']
    df_labelled = pd.read_csv(path_labelled_df)

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        # config['general']['excluded_users']
        if i not in excluded_user_list:
            path = config['path']['sync_prefix_eye'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            if is_ordered:
                media_names = config['general']['media_list']
            else:
                media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            user_list = []
            user_label_list = []
            for name in media_names:
                question_list = []
                question_label_list = []
                reduced_df = df_sync[df_sync['media_name'] == name]
                red_df = df_labelled.loc[(df_labelled['USER_ID'] == user_id) & (df_labelled['MEDIA_NAME'] == name)]
                arr_label = np.array(red_df['LABEL'].values[0])
                for j in range(0, 15):
                    shifted = reduced_df.iloc[j::15, :]
                    feature_list = []
                    for f in config['algorithm']['gaze_features']:
                        arr = np.asarray(shifted[f]).astype('float32')
                        pad_len = max_len - len(arr)
                        padded_array = np.pad(arr, pad_width=(0, pad_len), mode='constant', constant_values=-10)
                        feature_list.append(padded_array)
                    question_list.append(feature_list)
                    question_label_list.append(np.expand_dims(arr_label, axis=(0)))
                user_list.append(question_list)
                user_label_list.append(question_label_list)
            complete_y.append(user_label_list)
            complete_x.append(user_list)

    return np.array(complete_x, dtype=np.ndarray), np.asarray(complete_y).astype('int')



def get_users_arrays_eeg(is_ordered, excluded_user_list):

    complete_x = []
    complete_y = []

    max_len = get_max_series_len()

    path_labelled_df = config['path']['labelled_dataset']
    df_labelled = pd.read_csv(path_labelled_df)

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        # config['general']['excluded_users']
        if i not in excluded_user_list:
            path = config['path']['sync_prefix_eeg'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            if is_ordered:
                media_names = config['general']['media_list']
            else:
                media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            user_list = []
            user_label_list = []
            for name in media_names:
                question_list = []
                question_label_list = []
                reduced_df = df_sync[df_sync['media_name'] == name]
                red_df = df_labelled.loc[(df_labelled['USER_ID'] == user_id) & (df_labelled['MEDIA_NAME'] == name)]
                arr_label = np.array(red_df['LABEL'].values[0])
                for f in config['algorithm']['eeg_features']:
                    arr = np.asarray(reduced_df[f]).astype('float32')
                    pad_len = max_len - len(arr)
                    padded_array = np.pad(arr, pad_width=(0, pad_len), mode='constant', constant_values=-10)
                    question_list.append(padded_array)
                question_label_list.append(np.expand_dims(arr_label, axis=(0)))
                user_list.append(question_list)
                user_label_list.append(question_label_list)
            complete_y.append(user_label_list)
            complete_x.append(user_list)

    return np.array(complete_x, dtype=np.ndarray), np.asarray(complete_y).astype('int')


def get_users_validation_arrays_eeg(is_ordered, validation_user_list):

    complete_x = []
    complete_y = []

    max_len = get_max_series_len()

    path_labelled_df = config['path']['labelled_dataset']
    df_labelled = pd.read_csv(path_labelled_df)

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        # config['general']['validation_users']
        if i in validation_user_list:
            path = config['path']['sync_prefix_eeg'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            if is_ordered:
                media_names = config['general']['media_list']
            else:
                media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            user_list = []
            user_label_list = []
            for name in media_names:
                question_list = []
                question_label_list = []
                reduced_df = df_sync[df_sync['media_name'] == name]
                red_df = df_labelled.loc[(df_labelled['USER_ID'] == user_id) & (df_labelled['MEDIA_NAME'] == name)]
                arr_label = np.array(red_df['LABEL'].values[0])
                for f in config['algorithm']['eeg_features']:
                    arr = np.asarray(reduced_df[f]).astype('float32')
                    pad_len = max_len - len(arr)
                    padded_array = np.pad(arr, pad_width=(0, pad_len), mode='constant', constant_values=-10)
                    question_list.append(padded_array)
                question_label_list.append(np.expand_dims(arr_label, axis=(0)))
                user_list.append(question_list)
                user_label_list.append(question_label_list)
            complete_y.append(user_label_list)
            complete_x.append(user_list)

    return np.array(complete_x, dtype=np.ndarray), np.asarray(complete_y).astype('int')


def get_users_validation_arrays_eye(is_ordered, validation_user_list):

    complete_x = []
    complete_y = []

    max_len = get_max_series_len_shifted()

    path_labelled_df = config['path']['labelled_dataset']
    df_labelled = pd.read_csv(path_labelled_df)

    for i in range(1, 53):
        user_id = 'USER_' + str(i)
        # config['general']['validation_users']
        if i in validation_user_list:
            path = config['path']['sync_prefix_eye'] + 'sync_dataset_' + user_id.lower() + '.csv'
            df_sync = pd.read_csv(path)
            if is_ordered:
                media_names = config['general']['media_list']
            else:
                media_names = df_sync.drop_duplicates('media_name', keep='last')['media_name']
            user_list = []
            user_label_list = []
            for name in media_names:
                question_list = []
                question_label_list = []
                reduced_df = df_sync[df_sync['media_name'] == name]
                red_df = df_labelled.loc[(df_labelled['USER_ID'] == user_id) & (df_labelled['MEDIA_NAME'] == name)]
                arr_label = np.array(red_df['LABEL'].values[0])
                for j in range(0, 15):
                    shifted = reduced_df.iloc[j::15, :]
                    feature_list = []
                    for f in config['algorithm']['gaze_features']:
                        arr = np.asarray(shifted[f]).astype('float32')
                        pad_len = max_len - len(arr)
                        padded_array = np.pad(arr, pad_width=(0, pad_len), mode='constant', constant_values=-10)
                        feature_list.append(padded_array)
                    question_list.append(feature_list)
                    question_label_list.append(np.expand_dims(arr_label, axis=(0)))
                user_list.append(question_list)
                user_label_list.append(question_label_list)
            complete_y.append(user_label_list)
            complete_x.append(user_list)

    return np.array(complete_x, dtype=np.ndarray), np.asarray(complete_y).astype('int')

