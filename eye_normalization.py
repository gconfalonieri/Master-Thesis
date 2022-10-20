import toml
import pandas as pd
from datetime import datetime

import experiments

config = toml.load('config.toml')
experiments.utilities.fix_seeds()

min_norm_value = config['preprocessing']['min_normalization']
max_norm_value = config['preprocessing']['max_normalization']


def normalize_gaze_in_range(f, df_gaze):

    diff = max_norm_value - min_norm_value

    min_fpogx = min(df_gaze['FPOGX'])
    max_fpogx = max(df_gaze['FPOGX'])
    min_fpogy = min(df_gaze['FPOGY'])
    max_fpogy = max(df_gaze['FPOGY'])
    min_rpd = min(df_gaze['RPD'])
    max_rpd = max(df_gaze['RPD'])
    min_lpd = min(df_gaze['LPD'])
    max_lpd = max(df_gaze['LPD'])

    diff_fpogx = max_fpogx - min_fpogx
    diff_fpogy = max_fpogy - min_fpogy
    diff_rpd = max_rpd - min_rpd
    diff_lpd = max_lpd - min_lpd

    for i in df_gaze.index:

        norm_fpogx = (((df_gaze['FPOGX'][i] - min_fpogx) * diff) / diff_fpogx) + min_norm_value
        norm_fpogy = (((df_gaze['FPOGY'][i] - min_fpogy) * diff) / diff_fpogy) + min_norm_value
        norm_rpd = (((df_gaze['RPD'][i] - min_rpd) * diff) / diff_rpd) + min_norm_value
        norm_lpd = (((df_gaze['LPD'][i] - min_lpd) * diff) / diff_lpd) + min_norm_value

        line = df_gaze['MEDIA_NAME'][i] + ',' + str(df_gaze['CNT'][i]) + ',' + \
               str(norm_fpogx) + ',' + str(norm_fpogy) + ',' + str(df_gaze['FPOGV'][i]) + \
               ',' + str(norm_rpd) + ',' + str(norm_lpd) + '\n'

        f.write(line)


for i in range(0, 53):
    user_id = 'USER_' + str(i+1)
    if i not in config['general']['not_valid_users']:
        df_gaze = pd.read_csv('datasets/eye-tracker/User ' + str(i) + '_all_gaze.csv')

        f = open(config['path']['sync_prefix'] + 'sync_dataset_user_' + str(i) + '.csv', 'w')
        f.write('media_name,CNT,FPOGX,FPOGY,FPOGV,RPD,LPD\n')
        normalize_gaze_in_range(f, df_gaze)
        f.close()

        print('# # # - ' + user_id + " DONE - # # #")
