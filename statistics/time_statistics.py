from numpy import mean, std, zeros
import pandas as pd
import toml
import experiments

config = toml.load('config.toml')
experiments.utilities.fix_seeds_unique()


def get_times_for_question_df(df_correct_answer, n_users):

    df_times_for_question = pd.DataFrame(columns=['MEDIA_NAME'])
    df_times_for_question['MEDIA_NAME'] = df_correct_answer['MEDIA_NAME']

    for i in range(1, n_users):

        if i not in config['general']['excluded_users']:

            user_id = 'USER_' + str(i)
            user_eye = 'datasets/eye-tracker/User ' + str(i) + '_all_gaze.csv'

            df_user = pd.read_csv(user_eye)
            times = dict.fromkeys(df_correct_answer['MEDIA_NAME'], 0)

            for j in range(1, 25):
                df_media = df_user[df_user['MEDIA_NAME'] == ('NewMedia' + str(j))]
                times[('NewMedia' + str(j))] = df_media.iloc[-1, 3]

            df_times_for_question[user_id] = times.values()

            print(user_id + " DONE")

    return df_times_for_question


def get_times_for_user_df(df_times_for_question, n_users):

    df_times_for_user = pd.DataFrame(columns=['USER_ID'])

    user_list = []
    index = 0

    for i in range (1, 25):
        media_id = 'NewMedia' + str(i)
        df_times_for_user[media_id] = zeros(n_users-len(config['general']['excluded_users'])-1)

    for i in range(1, n_users):

        if i not in config['general']['excluded_users']:

            user_id = 'USER_' + str(i)
            user_list.append(user_id)

            for j in range(0, 24):
                media_id = 'NewMedia' + str(j+1)
                df_times_for_user[media_id][index] = df_times_for_question[user_id][j]

            index += 1

    df_times_for_user['USER_ID'] = user_list

    return df_times_for_user

def get_statistics_times_for_questions_df(df_times_for_question, n_users):

    df_statistics_times_for_questions = pd.DataFrame(columns=['MEDIA_NAME', 'AVERAGE_TIME', 'STANDARD_DEVIATION'])
    df_statistics_times_for_questions['MEDIA_NAME'] = df_times_for_question['MEDIA_NAME']

    average_times = []
    standard_deviations = []

    for i in df_times_for_question.index:
        list_times = []
        for j in range(1, n_users):
            if j not in config['general']['excluded_users']:
                list_times.append(df_times_for_question['USER_' + str(j)][i])
        average_time = mean(list_times)
        standard_deviation = std(list_times)
        average_times.append(average_time)
        standard_deviations.append(standard_deviation)

    df_statistics_times_for_questions['AVERAGE_TIME'] = average_times
    df_statistics_times_for_questions['STANDARD_DEVIATION'] = standard_deviations

    return df_statistics_times_for_questions


def get_statistics_times_for_users_df(df_times_for_users):

    df_statistics_times_for_users = pd.DataFrame(columns=['USER_ID', 'AVERAGE_TIME', 'STANDARD_DEVIATION'])
    df_statistics_times_for_users['USER_ID'] = df_times_for_users['USER_ID']

    average_times = []
    standard_deviations = []

    for i in df_times_for_users.index:
        list_times = []
        for j in range(0, 24):
            media_id = 'NewMedia' + str(j + 1)
            list_times.append(df_times_for_users[media_id][i])
        average_time = mean(list_times)
        standard_deviation = std(list_times)
        average_times.append(average_time)
        standard_deviations.append(standard_deviation)

    df_statistics_times_for_users['AVERAGE_TIME'] = average_times
    df_statistics_times_for_users['STANDARD_DEVIATION'] = standard_deviations

    return df_statistics_times_for_users
