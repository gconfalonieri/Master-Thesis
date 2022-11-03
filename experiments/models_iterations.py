import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import toml
from sklearn.model_selection import KFold
import experiments.utilities
import models
import models.deep_learning_models as dl_models

config = toml.load('config.toml')

experiments.utilities.fix_seeds_unique()

filename_results = config['path']['filename_results']
filename_performances_cross_corr = config['path']['filename_performances_cross_corr']
filename_performances_aggregate = config['path']['filename_performances_aggregate']
# filename_learning_rate = config['path']['filename_learning_rate']

data_source = config['general']['data_source']

n_split = config['algorithm']['n_kfold_splits']

is_filter = config['general']['is_filter']


def filter_array(old_array, filter):

    new_array = []

    for x in old_array:
        cnt = 0
        user_array = []
        for y in x:
            cnt += 1
            if (filter and cnt not in config['general']['excluded_media']) or not filter:
                user_array.append(y)
        new_array.append(user_array)

    return np.array(new_array).astype(np.float32)


def expand_array(input_array):

    total_array = []

    for x in input_array:
        for y in x:
            total_array.append(y)

    return np.asarray(total_array).astype('float32')



def expand_array2(input_array):

    new_array = []
    total_array = []

    cnt = 0

    for x in input_array:
        cnt += 1
        new_array.append(x)
        if cnt == 15:
            total_array.append(new_array)
            new_array = []
            cnt = 0

    return np.asarray(total_array).astype('float32')


def cross_validation_users(c, model, array_x, array_y, model_name):
    cross_counter = 0
    acc_list = []
    val_acc_list = []
    loss_list = []
    val_loss_list = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for train_index, test_index in KFold(n_split).split(array_x):
        if data_source == 'EYE':
            x_train, x_test = models.utilities.aggregate_users(array_x[train_index], filter=is_filter),\
                              models.utilities.aggregate_users(array_x[test_index], filter=is_filter)
            y_train, y_test = models.utilities.aggregate_users_labels(array_y[train_index], filter=is_filter),\
                              models.utilities.aggregate_users_labels(array_y[test_index], filter=is_filter)
        if data_source == 'EEG' and not (model_name=='C2D' or model_name=='D-C2D' or model_name=='D-C2D-L' or model_name=='C2D-L'):
            x_train, x_test = models.utilities.aggregate_questions(array_x[train_index], filter=is_filter),\
                              models.utilities.aggregate_questions(array_x[test_index], filter=is_filter)
            y_train, y_test = models.utilities.aggregate_questions_labels(array_y[train_index], filter=is_filter),\
                              models.utilities.aggregate_questions_labels(array_y[test_index], filter=is_filter)
        if data_source == 'EEG' and (model_name=='C2D' or model_name=='D-C2D' or model_name=='D-C2D-L' or model_name=='C2D-L'):
            x_train, x_test = array_x[train_index], array_x[test_index]
            y_train, y_test = array_y[train_index], array_y[test_index]
            x_train = np.asarray(x_train).astype('float32')
            x_test = np.asarray(x_test).astype('float32')
        # learning_rate = K.eval(model.optimizer.lr)
        print(x_train.shape)
        history = model.fit(x_train, y_train, epochs=config['general']['n_epochs'],
                            validation_data=(x_test, y_test), shuffle=True)
        history_dict = history.history
        # experiments.utilities.write_line_learning_rate(filename_learning_rate, c, model_name, learning_rate)
        experiments.utilities.write_line_performances_cross_correlation(filename_performances_cross_corr, c, model_name,
                                                                        cross_counter,
                                                                        history_dict['accuracy'][-1],
                                                                        history_dict['val_accuracy'][-1],
                                                                        history_dict['loss'][-1],
                                                                        history_dict['val_loss'][-1])
        acc_list.append(history_dict['accuracy'][-1])
        val_acc_list.append(history_dict['val_accuracy'][-1])
        loss_list.append(history_dict['loss'][-1])
        val_loss_list.append(history_dict['val_loss'][-1])
        name = model_name + '[' + str(c) + ' - ' + str(cross_counter) + ']'
        models.plots.plot_model_loss(history_dict, name)
        models.plots.plot_model_accuracy(history_dict, name)
        cross_counter += 1
        # if cross_counter == 2:
        #    break
    experiments.utilities.write_line_performances(filename_performances_aggregate, c, model_name, np.mean(acc_list),
                                                  np.mean(val_acc_list), np.mean(loss_list), np.mean(val_loss_list))


def cross_validation_users_evaluate(c, model, array_x, array_y, x_val_array, y_val_array, model_name):
    cross_counter = 0
    acc_list = []
    test_acc_list = []
    val_acc_list = []
    loss_list = []
    test_loss_list = []
    val_loss_list = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    x_val = []
    y_val = []
    for train_index, test_index in KFold(n_split).split(array_x):
        if data_source == 'EYE':
            x_train, x_test = models.utilities.aggregate_users(array_x[train_index], filter=is_filter),\
                              models.utilities.aggregate_users(array_x[test_index], filter=is_filter)
            y_train, y_test = models.utilities.aggregate_users_labels(array_y[train_index], filter=is_filter),\
                              models.utilities.aggregate_users_labels(array_y[test_index], filter=is_filter)
            x_val = models.utilities.aggregate_users(x_val_array, filter=is_filter)
            y_val = models.utilities.aggregate_users(y_val_array, filter=is_filter)
        if data_source == 'EEG':
            x_train, x_test = models.utilities.aggregate_questions(array_x[train_index], filter=is_filter),\
                              models.utilities.aggregate_questions(array_x[test_index], filter=is_filter)
            y_train, y_test = models.utilities.aggregate_questions_labels(array_y[train_index], filter=is_filter),\
                              models.utilities.aggregate_questions_labels(array_y[test_index], filter=is_filter)
            x_val = models.utilities.aggregate_questions(x_val_array, filter=is_filter)
            y_val = models.utilities.aggregate_questions_labels(y_val_array, filter=is_filter)
            x_train = np.array(x_train).astype(np.float32)
            x_test = np.asarray(x_test).astype(np.float32)
            x_val = np.asarray(x_val).astype(np.float32)
        history = model.fit(x_train, y_train, epochs=config['general']['n_epochs'],
                            validation_data=(x_test, y_test), shuffle=True)
        history_dict = history.history
        evaluate_loss, evaluate_acc = model.evaluate(x_val, y_val)
        # experiments.utilities.write_line_learning_rate(filename_learning_rate, c, model_name, learning_rate)
        experiments.utilities.write_line_validation_cross(filename_performances_cross_corr, c, model_name,
                                                                        cross_counter,
                                                                        history_dict['accuracy'][-1],
                                                                        history_dict['val_accuracy'][-1],
                                                                        evaluate_acc,
                                                                        history_dict['loss'][-1],
                                                                        history_dict['val_loss'][-1],
                                                          evaluate_loss)
        acc_list.append(history_dict['accuracy'][-1])
        test_acc_list.append(history_dict['val_accuracy'][-1])
        val_acc_list.append(evaluate_acc)
        loss_list.append(history_dict['loss'][-1])
        test_loss_list.append(history_dict['val_loss'][-1])
        val_loss_list.append(evaluate_loss)
        name = model_name + '[' + str(c) + ' - ' + str(cross_counter) + ']'
        models.plots.plot_model_loss(history_dict, name)
        models.plots.plot_model_accuracy(history_dict, name)
        cross_counter += 1
        # if cross_counter == 2:
        #    break
    experiments.utilities.write_line_validation_aggr(filename_performances_aggregate, c, model_name, np.mean(acc_list),
                                                np.mean(test_acc_list), np.mean(val_acc_list), np.mean(loss_list),
                                                np.mean(test_loss_list), np.mean(val_loss_list))


def iterate_cnn1d(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name):

    for dense_output_activation in config['algorithm']['output_activation_types']:
        for n_cnn_filters in config['algorithm']['n_cnn_filters']:
            for cnn_kernel_size in config['algorithm']['cnn_kernel_size']:
                for cnn_pool_size in config['algorithm']['cnn_pool_size']:
                    for dropout_value in config['algorithm']['dropout_value']:
                        for dense_input in config['general']['binary_value']:
                            if dense_input:
                                for dense_input_dim in config['algorithm']['dense_input_dim']:
                                    for dense_input_activation in config['algorithm']['input_activation_types']:

                                        model_name = 'D-C1D'

                                        model = dl_models.get_model_ccn1d(dense_input, dense_input_dim,
                                                                          dense_input_activation,
                                                                          dense_output_activation,
                                                                          n_cnn_filters,
                                                                          cnn_kernel_size, cnn_pool_size,
                                                                          loss_type,
                                                                          optimizer_type, 1, dropout_value)

                                        experiments.utilities.write_line_results(filename_results, c, model_name,
                                                                                 label_name,
                                                                                 input_name,
                                                                                 loss_type, optimizer_type,
                                                                                 dense_input,
                                                                                 dense_input_dim,
                                                                                 dense_input_activation,
                                                                                 dense_output_activation, '',
                                                                                 n_cnn_filters,
                                                                                 cnn_kernel_size, cnn_pool_size, 1,
                                                                                 '', '', dropout_value)

                                        cross_validation_users(c, model, x_array, y_array, model_name)

                                        # model.save('tf_models/test_model_' + str(c))

                                        c += 1
                            else:

                                model_name = 'C1D'

                                model = dl_models.get_model_ccn1d(dense_input, 0, '',
                                                                  dense_output_activation, n_cnn_filters,
                                                                  cnn_kernel_size, cnn_pool_size, loss_type,
                                                                  optimizer_type, 1, dropout_value)

                                experiments.utilities.write_line_results(filename_results, c, model_name, label_name,
                                                                         input_name,
                                                                         loss_type, optimizer_type, dense_input,
                                                                         '',
                                                                         '',
                                                                         dense_output_activation, '',
                                                                         n_cnn_filters,
                                                                         cnn_kernel_size, cnn_pool_size, 1,
                                                                         '', '', dropout_value)

                                cross_validation_users(c, model, x_array, y_array, model_name)

                                c += 1

    return c


def iterate_lstm(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name):

    for dense_output_activation in config['algorithm']['output_activation_types']:
        for n_lstm_units in config['algorithm']['n_lstm_units']:
            for dropout_value in config['algorithm']['dropout_value']:
                for dense_input in config['general']['binary_value']:

                    if dense_input:
                        for dense_input_dim in config['algorithm']['dense_input_dim']:
                            for dense_input_activation in config['algorithm']['input_activation_types']:

                                model_name = 'D-L'

                                model = dl_models.get_model_lstm(dense_input, dense_input_dim,
                                                                 dense_input_activation,
                                                                 dense_output_activation,
                                                                 n_lstm_units, loss_type,
                                                                 optimizer_type, 1, dropout_value)

                                experiments.utilities.write_line_results(filename_results, c, model_name, label_name,
                                                                         input_name, loss_type,
                                                                         optimizer_type, dense_input, dense_input_dim,
                                                                         dense_input_activation,
                                                                         dense_output_activation, n_lstm_units, '',
                                                                         '', '', '', '', 1,
                                                                         dropout_value)

                                cross_validation_users(c, model, x_array, y_array, model_name)

                                # model.save('tf_models/test_model_' + str(c))

                                c += 1
                    else:

                        model_name = 'L'

                        model = dl_models.get_model_lstm(dense_input, 0, '', dense_output_activation, n_lstm_units,
                                                         loss_type, optimizer_type, 1, dropout_value)

                        experiments.utilities.write_line_results(filename_results, c, 'LSTM', label_name, input_name,
                                                                 loss_type, optimizer_type, dense_input, '', '',
                                                                 dense_output_activation, n_lstm_units, '', '', '',
                                                                 '', '', 1, dropout_value)

                        cross_validation_users(c, model, x_array, y_array, model_name)

                        # model.save('tf_models/test_model_' + str(c))

                        c += 1

    return c


def iterate_cnn1d_lstm(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name):

    for dense_output_activation in config['algorithm']['output_activation_types']:
        for n_cnn_filters in config['algorithm']['n_cnn_filters']:
            for cnn_kernel_size in config['algorithm']['cnn_kernel_size']:
                for cnn_pool_size in config['algorithm']['cnn_pool_size']:
                    for n_lstm_units in config['algorithm']['n_lstm_units']:
                        for dropout_value in config['algorithm']['dropout_value']:
                            for dense_input in config['general']['binary_value']:

                                if dense_input:
                                    for dense_input_dim in config['algorithm']['dense_input_dim']:
                                        for dense_input_activation in config['algorithm']['input_activation_types']:

                                            model_name = 'D-C1D-L'

                                            model = dl_models.get_model_cnn1d_lstm(dense_input,
                                                                                   dense_input_dim,
                                                                                   dense_input_activation,
                                                                                   dense_output_activation,
                                                                                   n_cnn_filters,
                                                                                   cnn_kernel_size,
                                                                                   cnn_pool_size,
                                                                                   n_lstm_units, loss_type,
                                                                                   optimizer_type, 1,
                                                                                   dropout_value)

                                            experiments.utilities.write_line_results(filename_results, c, model_name,
                                                                                     label_name,
                                                                                     input_name,
                                                                                     loss_type,
                                                                                     optimizer_type, dense_input,
                                                                                     dense_input_dim,
                                                                                     dense_input_activation,
                                                                                     dense_output_activation,
                                                                                     n_lstm_units,
                                                                                     n_cnn_filters,
                                                                                     cnn_kernel_size, cnn_pool_size,
                                                                                     '', '', 1, dropout_value)

                                            cross_validation_users(c, model, x_array, y_array, model_name)

                                            # model.save('tf_models/test_model_' + str(c))

                                            c += 1
                                else:

                                    model_name = 'C1D-L'

                                    model = dl_models.get_model_cnn1d_lstm(dense_input, 0, '',
                                                                           dense_output_activation,
                                                                           n_cnn_filters,
                                                                           cnn_kernel_size,
                                                                           cnn_pool_size,
                                                                           n_lstm_units,
                                                                           loss_type,
                                                                           optimizer_type, 1,
                                                                           dropout_value)

                                    experiments.utilities.write_line_results(filename_results, c, model_name,
                                                                             label_name,
                                                                             input_name,
                                                                             loss_type,
                                                                             optimizer_type, dense_input,
                                                                             '', '',
                                                                             dense_output_activation, n_lstm_units,
                                                                             n_cnn_filters,
                                                                             cnn_kernel_size, cnn_pool_size,
                                                                             '', '', 1,
                                                                             dropout_value)

                                    cross_validation_users(c, model, x_array, y_array, model_name)

                                    # model.save('tf_models/test_model_' + str(c))

                                    c += 1
    return c


def iterate_cnn1d_lstm_3dense(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name):

    for dense_output_activation in config['algorithm']['output_activation_types']:
        for n_cnn_filters in config['algorithm']['n_cnn_filters']:
            for cnn_kernel_size in config['algorithm']['cnn_kernel_size']:
                for cnn_pool_size in config['algorithm']['cnn_pool_size']:
                    for n_lstm_units in config['algorithm']['n_lstm_units']:
                        for dropout_value in config['algorithm']['dropout_value']:
                            for dense_input in config['general']['binary_value']:

                                if dense_input:
                                    for dense_input_dim in config['algorithm']['dense_input_dim']:
                                        for dense_input_activation in config['algorithm']['input_activation_types']:

                                            model_name = 'D-C1D-L-3xD'

                                            model = dl_models.get_model_cnn1d_lstm_3x_dense(dense_input,
                                                                                            dense_input_dim,
                                                                                            dense_input_activation,
                                                                                            dense_output_activation,
                                                                                            n_cnn_filters,
                                                                                            cnn_kernel_size,
                                                                                            cnn_pool_size,
                                                                                            n_lstm_units, loss_type,
                                                                                            optimizer_type, 1,
                                                                                            dropout_value)

                                            experiments.utilities.write_line_results(filename_results,
                                                                                     c, model_name, label_name,
                                                                                     input_name,
                                                                                     loss_type,
                                                                                     optimizer_type, dense_input,
                                                                                     dense_input_dim,
                                                                                     dense_input_activation,
                                                                                     dense_output_activation,
                                                                                     n_lstm_units,
                                                                                     n_cnn_filters,
                                                                                     cnn_kernel_size, cnn_pool_size,
                                                                                     '', '', 1, dropout_value)

                                            cross_validation_users(c, model, x_array, y_array, model_name)

                                            # model.save('tf_models/test_model_' + str(c))

                                            c += 1
                                else:

                                    model_name = 'C1D-L-3xD'

                                    model = dl_models.get_model_cnn1d_lstm_3x_dense(dense_input, 0, '',
                                                                                    dense_output_activation,
                                                                                    n_cnn_filters,
                                                                                    cnn_kernel_size,
                                                                                    cnn_pool_size,
                                                                                    n_lstm_units,
                                                                                    loss_type,
                                                                                    optimizer_type, 1,
                                                                                    dropout_value)

                                    experiments.utilities.write_line_results(filename_results,
                                                                             c, model_name, label_name,
                                                                             input_name,
                                                                             loss_type,
                                                                             optimizer_type, dense_input,
                                                                             '', '',
                                                                             dense_output_activation, n_lstm_units,
                                                                             n_cnn_filters,
                                                                             cnn_kernel_size, cnn_pool_size,
                                                                             '', '', 1,
                                                                             dropout_value)

                                    cross_validation_users(c, model, x_array, y_array, model_name)

                                    # model.save('tf_models/test_model_' + str(c))

                                    c += 1
    return c


def iterate_2xcnn1d_lstm(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name):

    for dense_output_activation in config['algorithm']['output_activation_types']:
        for n_cnn_filters_1 in config['algorithm']['n_cnn_filters']:
            for cnn_kernel_size_1 in config['algorithm']['cnn_kernel_size']:
                for n_cnn_filters_2 in config['algorithm']['n_cnn_filters_2']:
                    for cnn_kernel_size_2 in config['algorithm']['cnn_kernel_size_2']:
                        for cnn_pool_size in config['algorithm']['cnn_pool_size']:
                            for n_lstm_units in config['algorithm']['n_lstm_units']:
                                for dropout_value in config['algorithm']['dropout_value']:
                                    for dense_input in config['general']['binary_value']:

                                        if dense_input:
                                            for dense_input_dim in config['algorithm']['dense_input_dim']:
                                                for dense_input_activation in config['algorithm']['input_activation_types']:

                                                    model_name = 'D-2xC1D-L'

                                                    model = dl_models.get_model_2x_cnn1d_lstm(dense_input,
                                                                                              dense_input_dim,
                                                                                              dense_input_activation,
                                                                                              dense_output_activation,
                                                                                              n_cnn_filters_1,
                                                                                              cnn_kernel_size_1,
                                                                                              n_cnn_filters_2,
                                                                                              cnn_kernel_size_2,
                                                                                              cnn_pool_size,
                                                                                              n_lstm_units, loss_type,
                                                                                              optimizer_type, 1,
                                                                                              dropout_value)

                                                    experiments.utilities.write_line_results(filename_results, c, model_name,
                                                                                             label_name,
                                                                                             input_name,
                                                                                             loss_type,
                                                                                             optimizer_type, dense_input,
                                                                                             dense_input_dim,
                                                                                             dense_input_activation,
                                                                                             dense_output_activation,
                                                                                             n_lstm_units,
                                                                                             n_cnn_filters_1,
                                                                                             cnn_kernel_size_1, cnn_pool_size,
                                                                                             n_cnn_filters_2, cnn_kernel_size_2,
                                                                                             1, dropout_value)

                                                    cross_validation_users(c, model, x_array, y_array, model_name)

                                                    # model.save('tf_models/test_model_' + str(c))

                                                    c += 1
                                        else:

                                            model_name = '2xC1D-L'

                                            model = dl_models.get_model_2x_cnn1d_lstm(dense_input, 0, '',
                                                                                      dense_output_activation,
                                                                                      n_cnn_filters_1,
                                                                                      cnn_kernel_size_1,
                                                                                      n_cnn_filters_2,
                                                                                      cnn_kernel_size_2,
                                                                                      cnn_pool_size,
                                                                                      n_lstm_units,
                                                                                      loss_type,
                                                                                      optimizer_type, 1,
                                                                                      dropout_value)

                                            experiments.utilities.write_line_results(filename_results, c, model_name,
                                                                                     label_name,
                                                                                     input_name,
                                                                                     loss_type,
                                                                                     optimizer_type, dense_input,
                                                                                     '', '',
                                                                                     dense_output_activation, n_lstm_units,
                                                                                     n_cnn_filters_1,
                                                                                     cnn_kernel_size_1, cnn_pool_size,
                                                                                     n_cnn_filters_2, cnn_kernel_size_2,
                                                                                     1, dropout_value)

                                            cross_validation_users(c, model, x_array, y_array, model_name)

                                            # model.save('tf_models/test_model_' + str(c))

                                            c += 1
    return c


def iterate_cnn2d(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name):

    if data_source == 'EYE':
        x_array = np.expand_dims(x_array, 2)
        y_array = np.expand_dims(y_array, 2)

    for dense_output_activation in config['algorithm']['output_activation_types']:
        for n_cnn_filters in config['algorithm']['n_cnn_filters']:
            for cnn_kernel_size in config['algorithm']['cnn_kernel_size']:
                for cnn_pool_size in config['algorithm']['cnn_pool_size']:
                    for dropout_value in config['algorithm']['dropout_value']:
                        for dense_input in config['general']['binary_value']:

                            if dense_input:
                                for dense_input_dim in config['algorithm']['dense_input_dim']:
                                    for dense_input_activation in config['algorithm']['input_activation_types']:

                                        model_name = 'D-C2D'

                                        model = dl_models.get_model_cnn2d(dense_input,
                                                                          dense_input_dim,
                                                                          dense_input_activation,
                                                                          dense_output_activation,
                                                                          n_cnn_filters,
                                                                          cnn_kernel_size, cnn_pool_size,
                                                                          loss_type,
                                                                          optimizer_type, 1, dropout_value)

                                        experiments.utilities.write_line_results(filename_results, c, model_name,
                                                                                 label_name, input_name,
                                                                                 loss_type,
                                                                                 optimizer_type, dense_input,
                                                                                 dense_input_dim,
                                                                                 dense_input_activation,
                                                                                 dense_output_activation, '',
                                                                                 n_cnn_filters,
                                                                                 cnn_kernel_size, cnn_pool_size, '', '',
                                                                                 1, dropout_value)

                                        cross_validation_users(c, model, x_array, y_array, model_name)

                                        # model.save('tf_models/test_model_' + str(c))

                                        c += 1
                            else:

                                model_name = 'C2D'

                                model = dl_models.get_model_cnn2d(dense_input, 0, '', dense_output_activation,
                                                                  n_cnn_filters, cnn_kernel_size, cnn_pool_size,
                                                                  loss_type, optimizer_type, 1, dropout_value)

                                experiments.utilities.write_line_results(filename_results, c, model_name, label_name,
                                                                         input_name, loss_type, optimizer_type,
                                                                         dense_input,
                                                                         '', '', dense_output_activation, '',
                                                                         n_cnn_filters, cnn_kernel_size,
                                                                         cnn_pool_size, '', '', 1, dropout_value)

                                cross_validation_users(c, model, x_array, y_array, model_name)

                                # model.save('tf_models/test_model_' + str(c))

                                c += 1

    return c


def iterate_cnn2d_lstm(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name):

    if data_source == 'EYE':
        x_array = np.expand_dims(x_array, 2)
        y_array = np.expand_dims(y_array, 2)

    print(x_array.shape)

    for dense_output_activation in config['algorithm']['output_activation_types']:
        for n_cnn_filters in config['algorithm']['n_cnn_filters']:
            for cnn_kernel_size in config['algorithm']['cnn_kernel_size']:
                for cnn_pool_size in config['algorithm']['cnn_pool_size']:
                    for n_lstm_units in config['algorithm']['n_lstm_units']:
                        for dropout_value in config['algorithm']['dropout_value']:
                            for dense_input in config['general']['binary_value']:

                                if dense_input:
                                    for dense_input_dim in config['algorithm']['dense_input_dim']:
                                        for dense_input_activation in config['algorithm']['input_activation_types']:

                                            model_name = 'D-C2D-L'

                                            model = dl_models.get_model_cnn2d_lstm(dense_input,
                                                                                   dense_input_dim,
                                                                                   dense_input_activation,
                                                                                   dense_output_activation,
                                                                                   n_cnn_filters,
                                                                                   cnn_kernel_size,
                                                                                   cnn_pool_size,
                                                                                   n_lstm_units, loss_type,
                                                                                   optimizer_type,
                                                                                   dropout_value)

                                            experiments.utilities.write_line_results(filename_results, c, model_name,
                                                                                     label_name,
                                                                                     input_name,
                                                                                     loss_type,
                                                                                     optimizer_type, dense_input,
                                                                                     dense_input_dim,
                                                                                     dense_input_activation,
                                                                                     dense_output_activation,
                                                                                     n_lstm_units,
                                                                                     n_cnn_filters,
                                                                                     cnn_kernel_size, cnn_pool_size, '',
                                                                                     '', 1,
                                                                                     dropout_value)

                                            cross_validation_users(c, model, x_array, y_array, model_name)

                                            # model.save('tf_models/test_model_' + str(c))

                                            c += 1
                                else:

                                    model_name = 'C2D-L'

                                    model = dl_models.get_model_cnn2d_lstm(dense_input, 0, n_lstm_units,
                                                                           dense_output_activation,
                                                                           n_cnn_filters, cnn_kernel_size,
                                                                           cnn_pool_size,
                                                                           n_lstm_units, loss_type,
                                                                           optimizer_type, dropout_value)

                                    experiments.utilities.write_line_results(filename_results, c, 'CNN2D_LSTM',
                                                                             label_name, input_name,
                                                                             loss_type,
                                                                             optimizer_type, dense_input,
                                                                             '', '',
                                                                             dense_output_activation, n_lstm_units,
                                                                             n_cnn_filters,
                                                                             cnn_kernel_size, cnn_pool_size, '', '', 1,
                                                                             dropout_value)

                                    cross_validation_users(c, model, x_array, y_array, model_name)

                                    # model.save('tf_models/test_model_' + str(c))

                                    c += 1

    return c


def cross_validation_users_target(c, model, array_x, array_y, model_name, excluded_list):
    cross_counter = 0
    acc_list = []
    val_acc_list = []
    loss_list = []
    val_loss_list = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for train_index, test_index in KFold(n_split).split(array_x):
        if data_source == 'EYE':
            x_train, x_test = models.utilities.aggregate_users_target(array_x[train_index], is_filter, excluded_list),\
                              models.utilities.aggregate_users_target(array_x[test_index], is_filter, excluded_list)
            y_train, y_test = models.utilities.aggregate_users_labels_target(array_y[train_index], is_filter, excluded_list),\
                              models.utilities.aggregate_users_labels_target(array_y[test_index], is_filter, excluded_list)
        if data_source == 'EEG':
            x_train, x_test = models.utilities.aggregate_questions(array_x[train_index], filter=is_filter),\
                              models.utilities.aggregate_questions(array_x[test_index], filter=is_filter)
            y_train, y_test = models.utilities.aggregate_questions_labels(array_y[train_index], filter=is_filter),\
                              models.utilities.aggregate_questions_labels(array_y[test_index], filter=is_filter)
        # learning_rate = K.eval(model.optimizer.lr)
        history = model.fit(x_train, y_train, epochs=config['general']['n_epochs'],
                            validation_data=(x_test, y_test), shuffle=True)
        history_dict = history.history
        # experiments.utilities.write_line_learning_rate(filename_learning_rate, c, model_name, learning_rate)
        experiments.utilities.write_line_performances_cross_correlation(filename_performances_cross_corr, c, model_name,
                                                                        cross_counter,
                                                                        history_dict['accuracy'][-1],
                                                                        history_dict['val_accuracy'][-1],
                                                                        history_dict['loss'][-1],
                                                                        history_dict['val_loss'][-1])
        acc_list.append(history_dict['accuracy'][-1])
        val_acc_list.append(history_dict['val_accuracy'][-1])
        loss_list.append(history_dict['loss'][-1])
        val_loss_list.append(history_dict['val_loss'][-1])
        name = model_name + '[' + str(c) + ' - ' + str(cross_counter) + ']'
        models.plots.plot_model_loss(history_dict, name)
        models.plots.plot_model_accuracy(history_dict, name)
        cross_counter += 1
        # if cross_counter == 2:
        #    break
    experiments.utilities.write_line_performances(filename_performances_aggregate, c, model_name, np.mean(acc_list),
                                                  np.mean(val_acc_list), np.mean(loss_list), np.mean(val_loss_list))


def iterate_target_model_eye(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name, excluded_list):

    if data_source == 'EEG':
        x_array = models.utilities.aggregate_questions(x_array, False)
        y_array = models.utilities.aggregate_questions_labels(y_array, False)

    model_name = 'TARGET_EYE'

    model = dl_models.get_target_model_eye_1()

    experiments.utilities.write_line_results(filename_results, c, model_name, label_name, input_name,
                                             loss_type, optimizer_type, 0, '', '', 'relu', 64,
                                             32, 3, 4, 1, '', '', 0.1)

    cross_validation_users_target(c, model, x_array, y_array, model_name, excluded_list)

    # model.save('tf_models/test_model_' + str(c))

    c += 1

    return c


def iterate_target_model_eeg(c, x_array, y_array, loss_type, optimizer_type, label_name, input_name, excluded_list):

    model_name = 'TARGET_EEG'

    model = dl_models.get_target_model_eeg_1()

    experiments.utilities.write_line_results(filename_results, c, model_name, label_name, input_name,
                                             loss_type, optimizer_type, 1, 'relu', '', 'relu', 64,
                                             32, 4, 3, 1, '', '', 0.2)

    cross_validation_users_target(c, model, x_array, y_array, model_name, excluded_list)

    # model.save('tf_models/test_model_' + str(c))

    c += 1

    return c