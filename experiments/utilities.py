import os
import random
import tensorflow as tf
import numpy as np
import toml

config = toml.load('config.toml')

filename_results = config['path']['filename_results']
filename_performances_cross_corr = config['path']['filename_performances_cross_corr']
filename_performances_aggregate = config['path']['filename_performances_aggregate']
filename_performances_validation = config['path']['filename_performances_validation']
filename_learning_rate = config['path']['filename_learning_rate']

filename_validation_cross_corr = config['path']['filename_validation_cross_corr']
filename_validation_aggregate = config['path']['filename_validation_aggr']

def get_input_array_string(input_array):
    name = ''
    if input_array == 'datasets/arrays/undersampled/input_1_1_oversampled.npy' or input_array == 'datasets/arrays/undersampled_shifted/input_1_1_oversampled.npy':
        name = '-1_1_OVERSAMPLED'
    elif input_array == 'datasets/arrays/undersampled/input_1_1_padded_begin.npy':
        name = '-1_1_PADDED_BEGIN'
    elif input_array == 'datasets/arrays/undersampled/input_1_1_padded_end.npy':
        name = '-1_1_PADDED_END'
    elif input_array == 'datasets/arrays/undersampled/input_0_1_oversampled.npy':
        name = '0_1_OVERSAMPLED'
    elif input_array ==  'datasets/arrays/undersampled/input_0_1_padded_begin.npy':
        name = '0_1_PADDED_BEGIN'
    elif input_array == 'datasets/arrays/undersampled/input_0_1_padded_end.npy':
        name = '0_1_PADDED_END'
    else:
        name = 'MASKED'

    return name


def get_labels_array_string(labels_array):
    name = ''
    if labels_array == 'datasets/arrays/labels/users_labels_v2.npy':
        name = 'TIMES_ONLY_V2'
    elif labels_array == 'datasets/arrays/labels/users_labels_v2_2F.npy':
        name = 'TIMES_ONLY_V2_2F'

    return name


def fix_seeds():
    os.environ['PYTHONHASHSEED'] = str(config['random_seed']['pythonhashseed'])
    random.seed(config['random_seed']['python_seed'])
    np.random.seed(config['random_seed']['numpy_seed'])
    tf.random.set_seed(config['random_seed']['tf_seed'])

def fix_seeds_unique():
    os.environ['PYTHONHASHSEED'] = str(config['random_seed']['unique_seed'])
    random.seed(config['random_seed']['unique_seed'])
    np.random.seed(config['random_seed']['unique_seed'])
    tf.random.set_seed(config['random_seed']['unique_seed'])


def fix_seeds_unique_cycle(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def init_files():

    f = open(filename_results, 'w')
    f.write('index,model,label,input,loss_type,optimizer_type,dense_input,dense_input_dim,dense_input_activation,'
            'dense_output_activation,lstm_units,cnn_fiters_1,cnn_kernel_1,cnn_pool_size,cnn_fiters_2,cnn_kernel_2,'
            'dropout,dropout_value\n')
    f.close()

    f = open(filename_performances_cross_corr, 'w')
    f.write('index,model,cross_step,acc,val_acc,loss,val_loss\n')
    f.close()

    f = open(filename_performances_aggregate, 'w')
    f.write('index,model,acc,val_acc,loss,val_loss\n')
    f.close()

    f = open(filename_validation_cross_corr, 'w')
    f.write('index,model,cross_step,acc,test_acc,val_acc,loss,test_loss,acc_loss\n')
    f.close()

    f = open(filename_validation_aggregate, 'w')
    f.write('index,model,acc,val_acc,loss,val_loss\n')
    f.write('index,model,acc,test_acc,val_acc,loss,test_loss,acc_loss\n')
    f.close()

    # f = open(filename_learning_rate, 'w')
    # f.write('index,model,learning_rate\n')
    # f.close()


def init_validation_file():

    f = open(filename_performances_validation, 'w')
    f.write('index,acc,loss\n')
    f.close()


def write_line_results(filename, c, model, label_name, input_name, loss_type, optimizer_type, dense_input, dense_input_dim,
               dense_input_activation, dense_output_activation, lstm_cells, n_cnn_filters_1, cnn_kernel_size_1,
               cnn_pool_size, n_cnn_filters_2, cnn_kernel_size_2, dropout, dropout_value):
    f = open(filename, 'a')
    line = str(c) + ',' + model + ',' + label_name + ',' + input_name \
           + ',' + loss_type + ',' + optimizer_type + ',' + str(dense_input) \
           + ',' + str(dense_input_dim) + ',' + dense_input_activation + ',' \
           + dense_output_activation + ',' + str(lstm_cells) + ',' + str(n_cnn_filters_1) \
           + ',' + str(cnn_kernel_size_1) + ',' + str(cnn_pool_size) + ',' + str(n_cnn_filters_2) \
           + ',' + str(cnn_kernel_size_2) + ',' + str(dropout) + ',' + str(dropout_value) + '\n'
    f.write(line)
    f.close()


def write_line_performances(filename, c, model, acc, val_acc, loss, val_loss):
    f1 = open(filename, 'a')
    line = str(c) + ',' + model + ',' + str(acc) + ',' + str(val_acc) + ',' + str(loss) + ',' + str(val_loss) + '\n'
    f1.write(line)
    f1.close()


def write_line_performances_cross_correlation(filename, c, model, cross_counter, acc, val_acc, loss, val_loss):
    f1 = open(filename, 'a')
    line = str(c) + ',' + model + ',' + str(cross_counter) + ',' + str(acc) + ',' + str(val_acc) + ',' + str(loss) + ',' + str(val_loss) + '\n'
    f1.write(line)
    f1.close()


def write_line_validation_cross(filename, c, model, cross_counter, acc, train_acc, val_acc, loss, train_loss, val_loss):
    f1 = open(filename, 'a')
    line = str(c) + ',' + model + ',' + str(cross_counter) + ',' + str(acc) + ',' + str(train_acc) + ',' + str(val_acc) \
           + ','+ str(loss) + ',' + str(train_loss) + ',' + str(val_loss) + '\n'
    f1.write(line)
    f1.close()


def write_line_validation_aggr(filename, c, model, acc, train_acc, val_acc, loss, train_loss, val_loss):
    f1 = open(filename, 'a')
    line = str(c) + ',' + model + ',' + str(acc) + ',' + str(train_acc) + ',' + str(val_acc) \
           + ','+ str(loss) + ',' + str(train_loss) + ',' + str(val_loss) + '\n'
    f1.write(line)
    f1.close()


def write_line_learning_rate(filename, c, model, learning_rate):
    f1 = open(filename, 'a')
    line = str(c) + ',' + model + ',' + str(learning_rate) + '\n'
    f1.write(line)
    f1.close()
