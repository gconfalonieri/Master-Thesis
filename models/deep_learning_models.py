import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import toml
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, LSTM, Conv2D, MaxPooling2D, Dropout, Flatten, Masking, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
import experiments

config = toml.load('config.toml')
experiments.utilities.fix_seeds_unique()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def get_model_ccn1d(dense_input, dense_input_dim, dense_input_activation, dense_output_activation,
                    n_cnn_filters, cnn_kernel_size, cnn_pool_size, loss_type, optimizer_type,
                    dropout, dropout_value):
    model = Sequential()
    model.add(Masking(mask_value=-10.0))
    if dense_input:
        model.add(Dense(dense_input_dim, activation=dense_input_activation))
    model.add(Conv1D(filters=n_cnn_filters, kernel_size=cnn_kernel_size, padding='same', activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(MaxPooling1D(pool_size=cnn_pool_size, padding='same'))
    if dropout:
        model.add(Dropout(dropout_value))
    model.add(Dense(1, dense_output_activation))
    model.compile(loss=loss_type, optimizer=optimizer, metrics=['accuracy'])
    return model


def get_model_lstm(dense_input, dense_input_dim, dense_input_activation, dense_output_activation, n_lstm_units,
                   loss_type, optimizer_type, dropout, dropout_value):
    model = Sequential()
    model.add(Masking(mask_value=-10.0))
    if dense_input:
        model.add(Dense(dense_input_dim, activation=dense_input_activation))
    model.add(LSTM(n_lstm_units))
    if dropout:
        model.add(Dropout(dropout_value))
    model.add(Dense(1, activation=dense_output_activation))
    model.compile(loss=loss_type, optimizer=optimizer, metrics=['accuracy'])
    return model


def get_model_cnn1d_lstm(dense_input, dense_input_dim,
                         dense_input_activation, dense_output_activation, n_cnn_filters, cnn_kernel_size,
                         cnn_pool_size, n_lstm_units, loss_type, optimizer_type, dropout, dropout_value):
    model = Sequential()
    model.add(Masking(mask_value=-10.0))
    if dense_input:
        model.add(Dense(dense_input_dim, activation=dense_input_activation))
    model.add(Conv1D(filters=n_cnn_filters, kernel_size=cnn_kernel_size, padding='same', activation='relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(MaxPooling1D(pool_size=cnn_pool_size, padding='same'))
    if dropout:
        model.add(Dropout(dropout_value))
    model.add(LSTM(n_lstm_units))
    model.add(Dense(1, activation=dense_output_activation))
    model.compile(loss=loss_type, optimizer=optimizer, metrics=['accuracy'])
    return model


def get_model_2x_cnn1d_lstm(dense_input, dense_input_dim,
                         dense_input_activation, dense_output_activation, n_cnn_filters_1, cnn_kernel_size_1,
                            n_cnn_filters_2, cnn_kernel_size_2, cnn_pool_size, n_lstm_units, loss_type,
                            optimizer_type, dropout, dropout_value):
    model = Sequential()
    model.add(Masking(mask_value=-10.0))
    if dense_input:
        model.add(Dense(dense_input_dim, activation=dense_input_activation))
    model.add(Conv1D(filters=n_cnn_filters_1, kernel_size=cnn_kernel_size_1, padding='same', activation='relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(MaxPooling1D(pool_size=cnn_pool_size))
    # model.add(BatchNormalization())
    model.add(Conv1D(filters=n_cnn_filters_2, kernel_size=cnn_kernel_size_2, padding='same', activation='relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(LSTM(n_lstm_units))
    if dropout:
        model.add(Dropout(dropout_value))
    model.add(Dense(1, activation=dense_output_activation))
    model.compile(loss=loss_type, optimizer=optimizer, metrics=['accuracy'])
    return model


def get_model_cnn1d_lstm_3x_dense(dense_input, dense_input_dim,
                         dense_input_activation, dense_output_activation, n_cnn_filters, cnn_kernel_size,
                         cnn_pool_size, n_lstm_units, loss_type, optimizer_type, dropout, dropout_value):
    model = Sequential()
    model.add(Masking(mask_value=-10.0))
    if dense_input:
        model.add(Dense(dense_input_dim, activation=dense_input_activation))
    model.add(Conv1D(filters=n_cnn_filters, kernel_size=cnn_kernel_size, padding='same', activation='relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(MaxPooling1D(pool_size=cnn_pool_size, padding='same'))
    model.add(LSTM(n_lstm_units))
    if dropout:
        model.add(Dropout(dropout_value))
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Dense(1, activation=dense_output_activation))
    model.compile(loss=loss_type, optimizer=optimizer, metrics=['accuracy'])
    return model


def get_model_cnn2d(dense_input, dense_input_dim, dense_input_activation, dense_output_activation,
                    n_cnn_filters, cnn_kernel_size, cnn_pool_size, loss_type, optimizer_type, dropout, dropout_value):
    model = Sequential()
    model.add(Masking(mask_value=-10.0))
    if dense_input:
        model.add(Dense(dense_input_dim, activation=dense_input_activation))
    model.add(Conv2D(filters=n_cnn_filters, kernel_size=cnn_kernel_size, padding='same', activation='relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=cnn_pool_size, padding='same'))
    if dropout:
        model.add(Dropout(dropout_value))
    model.add(Flatten())
    model.add(Dense(1, activation=dense_output_activation))
    model.compile(loss=loss_type, optimizer=optimizer, metrics=['accuracy'])
    return model


def get_model_cnn2d_lstm(dense_input, dense_input_dim, dense_input_activation, dense_output_activation,
                         n_cnn_filters, cnn_kernel_size, cnn_pool_size, n_lstm_units, loss_type, optimizer_type, dropout_value):
    model = Sequential()
    model.add(Masking(mask_value=-10.0))
    if dense_input:
        model.add(Dense(dense_input_dim, activation=dense_input_activation))
    model.add(Conv2D(filters=n_cnn_filters, kernel_size=cnn_kernel_size, padding='same', activation='relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=cnn_pool_size, padding='same'))
    model.add(Dropout(dropout_value))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(n_lstm_units))
    model.add(Dense(1, activation=dense_output_activation))
    model.compile(loss=loss_type, optimizer=optimizer, metrics=['accuracy'])
    return model

# 126,C1D-L-3xD,TIMES_ONLY_V2,MASKED,mean_squared_error,adam,0,,,relu,64,32,3,4,,,1,0.1
# 125,D-C1D-L-3xD,TIMES_ONLY_V2,MASKED,mean_squared_error,adam,1,2646,relu,relu,32,32,3,4,,,1,0.5
# 128,C1D-L-3xD,TIMES_ONLY_V2,MASKED,mean_squared_error,adam,0,,,relu,64,32,3,4,,,1,0.2
# 176,C1D-L-3xD,TIMES_ONLY_V2,MASKED,mean_squared_error,adam,0,,,relu,64,64,3,4,,,1,0.2
def get_target_model_eye_1():
    model = Sequential()
    model.add(Masking(mask_value=-10.0))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(MaxPooling1D(pool_size=4, padding='same'))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    return model


# 37,D-C1D-L,TIMES_ONLY_V2,MASKED,mean_squared_error,adam,1,265,relu,relu,32,32,4,3,,,1,0.1
# 21,D-C1D-L,TIMES_ONLY_V2,MASKED,mean_squared_error,adam,1,265,relu,relu,64,32,3,3,,,1,0.2
def get_target_model_eeg_1():
    model = Sequential()
    model.add(Masking(mask_value=-10.0))
    model.add(Dense(265, activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    return model


def get_target_model_eeg_2():
    model = Sequential()
    model.add(Masking(mask_value=-10.0))
    model.add(Dense(265, activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(MaxPooling1D(pool_size=3, padding='same'))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    return model


def get_parallel_output_model(eye_data, eeg_data, dropout_value):

    input_1 = Input(shape=(eye_data.shape[1], eye_data.shape[2])) # 2646
    mask_1 = Masking(mask_value=-10.0)(input_1)
    lstm_1 = LSTM(config['parallel_model']['lstm_1'])(mask_1)
    dropout_1 = Dropout(dropout_value)(lstm_1)
    dense_1_1 = Dense(config['parallel_model']['dense_1_1'])(dropout_1)

    input_2 = Input(shape=(eeg_data.shape[1], eeg_data.shape[2])) # 265
    mask_2 = Masking(mask_value=-10.0)(input_2)
    lstm_2 = LSTM(config['parallel_model']['lstm_2'])(mask_2)
    dropout_2 = Dropout(dropout_value)(lstm_2)
    dense_2_1 = Dense(config['parallel_model']['dense_2_1'])(dropout_2)

    concat_layer = Concatenate(-1)([dense_1_1, dense_2_1])

    dense_1 = Dense(1024)(concat_layer)
    dense_2 = Dense(512)(dense_1)
    dense_3 = Dense(256)(dense_2)
    dense_4 = Dense(128)(dense_3)
    dense_5 = Dense(64)(dense_4)
    dense_6 = Dense(32)(dense_5)
    dense_7 = Dense(16)(dense_6)

    output_layer = Dense(1, activation='relu')(dense_7)

    model = Model(inputs=[input_1, input_2], outputs=output_layer)

    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    model.summary()

    return model
