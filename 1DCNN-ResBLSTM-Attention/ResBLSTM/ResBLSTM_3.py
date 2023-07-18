import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import BatchNormalization, Dropout, LSTM, Dense, Activation, Concatenate, Bidirectional,\
     Add, LayerNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import json


root_train = 'D:/human activity recognition/UCI-HAR/UCI HAR Dataset/train/Inertial Signals/'
root_test = 'D:/human activity recognition/UCI-HAR/UCI HAR Dataset/test/Inertial Signals/'
model_save_path = 'D:/human activity recognition/UCI-HAR/ResBLSTM/Res_BLSTM.hdf5'


def file_name(file_dir):

    for root, dirs, files in os.walk(file_dir): # root表示正在遍历的文件夹的名字（根/子）, dirs 记录正在遍历的文件夹下的子文件夹集合,files 记录正在遍历的文件夹中的文件集合
        # print('root_dir:', root)  # 当前目录路径
        # print('sub_dirs:', dirs)  # 当前路径下所有子目录
        # print('files:', files)  # 当前路径下所有非目录子文件,列表形式
        return files

file_x_train = ['body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt',
                'body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt',
                'total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt']
file_x_test = ['body_acc_x_test.txt', 'body_acc_y_test.txt', 'body_acc_z_test.txt',
               'body_gyro_x_test.txt', 'body_gyro_y_test.txt', 'body_gyro_z_test.txt',
               'total_acc_x_test.txt', 'total_acc_y_test.txt', 'total_acc_z_test.txt']

file_y_train = 'D:/human activity recognition/UCI-HAR/UCI HAR Dataset/train/y_train.txt'
file_y_test = 'D:/human activity recognition/UCI-HAR/UCI HAR Dataset/test/y_test.txt'

def load_x(data,root):
    x_body_acc =[]
    for file in data[0:3]:
        x_acc_input = np.array(pd.DataFrame(pd.read_csv(root+file, sep='\s+',header=None)))
        x_body_acc.append(x_acc_input)
    x_body_acc = np.transpose(np.array(x_body_acc), (1, 2, 0))

    x_gro =[]
    for file in data[3:6]:
        x_gro_input = np.array(pd.DataFrame(pd.read_csv(root+file, sep='\s+',header=None)))
        x_gro.append(x_gro_input)
    x_gro = np.transpose(np.array(x_gro), (1, 2, 0))

    x_total_acc =[]
    for file in data[6:9]:
        x_total_acc_input = np.array(pd.DataFrame(pd.read_csv(root+file, sep='\s+',header=None)))
        x_total_acc.append(x_total_acc_input)
    x_total_acc = np.transpose(np.array(x_total_acc), (1, 2, 0))
    return x_body_acc, x_gro, x_total_acc

def load_y(data):
    file = open(data, 'r')
    y_input = np.array([elem for elem in [row.strip().split(' ') for row in file]], dtype=np.int32)
    file.close()
    return to_categorical(y_input - 1.0)

x_body_acc_train, x_gro_train, x_total_acc_train = load_x(file_x_train, root_train)
x_body_acc_test, x_gro_test, x_total_acc_test = load_x(file_x_test, root_test)
y_train = load_y(file_y_train)
y_test = load_y(file_y_test)

class parameter(object):
    def __init__(self):
        self.epoch = 150
        self.batch_size = 64
        self.kernel_size = 3
        self.pool_size = 2
        self.drop_rate = 0.2
        self.number = 6

Parameter = parameter()

checkpoint = ModelCheckpoint(model_save_path, save_freq='epoch', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

def ResBLSTM(input):
    model = Bidirectional(LSTM(64, return_sequences=True))(input)
    model = Dropout(0.2)(model)
    x = LayerNormalization()(model)

    model = Bidirectional(LSTM(64, return_sequences=True))(model)
    model = Dropout(0.2)(model)
    y = LayerNormalization()(model)

    model = Add()([x, y])
    output = Bidirectional(LSTM(64, return_sequences=False))(model)
    return output

input_1 = Input(shape=(128, 3))
input_2 = Input(shape=(128, 3))
input_3 = Input(shape=(128, 3))

model_1 = ResBLSTM(input_1)
model_2 = ResBLSTM(input_2)
model_3 = ResBLSTM(input_3)

model = Concatenate()([model_1, model_2, model_3])
model = Dropout(0.5)(model)
model = Dense(Parameter.number)(model)
output = Activation('softmax', name="softmax")(model)

model = Model([input_1, input_2, input_3], output)



model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001), metrics=['accuracy'])

model.summary()

history = model.fit([x_body_acc_train, x_gro_train, x_total_acc_train], y_train,
                    batch_size=Parameter.batch_size,
                    validation_data=([x_body_acc_test, x_gro_test, x_total_acc_test], y_test),
                    epochs=Parameter.epoch,
                    callbacks=[checkpoint])

accuarcy_list = history.history['val_accuracy']
best_accuracy = max(accuarcy_list)
print('best accuracy: {}'.format(best_accuracy))
ResBLSTM_history_file = open('Res_BLSTM_3.json', 'w')
json.dump(history.history, ResBLSTM_history_file)
ResBLSTM_history_file.close()