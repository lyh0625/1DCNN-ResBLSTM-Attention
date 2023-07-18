import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, LSTM, Dense, Activation,Concatenate,Bidirectional,AveragePooling1D,Layer, LayerNormalization,Add
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score,accuracy_score
from keras_svm.model_svm_wrapper import ModelSVMWrapper

root_test = 'D:/human activity recognition/UCI-HAR/UCI HAR Dataset/test/Inertial Signals/'
model_load_path = 'D:/human activity recognition/UCI-HAR/CNN_BLSTM_Attention/UCI-CNN-BLSTM-Attention.hdf5'


def file_name(file_dir):

    for root, dirs, files in os.walk(file_dir): # root表示正在遍历的文件夹的名字（根/子）, dirs 记录正在遍历的文件夹下的子文件夹集合,files 记录正在遍历的文件夹中的文件集合
        # print('root_dir:', root)  # 当前目录路径
        # print('sub_dirs:', dirs)  # 当前路径下所有子目录
        # print('files:', files)  # 当前路径下所有非目录子文件,列表形式
        return files


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


x_body_acc_test, x_gro_test, x_total_acc_test = load_x(file_x_test, root_test)
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



class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = tf.keras.initializers.get('glorot_uniform')
        # W_regularizer: 权重上的正则化
        # b_regularizer: 偏置项的正则化
        self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.b_regularizer = tf.keras.regularizers.get(b_regularizer)
        # W_constraint: 权重上的约束项
        # b_constraint: 偏置上的约束项
        self.W_constraint = tf.keras.constraints.get(W_constraint)
        self.b_constraint = tf.keras.constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'step_dim': self.step_dim,
            'W_regularizer': self.W_regularizer,
            'b_regularizer': self.b_regularizer,
            'W_constraint': self.W_constraint,
            'b_constraint': self.b_constraint,
            'bias': self.bias,
        })
        return config

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        '''
        keras.backend.cast(x, dtype): 将张量转换到不同的 dtype 并返回
        '''
        if mask is not None:
            a *= K.cast(mask, K.floatx())

        '''
        keras.backend.epsilon(): 返回浮点数
        '''
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

from tensorflow.keras.models import load_model
_custom_objects = {
    "Attention":Attention
}#告知load_model如何解析自定义Attention
model = load_model(model_load_path,custom_objects=_custom_objects)

val_loss, accuracy = model.evaluate([x_body_acc_test, x_gro_test, x_total_acc_test],y_test)
print('测试集loss：', val_loss, '测试集accuracy：', accuracy)
y_pread = model.predict([x_body_acc_test, x_gro_test, x_total_acc_test])
y_pre = tf.argmax(y_pread,1)
def load_y_true(data):
    df = pd.DataFrame(pd.read_csv(data, header=None))
    df.columns = ['y_true']
    df = df['y_true'].iloc[:]-1
    y_true = df.tolist()
    return y_true
y_true = load_y_true(file_y_test)
# print('真实值:' ,y_true,'\n','预测值:',y_pre)

from sklearn.metrics import classification_report
result = classification_report(y_true, y_pre, digits=4)

import json
CRBA_history_file = open('CRBA_history_test.json', 'w')
json.dump(result, CRBA_history_file)
CRBA_history_file.close()
print('P、R、F:', '\n', result)
print(accuracy_score(y_true, y_pre))
print(precision_score(y_true, y_pre, average='macro'))
print(recall_score(y_true, y_pre, average='macro'))
print(f1_score(y_true, y_pre, average='macro'))

import seaborn as sns
cm = confusion_matrix(y_true, y_pre)
cm = pd.DataFrame(cm, columns=['WALKING', 'UPSTAIRS', 'DOWNSTAIRS', 'SITTING', 'STANTING', 'LAYING'],
                  index=['WALKING', 'UPSTAIRS', 'DOWNSTAIRS', 'SITTING', 'STANTING', 'LAYING'])
sns.heatmap(cm, cmap="YlOrBr", fmt="d", annot=True)
plt.tight_layout()
plt.show()



