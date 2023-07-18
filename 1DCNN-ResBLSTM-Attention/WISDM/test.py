import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, LSTM, Dense, Activation,\
    Bidirectional, AveragePooling1D, Layer, LayerNormalization, Add
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score,f1_score,recall_score,accuracy_score


file_path = 'D:/human activity recognition/WISDM/WISDM_ar_v1.1_raw.txt'
model_load_path = 'D:/human activity recognition/WISDM/WISDM-CNN-BLSTM-Attention-test.hdf5'

colnames = ['users', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
dataset = pd.read_csv(file_path, names=colnames)
dataset = dataset.dropna()

n_time_step = 128
n_feature = 3
step = 64
segments = []
labels = []

for i in range(0, len(dataset) - n_time_step, step):
    x = dataset['x-axis'].values[i: i + n_time_step]
    y = dataset['y-axis'].values[i: i + n_time_step]
    z = dataset['z-axis'].values[i: i + n_time_step]
    label = stats.mode(dataset['activity'][i: i + n_time_step])[0][0]
    segments.append([x, y, z])
    labels.append(label)

reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, n_time_step, n_feature)
labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)

X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labels, test_size=0.2, random_state=1)


class parameter(object):
    def __init__(self):
        self.epoch = 150
        self.batch_size = 128
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


def CNN_ResBLSTM_Attention(input):
    model = Conv1D(256, Parameter.kernel_size, input_shape=(128, 3), activation='swish', padding='same')(input)
    model = BatchNormalization()(model)
    model = MaxPooling1D(pool_size=Parameter.pool_size)(model)
    model = Dropout(Parameter.drop_rate)(model)

    model = Conv1D(128, Parameter.kernel_size, activation='swish', padding='same')(model)
    model = BatchNormalization()(model)
    model = MaxPooling1D(pool_size=Parameter.pool_size)(model)
    model = Dropout(Parameter.drop_rate)(model)

    model = Conv1D(64, Parameter.kernel_size, activation='swish', padding='same')(model)
    model = BatchNormalization()(model)
    model = MaxPooling1D(pool_size=Parameter.pool_size)(model)
    model = Dropout(Parameter.drop_rate)(model)

    model = Conv1D(32, Parameter.kernel_size, activation='swish', padding='same')(model)
    model = BatchNormalization()(model)
    model = MaxPooling1D(pool_size=Parameter.pool_size)(model)
    model = Dropout(Parameter.drop_rate)

    model = LayerNormalization()(model)
    x = Bidirectional(LSTM(64, return_sequences=True))(model)

    model = LayerNormalization()(x)
    y = Bidirectional(LSTM(64, return_sequences=True))(model)
    model = Add()([x, y])

    output = Attention(8)(model)

    return output

input_1 = Input(shape=(128, 3))
model = CNN_ResBLSTM_Attention(input_1)
model = Dropout(0.5)(model)
model = Dense(Parameter.number)(model)
output = Activation('softmax', name="softmax")(model)

model = Model(input_1, output)

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,epsilon=1e-07), metrics=['accuracy'])

from tensorflow.keras.models import load_model
_custom_objects = {
    "Attention":Attention,
}#告知load_model如何解析自定义Attention
model = load_model(model_load_path, custom_objects=_custom_objects)
val_loss, accuracy = model.evaluate(X_test, y_test)
print('测试集loss：', val_loss,'测试集accuracy：',accuracy)
y_pread = model.predict(X_test)
y_pre = tf.argmax(y_pread,1)
y_true = tf.argmax(y_test,1)
# print('真实值:', y_true, '\n', '预测值:',y_pre)

from sklearn.metrics import classification_report
result = classification_report(y_true, y_pre, digits=4)
print('P、R、F:', '\n', result)
print(accuracy_score(y_true, y_pre))
print(precision_score(y_true, y_pre, average='macro'))
print(recall_score(y_true, y_pre, average='macro'))
print(f1_score(y_true, y_pre, average='macro'))
import seaborn as sns
cm = confusion_matrix(y_true, y_pre)
cm = pd.DataFrame(cm, columns=['DOWNSTAIRS', 'JOGGING', 'SITTING', 'STANDING', 'UPSTAIRS', 'WALKING'],
                  index=['DOWNSTAIRS', 'JOGGING', 'SITTING', 'STANDING', 'UPSTAIRS', 'WALKING'])
sns.heatmap(cm, cmap="YlOrBr", fmt="d", annot=True)
plt.tight_layout()
plt.show()
