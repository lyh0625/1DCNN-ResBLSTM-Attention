import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

model_load_path = 'D:/human activity recognition/ku_har/CNN_ResBLSTM_Attention.hdf5'
# plt.rc('font', family='Times New Roman')
# plt.rc('font', family='Arial')

f = np.load('D:/human activity recognition/ku_har/new_dataset.npz')
signals = f['signals']
labels = f['labels']

# split to train-test
X_train, X_test, y_train, y_test = train_test_split(
    signals, labels, test_size=0.2, random_state=9, stratify=labels
)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

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



"""
导入模型
"""
_custom_objects = {
    "Attention": Attention
}#告知load_model如何解析自定义Attention
model = load_model(model_load_path, custom_objects=_custom_objects)
val_loss, accuracy = model.evaluate(X_test, y_test)
print('测试集loss：', val_loss, '测试集accuracy：', accuracy)
y_pread = model.predict(X_test)
y_pre = tf.argmax(y_pread, 1)   #将one-hot变回标签
y_true = tf.argmax(y_test, 1)
# print('真实值:', y_true, '\n', '预测值:',y_pre)
"""
计算每个动作的评价指标以及总体评价指标
"""
from sklearn.metrics import classification_report
result = classification_report(y_true, y_pre, digits=4)
print('P、R、F:', '\n', result)

print('accuracy:', accuracy_score(y_true, y_pre))
print('precision:', precision_score(y_true, y_pre, average='macro'))
print('recall:', recall_score(y_true, y_pre, average='macro'))
print('f1_score:', f1_score(y_true, y_pre, average='macro'))
"""
画出混淆矩阵
"""
import seaborn as sns
cm = confusion_matrix(y_true, y_pre)
cm = pd.DataFrame(cm, columns=["Stand", "Sit", "Talk-sit", "Talk-stand", "Stand-sit", "Lay", "Lay-stand", "Pick",
                               "Jump", "Push-up", "Sit-up", "Walk", "Walk-backward", "Walk-circle", "Run", "Stair-up",
                               "Stair-down", "Table-tennis"],
                  index=["Stand", "Sit", "Talk-sit", "Talk-stand", "Stand-sit", "Lay", "Lay-stand", "Pick", "Jump",
                         "Push-up", "Sit-up", "Walk", "Walk-backward", "Walk-circle", "Run", "Stair-up",  "Stair-down",
                         "Table-tennis"])
sns.heatmap(cm, cmap="YlOrBr", fmt="d", annot=True)
plt.tight_layout()
plt.show()
