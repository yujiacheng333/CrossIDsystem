import tensorflow as tf
from tensorflow.python.keras import backend as k
import matplotlib.pyplot as plt


class CIN(tf.keras.Model):
    def __init__(self, channels):
        super(CIN, self).__init__()

        self.gamma_mat = tf.Variable(tf.ones([6*16, channels]), trainable=True, name="gamma")
        self.beta_mat = tf.Variable(tf.zeros([6*16, channels]), trainable=True, name="beta")

    def call(self, inputs, **kwargs):
        inputs, label = inputs
        label = tf.cast(tf.one_hot(label, depth=6*16), tf.float32)
        local_gamma = tf.einsum("sn,bs->bn", self.gamma_mat, label)[:, tf.newaxis]
        local_beta = tf.einsum("sn,bs->bn", self.beta_mat, label)[:, tf.newaxis]

        # buff_gamma = tf.einsum("sn,an->sa", self.gamma_mat, self.gamma_mat)
        # plt.imshow(buff_gamma.numpy())
        # plt.show()

        # local_gamma = k.sum(self.gamma_mat[:3], axis=0)/3
        # local_beta = k.sum(self.beta_mat[:3], axis=0)/3
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        inputs = local_beta + local_gamma * (inputs - mean) / k.sqrt(var + 1e-12)
        return inputs


class Conv1D(tf.keras.Model):
    def __init__(self, filters, kernel_size, stride):
        super(Conv1D, self).__init__()
        self.conv = tf.keras.layers.Conv1D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=stride, use_bias=False)
        self.norm = CIN(filters)
        # self.norm = tf.keras.layers.BatchNormalization(moment=0.9)

    def call(self, inputs, training=None, mask=None):
        inputs, pid = inputs
        inputs = self.conv(inputs)
        inputs = self.norm([inputs, pid])
        inputs = tf.nn.relu(inputs)
        return inputs


class CMFace(tf.keras.Model):
    def __init__(self, units, s=30., m1=0.9, m2=0.5, m3=0.4):
        super(CMFace, self).__init__()
        self.w = None
        self.units = units
        self.s = s
        self.m_list = [m1, m2, m3]
        self.eps = 1e-6

    def build(self, input_shape):
        try:
            self.w = self.add_weight(name='embedding_weights', shape=[input_shape[0][-1], self.units],
                                     dtype=tf.float32, trainable=True)
        except:
            self.w = self.add_weight(name='embedding_weights', shape=[input_shape[-1], self.units],
                                     dtype=tf.float32, trainable=True)

    def call(self, inputs, training=None, mask=None):

        if training:
            featurerep, labels = inputs
            labels = tf.cast(tf.one_hot(labels, depth=self.units), tf.float32)
            embedding = tf.nn.l2_normalize(featurerep, axis=-1)
            weights = tf.nn.l2_normalize(self.w, axis=0)
            cos_t = tf.matmul(embedding, weights)
            # cos_t = tf.clip_by_value(cos_t, clip_value_min=-1., clip_value_max=1.)
            theta = tf.math.acos(cos_t)
            theta_m = theta * self.m_list[0] + self.m_list[1]
            theta_m = tf.where(theta_m < 3.1415926535, theta, theta_m)
            cos_t_m = tf.math.cos(theta_m) - self.m_list[2]
            cos_t_m = tf.where(cos_t_m > -1., cos_t_m, cos_t)  # release margin
            inv_labels = 1. - labels
            output = self.s * (cos_t * inv_labels + cos_t_m * labels)
        else:
            output = tf.matmul(inputs, self.w)
        return output


class CRN(tf.keras.Model):
    def __init__(self, actnum=4):
        super(CRN, self).__init__()
        self.conv0 = Conv1D(filters=128, kernel_size=3, stride=1)
        self.conv1 = Conv1D(filters=64, kernel_size=3, stride=1)
        self.conv2 = Conv1D(filters=128, kernel_size=3, stride=2)
        self.conv3 = Conv1D(filters=256, kernel_size=3, stride=2)
        self.conv4 = Conv1D(filters=512, kernel_size=3, stride=2)
        self.conv5 = Conv1D(filters=1024, kernel_size=3, stride=2)
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dense = tf.keras.layers.Dense(units=128, use_bias=False)
        self.logitsout = CMFace(units=actnum)
        # self.logitsout = tf.keras.layers.Dense(actnum)  # for dense layers

    def call(self, inputs, training=None, mask=None):
        if training:
            inputs, pid, labela = inputs
        else:
            inputs, pid = inputs
        mean, var = tf.nn.moments(inputs, axes=-1, keepdims=True)
        inputs -= mean
        inputs /= k.sqrt(var)
        inputs = self.conv0([inputs, pid])
        inputs = self.conv1([inputs, pid])
        inputs = self.conv2([inputs, pid])
        inputs = self.conv3([inputs, pid])
        inputs = self.conv4([inputs, pid])
        inputs = self.conv5([inputs, pid])
        inputs = self.dense(self.pool(inputs))
        if training:
            inputs = self.logitsout([inputs, labela], training)
        else:
            inputs = self.logitsout(inputs, training)
        # inputs = self.logitsout(inputs)  # for dense layers
        return inputs


if __name__ == '__main__':
    test_input = tf.ones([1, 800, 270])
    test_pid = tf.ones([1], dtype=tf.int64)
    test_aid = tf.ones([1], dtype=tf.int64)
    model = CRN()
    model([test_input, test_aid, test_pid], training=True)
    a = model.trainable_variables
    trainable_var = []
    for i in a:
        if "gamma" in i.name or "beta" in i.name:
            trainable_var.append(i)
