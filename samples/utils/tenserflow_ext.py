import tensorflow as tf
import numpy as np


def weight_variable(shape, name="W", stddev=0.1):
    initial = tf.truncated_normal(shape=shape, stddev=stddev)
    return tf.Variable(initial, name=name, trainable=True)


def bias_variable(shape, name="b", stddev=0.1):
    initial = tf.constant(stddev, shape=shape)
    return tf.Variable(initial, name=name, trainable=True)


def add_pad2d(x, pad_size=1, name="pad2d", format="NHWC"):
    with tf.name_scope(name) as scope:
        z = tf.convert_to_tensor(x)
        if format == "NHWC":
            z = tf.pad(z, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], name=scope)
        elif format == "NCHW":
            z = tf.pad(z, [[0, 0], [0, 0], [pad_size, pad_size], [pad_size, pad_size]], name=scope)
    return z


def add_conv2d(x, output_size, h_kernel, w_kernel, name, h_stride=1, w_stride=1, padding="SAME",
               activation="relu", leaky_relu_alpha=0.1, format="NHWC", batch_normalization=False, training=True):
    x = tf.convert_to_tensor(x)
    input_size = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weight = weight_variable([h_kernel, w_kernel, input_size, output_size], name="W_" + scope)
        bias = bias_variable([output_size], name="b_" + scope)
        conv = tf.nn.conv2d(x, weight, strides=[1, h_stride, w_stride, 1], padding=padding, data_format=format)
        z = tf.nn.bias_add(conv, bias)
        if batch_normalization:
            z = tf.layers.batch_normalization(z, training=training)

        if activation == "relu":
            z = tf.nn.relu(z, name=scope)
        elif activation == "sigmoid":
            z = tf.nn.sigmoid(z, name=scope)
        elif activation == "leaky_relu":
            z = tf.nn.leaky_relu(z, alpha=leaky_relu_alpha, name=scope)
        elif activation == "linear":
            z = add_linear(z, name=scope)
        return z


def add_fc(x, output_size, name, activation="relu", leaky_relu_alpha=0.1):
    input_size = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weight = weight_variable([input_size, output_size], name="W_" + scope)
        bias = bias_variable([output_size], name="b_" + scope)
        mul = tf.matmul(x, weight)
        z = tf.add(mul, bias)
        if activation == "relu":
            z = tf.nn.relu(z, name=scope)
        elif activation == "sigmoid":
            z = tf.nn.sigmoid(z, name=scope)
        elif activation == "leaky_relu":
            z = tf.nn.leaky_relu(z, alpha=leaky_relu_alpha, name=scope)
        elif activation == "linear":
            z = add_linear(z, name=scope)

        return z


def add_pool(x, name, h_kernel=2, w_kernel=2, h_stride=2, w_stride=2, padding="SAME"):
    with tf.name_scope(name) as scope:
        z = tf.nn.max_pool(x, ksize=[1, h_kernel, w_kernel, 1], strides=[1, h_stride, w_stride, 1], padding=padding,
                           name="p_" + scope)
        return z


def add_flatten(x, name):
    input_size = x.get_shape()
    flat_shape = input_size[1].value * input_size[2].value * input_size[3].value
    z = tf.transpose(x, perm=[0, 3, 1, 2])
    z = tf.reshape(z, [-1, flat_shape], name=name)
    return z


def add_dropout(x, name, keep_prob=0.5, flag=True):
    if flag:
        dropout = tf.nn.dropout(x, keep_prob, name=name)
        return dropout
    else:
        return x


def add_shortcut(x, shortcut, name="short_cut"):
    z = tf.add(x, shortcut, name=name)
    return z


def add_linear(x, name="linear"):
    return tf.multiply(x, 1, name=name)


