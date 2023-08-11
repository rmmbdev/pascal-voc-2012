from keras.layers import Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, LocallyConnected2D, Add, Average, \
    AveragePooling2D, MaxPool2D, Concatenate, BatchNormalization, Dropout, MaxPool1D, Flatten, MaxPooling1D, Conv1D, \
    Activation

import keras.backend as k
from keras.engine.topology import Layer

import static_values as sv

import tensorflow as tf


def get_model(input):
    inp_bn = tf.layers.batch_normalization(input)

    inp_resized = tf.image.resize_images(input, [128, 128])
    inp_resized_bn = tf.layers.batch_normalization(inp_resized)

    with tf.name_scope("vgg19"):
        x = tf.keras.applications.VGG19(include_top=False, weights='imagenet')(inp_resized_bn)

    with tf.name_scope("densnet"):
        y = tf.keras.applications.DenseNet169(include_top=False, weights='imagenet')(inp_bn)

    num_maps = 20
    with tf.name_scope("dimensionality_reduction"):
        net1 = tf.layers.conv2d(x, num_maps * sv.STATIC_VALUES.labels_count, [1, 1], activation=tf.nn.relu)
        net2 = tf.layers.conv2d(y, num_maps * sv.STATIC_VALUES.labels_count, [1, 1], activation=tf.nn.relu)

    last = tf.keras.layers.Concatenate()(
        [tf.keras.layers.GlobalMaxPooling2D()(net1), tf.keras.layers.GlobalMaxPooling2D()(net2),
         tf.keras.layers.GlobalAveragePooling2D()(net1), tf.keras.layers.GlobalAveragePooling2D()(net2)])
    z = tf.layers.batch_normalization(last)
    with tf.name_scope("fully_connected"):
        z = tf.layers.dense(512)(z)
        z = tf.layers.batch_normalization(z)
        z = tf.nn.leaky_relu(z)
        z = tf.nn.dropout(z, 0.5)

        z = tf.layers.dense(sv.STATIC_VALUES.labels_count, activation=tf.nn.sigmoid)(z)

    return z


def NiNBlock(kernel, mlps, strides):
    def inner(x):
        l = Conv2D(mlps[0], kernel, strides=strides, padding='same')(x)
        l = Activation('relu')(l)
        for size in mlps[1:]:
            l = Conv2D(size, 1, strides=[1, 1])(l)
            l = Activation('sigmoid')(l)
        return l

    return inner


class NetworkInNetwork(Layer):
    def __init__(self, output_dims, **kwargs):
        super(NetworkInNetwork, self).__init__(**kwargs)
        self.trainable = True
        self.output_dims = output_dims

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, 1, input_shape[3], int(self.output_dims)),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(int(self.output_dims),),
                                    initializer='glorot_uniform',
                                    trainable=True)

    def call(self, inputs, **kwargs):
        outputs = k.conv2d(inputs, self.kernel)
        outputs = k.bias_add(outputs, self.bias)
        outputs = k.relu(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.output_dims)


class Slice(Layer):
    def __init__(self, begin, size, **kwargs):
        super(Slice, self).__init__(**kwargs)
        self.begin = begin * size
        self.size = size

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        outputs = tf.slice(inputs, begin=[0, 0, 0, self.begin], size=[-1, -1, -1, self.size])
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.size)


class ElementWiseMax(Layer):
    def __init__(self, **kwargs):
        super(ElementWiseMax, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        input1 = inputs[0]
        input2 = inputs[1]

        expanded_inp1 = tf.expand_dims(input1, axis=1)
        expanded_inp2 = tf.expand_dims(input2, axis=1)

        concatenated = tf.concat([expanded_inp1, expanded_inp2], axis=1)
        outputs = tf.reduce_max(concatenated, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])


class ElementWiseAvg(Layer):
    def __init__(self, **kwargs):
        super(ElementWiseAvg, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        input1 = inputs[0]
        input2 = inputs[1]

        expanded_inp1 = tf.expand_dims(input1, axis=1)
        expanded_inp2 = tf.expand_dims(input2, axis=1)

        concatenated = tf.concat([expanded_inp1, expanded_inp2], axis=1)
        outputs = tf.reduce_mean(concatenated, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])


class ElementWiseAdd(Layer):
    def __init__(self, **kwargs):
        super(ElementWiseAdd, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        input1 = inputs[0]
        input2 = inputs[1]

        expanded_inp1 = tf.expand_dims(input1, axis=1)
        expanded_inp2 = tf.expand_dims(input2, axis=1)

        concatenated = tf.concat([expanded_inp1, expanded_inp2], axis=1)
        outputs = tf.reduce_sum(concatenated, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])


class ElementWiseAvg3(Layer):
    def __init__(self, **kwargs):
        super(ElementWiseAvg3, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        input1 = inputs[0]
        input2 = inputs[1]
        input3 = inputs[2]

        expanded_inp1 = tf.expand_dims(input1, axis=1)
        expanded_inp2 = tf.expand_dims(input2, axis=1)
        expanded_inp3 = tf.expand_dims(input3, axis=1)

        concatenated = tf.concat([expanded_inp1, expanded_inp2, expanded_inp3], axis=1)
        outputs = tf.reduce_mean(concatenated, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])


class ElementWiseMax3(Layer):
    def __init__(self, **kwargs):
        super(ElementWiseMax3, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        input1 = inputs[0]
        input2 = inputs[1]
        input3 = inputs[2]

        expanded_inp1 = tf.expand_dims(input1, axis=1)
        expanded_inp2 = tf.expand_dims(input2, axis=1)
        expanded_inp3 = tf.expand_dims(input3, axis=1)

        concatenated = tf.concat([expanded_inp1, expanded_inp2, expanded_inp3], axis=1)
        outputs = tf.reduce_max(concatenated, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])


class GetHSV(Layer):
    def __init__(self, **kwargs):
        super(GetHSV, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        outputs = tf.image.rgb_to_hsv(inputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])


class ResizeInp(Layer):
    def __init__(self, new_size, **kwargs):
        super(ResizeInp, self).__init__(**kwargs)
        self.new_size = new_size

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        outputs = tf.image.resize_images(inputs, [int(self.new_size[0]), int(self.new_size[1])])
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], int(self.new_size[0]), int(self.new_size[1]), input_shape[3])


class ClassWisePooling(Layer):
    def __init__(self, **kwargs):
        super(ClassWisePooling, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        m = 20
        _, w, h, n = inputs.get_shape().as_list()
        n_classes = n // m
        ops = []
        for i in range(n_classes):
            class_avg_op = tf.reduce_mean(inputs[:, :, :, m * i:m * (i + 1)], axis=3, keep_dims=True)
            ops.append(class_avg_op)
        final_op = tf.concat(ops, axis=3)
        return final_op

    def compute_output_shape(self, input_shape):
        return (input_shape[0], int(self.new_size[0]), int(self.new_size[1]), input_shape[3])


def class_wise_pooling(x):
    m = 8
    _, _, _, n = x.get_shape().as_list()
    n_classes = n // m
    ops = []
    for i in range(n_classes):
        class_avg_op = tf.reduce_mean(x[:, :, :, m * i:m * (i + 1)], axis=3, keep_dims=True)
        ops.append(class_avg_op)
    final_op = tf.concat(ops, axis=3)
    return final_op


def spatial_pooling(x):
    k = 3
    alpha = 0.7

    batch_size, w, h, n_classes = x.get_shape().as_list()
    x_flat = tf.reshape(x, shape=(-1, w * h, n_classes))
    x_transp = tf.transpose(x_flat, perm=(0, 2, 1))
    k_maxs, _ = tf.nn.top_k(x_transp, k, sorted=False)
    k_maxs_mean = tf.reduce_mean(k_maxs, axis=2)
    result = k_maxs_mean
    if alpha:
        # top -x_flat to retrieve the k smallest values
        k_mins, _ = tf.nn.top_k(-x_transp, k, sorted=False)
        # flip back
        k_mins = -k_mins
        k_mins_mean = tf.reduce_mean(k_mins, axis=2)
        alpha = tf.constant(alpha, name='alpha', dtype=tf.float32)
        result += alpha * k_mins_mean
    return result
