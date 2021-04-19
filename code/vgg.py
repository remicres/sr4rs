from tricks import tf
import logging
import numpy as np

class Vgg19:
    """
    Enables to use VGG19 features maps
    """

    def __init__(self, vgg19_npy_path):
        self.data_dict = np.load(vgg19_npy_path, allow_pickle=True, encoding='latin1').item()
        logging.info("npy file for VGG model loaded")

    def build(self, rgb, mode="1234"):
        """
        load variable from npy to build the VGG
        :param rgb: Tensor for rgb image [batch, height, width, 3] with values scaled in the [0, 1] range
        :param mode: name of the perceptual loss to use
        """

        logging.info("building VGG model")
        with tf.compat.v1.variable_scope("vgg_model", reuse=tf.compat.v1.AUTO_REUSE):

            vgg_mean_rgb = [123.68, 116.779, 103.939]

            # floating point rgb image in [0, 1] range --> 8 bits rbg image in [0, 255] range
            rgb = 255.0 * tf.clip_by_value(rgb, 0.0, 1.0)

            # Convert RGB to BGR
            bgr = tf.concat([rgb[:, :, :, ch:ch + 1] - vgg_mean_rgb[ch] for ch in [2, 1, 0]], axis=-1)

            # Build partial network
            self.conv1_1, _ = self.conv_layer(bgr, "conv1_1")
            self.conv1_2, self.conv1_2lin = self.conv_layer(self.conv1_1, "conv1_2")
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

            self.conv2_1, _ = self.conv_layer(self.pool1, "conv2_1")
            self.conv2_2, self.conv2_2lin = self.conv_layer(self.conv2_1, "conv2_2")
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')

            self.conv3_1, _ = self.conv_layer(self.pool2, "conv3_1")
            self.conv3_2, _ = self.conv_layer(self.conv3_1, "conv3_2")
            self.conv3_3, _ = self.conv_layer(self.conv3_2, "conv3_3")
            self.conv3_4, self.conv3_4lin = self.conv_layer(self.conv3_3, "conv3_4")
            self.pool3 = self.max_pool(self.conv3_4, 'pool3')

            self.conv4_1, _ = self.conv_layer(self.pool3, "conv4_1")
            self.conv4_2, _ = self.conv_layer(self.conv4_1, "conv4_2")
            self.conv4_3, _ = self.conv_layer(self.conv4_2, "conv4_3")
            self.conv4_4, self.conv4_4lin = self.conv_layer(self.conv4_3, "conv4_4")
            self.pool4 = self.max_pool(self.conv4_4, 'pool4')

            self.conv5_1, _ = self.conv_layer(self.pool4, "conv5_1")
            self.conv5_2, _ = self.conv_layer(self.conv5_1, "conv5_2")
            self.conv5_3, _ = self.conv_layer(self.conv5_2, "conv5_3")
            self.conv5_4, self.conv5_4lin = self.conv_layer(self.conv5_3, "conv5_4")
            self.pool5 = self.max_pool(self.conv5_4, 'pool5')
            #
            #    self.fc6 = self.fc_layer(self.pool5, "fc6")
            #    #assert self.fc6.get_shape().as_list()[1:] == [4096]
            #    self.relu6 = tf.nn.relu(self.fc6)
            #
            #    self.fc7 = self.fc_layer(self.relu6, "fc7")
            #    self.relu7 = tf.nn.relu(self.fc7)
            #
            #    self.fc8 = self.fc_layer(self.relu7, "fc8")
            #
            #    self.prob = tf.nn.softmax(self.fc8, name="prob")
            #
            #    self.data_dict = None

            if mode == "1234":
                f1 = tf.reshape(self.pool1, shape=[-1, 1])
                f2 = tf.reshape(self.pool2, shape=[-1, 1])
                f3 = tf.reshape(self.pool3, shape=[-1, 1])
                f4 = tf.reshape(self.pool4, shape=[-1, 1])
                return [f1, f2, f3, f4]
            elif mode == "1234lin":
                f1 = tf.reshape(self.conv1_2lin, shape=[-1, 1])
                f2 = tf.reshape(self.conv2_2lin, shape=[-1, 1])
                f3 = tf.reshape(self.conv3_4lin, shape=[-1, 1])
                f4 = tf.reshape(self.conv4_4lin, shape=[-1, 1])
                f5 = tf.reshape(self.conv5_4lin, shape=[-1, 1])
                return [f1, f2, f3, f4, f5]
            elif mode == "vgg344454":
                f1 = tf.reshape(self.conv3_4, shape=[-1, 1])
                f2 = tf.reshape(self.conv4_4, shape=[-1, 1])
                f3 = tf.reshape(self.conv5_4, shape=[-1, 1])
                return [f1, f2, f3]
            elif mode == "vgg54":
                return [tf.reshape(self.conv5_4, shape=[-1, 1])]
            elif mode == "vgg54lin":
                return [tf.reshape(self.conv5_4lin, shape=[-1, 1])]
            else:
                raise Exception("VGG loss \"{}\" not implemented!".format(mode))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool2d(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool2d(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.compat.v1.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv_fn = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv_fn, conv_biases)

            relu = tf.nn.relu(bias)
            return relu, bias

    def fc_layer(self, bottom, name):
        with tf.compat.v1.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")


def compute_vgg_loss(ref, gen, mode, vggfile):
    """
    Compute "perceptual" (VGG19 based) loss
    :param ref: reference image (rgb image [batch, height, width, ch])
    :param gen: generated image (rgb image [batch, height, width, ch])
    :param mode: name of the perceptual loss to use
    """
    assert vggfile is not None

    logging.info("Using pre-trained VGG from {}".format(vggfile))
    vgg_model = Vgg19(vggfile)
    features_ref = vgg_model.build(ref[:, :, :, 0:3], mode=mode)
    features_gen = vgg_model.build(gen[:, :, :, 0:3], mode=mode)

    loss = 0.0
    for fr, fp in zip(features_ref, features_gen):
        loss += tf.reduce_mean(tf.square(fr - fp))

    return loss
