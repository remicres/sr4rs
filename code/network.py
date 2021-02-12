import tensorflow as tf
import numpy as np
from functools import partial
from ops import conv, lrelu, conv2d_downscale2d, upscale2d_conv2d, blur2d, pixel_norm
from ops import minibatch_stddev_layer


def discriminator(hr_images, scope, dim):
    """
    Discriminator
    """
    conv_lrelu = partial(conv, activation_fn=lrelu)

    def _combine(x, newdim, name, z=None):
        x = conv_lrelu(x, newdim, 1, 1, name)
        y = x if z is None else tf.concat([x, z], axis=-1)
        return minibatch_stddev_layer(y)

    def _conv_downsample(x, dim, ksize, name):
        y = conv2d_downscale2d(x, dim, ksize, name=name)
        y = lrelu(y)
        return y

    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope("res_4x"):
            net = _combine(hr_images[1], newdim=dim, name="from_input")
            net = conv_lrelu(net, dim, 3, 1, "conv1")
            net = conv_lrelu(net, dim, 3, 1, "conv2")
            net = conv_lrelu(net, dim, 3, 1, "conv3")
            net = _conv_downsample(net, dim, 3, "conv_down")

        with tf.compat.v1.variable_scope("res_2x"):
            net = _combine(hr_images[2], newdim=dim, name="from_input", z=net)
            dim *= 2
            net = conv_lrelu(net, dim, 3, 1, "conv1")
            net = conv_lrelu(net, dim, 3, 1, "conv2")
            net = conv_lrelu(net, dim, 3, 1, "conv3")
            net = _conv_downsample(net, dim, 3, "conv_down")

        with tf.compat.v1.variable_scope("res_1x"):
            net = _combine(hr_images[4], newdim=dim, name="from_input", z=net)
            dim *= 2
            net = conv_lrelu(net, dim, 3, 1, "conv")
            net = _conv_downsample(net, dim, 3, "conv_down")

        with tf.compat.v1.variable_scope("bn"):
            dim *= 2
            net = conv_lrelu(net, dim, 3, 1, "conv1")
            net = _conv_downsample(net, dim, 3, "conv_down1")
            net = minibatch_stddev_layer(net)

            # dense
            dim *= 2
            net = conv_lrelu(net, dim, 1, 1, "dense1")
            net = conv(net, 1, 1, 1, "dense2")
            net = tf.reduce_mean(net, axis=[1, 2])

            return net


def generator(lr_image, scope, nchannels, nresblocks, dim):
    """
    Generator
    """
    hr_images = dict()

    def conv_upsample(x, dim, ksize, name):
        y = upscale2d_conv2d(x, dim, ksize, name)
        y = blur2d(y)
        y = lrelu(y)
        y = pixel_norm(y)
        return y

    def _residule_block(x, dim, name):
        with tf.compat.v1.variable_scope(name):
            y = conv(x, dim, 3, 1, "conv1")
            y = lrelu(y)
            y = pixel_norm(y)
            y = conv(y, dim, 3, 1, "conv2")
            y = pixel_norm(y)
            return y + x

    def conv_bn(x, dim, ksize, name):
        y = conv(x, dim, ksize, 1, name)
        y = lrelu(y)
        y = pixel_norm(y)
        return y

    def _make_output(net, factor):
        hr_images[factor] = conv(net, nchannels, 1, 1, "output")

    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope("encoder"):
            net = lrelu(conv(lr_image, dim, 9, 1, "conv1_9x9"))
            conv1 = net
            for i in range(nresblocks):
                net = _residule_block(net, dim=dim, name="ResBlock{}".format(i))

        with tf.compat.v1.variable_scope("res_1x"):
            net = conv(net, dim, 3, 1, "conv1")
            net = pixel_norm(net)
            net += conv1
            _make_output(net, factor=4)

        with tf.compat.v1.variable_scope("res_2x"):
            net = conv_upsample(net, 4 * dim, 3, "conv_upsample")
            net = conv_bn(net, 4 * dim, 3, "conv1")
            net = conv_bn(net, 4 * dim, 3, "conv2")
            net = conv_bn(net, 4 * dim, 5, "conv3")
            _make_output(net, factor=2)

        with tf.compat.v1.variable_scope("res_4x"):
            net = conv_upsample(net, 4 * dim, 3, "conv_upsample")
            net = conv_bn(net, 4 * dim, 3, "conv1")
            net = conv_bn(net, 4 * dim, 3, "conv2")
            net = conv_bn(net, 4 * dim, 9, "conv3")
            _make_output(net, factor=1)

        return hr_images


def nice_preview(x, refs):
    """
    Beautiful previews
    Keep only first 3 bands --> RGB
    """
    bands = [0, 1, 2]

    _mean = np.zeros(3)
    _std = np.zeros(3)
    _ninv = 1.0 / float(len(refs))
    for ref in refs:
        _mean += _ninv * np.asarray([np.mean(ref[0, :, :, i]) for i in bands])
        _std += _ninv * np.asarray([np.std(ref[0, :, :, i]) for i in bands])
    _min = [__mean - 2 * __std for __mean, __std in zip(_mean, _std)]
    _max = [__mean + 2 * __std for __mean, __std in zip(_mean, _std)]
    return tf.cast(255 * tf.stack(
        [1.0 / (__max - __min) * (tf.clip_by_value(x[:, :, :, i], __min, __max) - __min) for i, __min, __max in
         zip(bands, _min, _max)],
        axis=3), tf.uint8)

