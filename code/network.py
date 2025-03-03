"""
Copyright (c) 2020-2022 Remi Cresson (INRAE)

Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from functools import partial

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

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


def nice_preview(x):
    """
    Beautiful previews
    Keep only first 3 bands --> RGB
    """
    x = x[:, :, :, :3]
    axis = [0, 1, 2]
    stds = tf.math.reduce_std(x, axis=axis, keepdims=True)
    means = tf.math.reduce_mean(x, axis=axis, keepdims=True)
    mins = means - 2 * stds
    maxs = means + 2 * stds
    x = tf.divide(x - mins, maxs - mins)
    x = tf.clip_by_value(x, clip_value_min=0, clip_value_max=1)
    return tf.cast(255 * x, tf.uint8)

