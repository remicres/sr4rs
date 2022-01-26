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
from tricks import tf
from functools import partial
import numpy as np

lrelu = partial(tf.nn.leaky_relu, alpha=0.2)


def get_weight(shape, gain=np.sqrt(2), lrmul=1):
    """
    Get weight, or return a new variable for weight
    """
    fan_in = np.prod(shape[:-1])
    he_std = gain / np.sqrt(fan_in)  # He init

    # Equalized learning rate and custom learning rate multiplier.
    init_std = 1.0 / lrmul
    runtime_coef = he_std * lrmul

    init = tf.initializers.random_normal(0, init_std)
    return tf.compat.v1.get_variable("weight", shape=shape, initializer=init) * runtime_coef


def apply_bias(x):
    """
    Apply bias to the input
    """
    b = tf.compat.v1.get_variable('bias', shape=[x.shape[-1]], initializer=tf.keras.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, 1, 1, -1])


def conv_base(x, fmaps, kernel_size, stride, name, gain=np.sqrt(2), activation_fn=None, normalizer_fn=None,
              transpose=False, padding='SAME'):
    """
    Convolutional layer base
    """
    assert (isinstance(name, str))
    with tf.compat.v1.variable_scope(name):
        strides = [1, stride, stride, 1]
        if not transpose:
            w = get_weight([kernel_size, kernel_size, x.shape[3].value, fmaps], gain=gain)
            w = tf.cast(w, x.dtype)
            out = tf.nn.conv2d(x, filter=w, strides=strides, padding=padding)
        else:
            sz0 = tf.shape(x)[0]
            sz1 = tf.shape(x)[1]
            sz2 = tf.shape(x)[2]
            output_shape = [sz0, stride * sz1, stride * sz2, fmaps]
            w = get_weight([kernel_size, kernel_size, fmaps, x.shape[3].value], gain=gain)
            w = tf.cast(w, x.dtype)
            out = tf.nn.conv2d_transpose(x, filter=w, output_shape=output_shape, strides=strides, padding=padding)
        out = apply_bias(out)
        if activation_fn is not None:
            out = activation_fn(out)
        if normalizer_fn is not None:
            out = normalizer_fn(out)
        return out


conv = partial(conv_base, transpose=False)
deconv = partial(conv_base, transpose=True)


def _blur2d(x, stride=1, flip=False):
    """
    Blur an image tensor
    """
    f = [1, 2, 1]
    f = np.array(f, dtype=np.float32)
    f = f[:, np.newaxis] * f[np.newaxis, :]
    f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1]
    f = f[:, :, np.newaxis, np.newaxis]
    f = np.tile(f, [1, 1, int(x.shape[-1]), 1])
    f = tf.constant(f, dtype=tf.float32, name="filter_blur2d")
    strides = [1, stride, stride, 1]
    return tf.nn.depthwise_conv2d(x, f, strides=strides, padding="SAME")


def minibatch_stddev_layer(x, group_size=4):
    """
    Mini-batch standard deviation layer
    """
    with tf.compat.v1.variable_scope('MinibatchStd'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])
        sz1 = tf.shape(x)[1]
        sz2 = tf.shape(x)[2]
        sz3 = tf.shape(x)[3]
        y = tf.reshape(x, [group_size, -1, sz1, sz2, sz3])
        y = tf.cast(y, tf.float32)
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        y = tf.reduce_mean(tf.square(y), axis=0)
        y = tf.sqrt(y + 1e-8)
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)
        y = tf.cast(y, x.dtype)
        y = tf.tile(y, [group_size, sz1, sz2, 1])
        return tf.concat([x, y], axis=3)


def pixel_norm(x):
    """
    Pixel normalization
    """
    with tf.compat.v1.variable_scope("PixelNorm"):
        epsilon = tf.constant(1e-8, dtype=x.dtype, name="epsilon")
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon)


def conv2d_downscale2d(x, fmaps, kernel, name):
    """
    Conv2D + downscaling
    """
    assert kernel >= 1 and kernel % 2 == 1
    with tf.compat.v1.variable_scope(name):
        w = get_weight([kernel, kernel, x.shape[3].value, fmaps])
        w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode="CONSTANT")
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
        w = tf.cast(w, x.dtype)
        return tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding="SAME")


def upscale2d_conv2d(x, fmaps, kernel, name):
    """
    Upscaling with transposed conv2D
    """
    assert kernel >= 1 and kernel % 2 == 1
    with tf.compat.v1.variable_scope(name):
        w = get_weight([kernel, kernel, x.shape[3].value, fmaps])
        w = tf.transpose(w, [0, 1, 3, 2])
        w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode="CONSTANT")
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
        w = tf.cast(w, x.dtype)
        sz0 = tf.shape(x)[0]
        sz1 = tf.shape(x)[1]
        sz2 = tf.shape(x)[2]
        output_shape = [sz0, 2 * sz1, 2 * sz2, fmaps]
        return tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, 2, 2, 1], padding="SAME")


def _upscale2d(x, factor=2, gain=1):
    """
    Nearest-neighborhood based upscaling
    """
    if gain != 1:
        x *= gain

    if factor == 1:
        return x

    s = tf.shape(x)
    x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
    x = tf.tile(x, [1, 1, factor, 1, factor, 1])
    x = tf.reshape(x, [-1, factor * s[1], factor * s[2], s[3]])
    return x


def _downscale2d(x, factor=2, gain=1):
    """
    Average-pooling + blur based downscaling
    """
    if gain != 1:
        x *= gain

    if factor == 2:
        return _blur2d(x, stride=factor)

    if factor == 1:
        return x

    ksize = [1, factor, factor, 1]
    return tf.nn.avg_pool2d(x, ksize=ksize, strides=ksize, padding="VALID")


def blur2d(x):
    """
    Blur with custom gradient
    """
    with tf.compat.v1.variable_scope("Blur2D"):
        @tf.custom_gradient
        def func(in_x):
            y = _blur2d(in_x)

            @tf.custom_gradient
            def grad(dy):
                dx = _blur2d(dy, flip=True)
                return dx, lambda ddx: _blur2d(ddx)

            return y, grad

        return func(x)


def upscale2d(x, factor=2):
    """
    Upscaling with custom gradient
    """
    with tf.compat.v1.variable_scope("Upscale2D"):
        @tf.custom_gradient
        def func(in_x):
            y = _upscale2d(in_x, factor)

            @tf.custom_gradient
            def grad(dy):
                dx = _downscale2d(dy, factor, gain=factor ** 2)
                return dx, lambda ddx: _upscale2d(ddx, factor)

            return y, grad

        return func(x)


def downscale2d(x, factor=2):
    """
    Downscaling with custom gradient
    """
    with tf.compat.v1.variable_scope("Downscale2D"):
        @tf.custom_gradient
        def func(in_x):
            y = _downscale2d(in_x, factor)

            @tf.custom_gradient
            def grad(dy):
                dx = _upscale2d(dy, factor, gain=1 / factor ** 2)
                return dx, lambda ddx: _downscale2d(ddx, factor)

            return y, grad

        return func(x)
