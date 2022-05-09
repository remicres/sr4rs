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
# Imports
from tricks import tf
import datetime
import argparse
from functools import partial
import otbtf
import logging
from ops import downscale2d
from vgg import compute_vgg_loss
import network
import constants

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.WARNING,
                    datefmt='%Y-%m-%d %H:%M:%S')

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

parser = argparse.ArgumentParser()

# Paths
parser.add_argument("--lr_patches", help="LR patches images list", required=True, nargs='+', default=[])
parser.add_argument("--hr_patches", help="HR patches images list", required=True, nargs='+', default=[])
parser.add_argument("--preview", help="LR image for preview (must have the same dynamic as lr_patches)")
parser.add_argument("--logdir", help="Output directory for tensorboard summaries (must be an existing directory)")
parser.add_argument("--save_ckpt", help="Prefix for the checkpoints that will be saved", required=True)
parser.add_argument("--load_ckpt", help="Path to an existing checkpoint (provide the full path without the .meta "
                                        "extension)")
parser.add_argument("--savedmodel", help="Create a SavedModel after the training step (must be a new directory)")
parser.add_argument("--vggfile", help="Path to the vgg19.npy file")

# Images scaling
parser.add_argument("--lr_scale", type=float, default=0.0001, help="LR image scaling. Set a value such as the scaled "
                                                                   "pixels values are mostly in the [0,1] range.")
parser.add_argument("--hr_scale", type=float, default=0.0001, help="HR image scaling. Set a value such as the scaled "
                                                                   "pixels values are mostly in the [0,1] range.")

# Parameters
parser.add_argument("--previews_step", type=int, default=200, help="Number of steps between each preview summary")
parser.add_argument("--depth", type=int, default=64, help="Generator and discriminator depth")
parser.add_argument("--nresblocks", type=int, default=16, help="Number of ResNet blocks in Generator")
parser.add_argument("--epochs", type=int, default=120, help="number of epochs")
parser.add_argument("--batchsize", type=int, default=4, help="batch size")
parser.add_argument("--adam_lr", type=float, default=0.001, help="Adam learning rate")
parser.add_argument("--adam_b1", type=float, default=0.0, help="Adam beta1")
parser.add_argument("--l1weight", type=float, default=0, help="L1 loss weight")
parser.add_argument("--l2weight", type=float, default=1000.0, help="L2 loss weight")
parser.add_argument("--vggweight", type=float, default=0.00001, help="VGG loss weight")
parser.add_argument('--vggfeatures', default="1234", const="1234", nargs="?",
                    choices=["vgg54", "vgg54lin", "vgg344454", "1234lin", "1234"])
parser.add_argument("--ganweight", type=float, default=1.0, help="GAN loss weight")
parser.add_argument('--losstype', default="WGAN-GP", const="WGAN-GP", nargs="?",
                    choices=["WGAN-GP", "LSGAN"], help="GAN loss type")
parser.add_argument('--streaming', dest='streaming', action='store_true',
                    help="Streaming reads patches from the file system. Consumes low RAM amount, but performs a lot of"
                         " filesystem reading operations.")
parser.set_defaults(streaming=False)
parser.add_argument('--pretrain', dest='pretrain', action='store_true',
                    help="Pre-train the network without GAN losses, using only L1 and/or L2 (depending of l1weight "
                         "and l2weight)")
parser.set_defaults(pretrain=False)
params = parser.parse_args()

step = 0

def main(unused_argv):
    logging.info("************ Parameters summary ************")
    logging.info("Number of epochs     : " + str(params.epochs))
    logging.info("Batch size           : " + str(params.batchsize))
    logging.info("Adam learning rate   : " + str(params.adam_lr))
    logging.info("Adam beta1           : " + str(params.adam_b1))
    logging.info("L1 loss weight       : " + str(params.l1weight))
    logging.info("L2 loss weight       : " + str(params.l2weight))
    logging.info("GAN loss weight      : " + str(params.ganweight))
    if params.vggfile is not None:
        logging.info("VGG file             : " + str(params.vggfile))
        logging.info("VGG loss weight      : " + str(params.vggweight))
        logging.info("VGG features         : " + str(params.vggfeatures))
    logging.info("Base depth           : " + str(params.depth))
    logging.info("Number of ResBlocks  : " + str(params.nresblocks))
    logging.info("Low-res image scale  : " + str(params.lr_scale))
    logging.info("Hi-res image scale   : " + str(params.hr_scale))
    logging.info("********************************************")

    # Preview
    lr_image_for_prev = None
    if params.preview is not None:
        lr_image_for_prev = otbtf.read_as_np_arr(otbtf.gdal_open(params.preview), False)

    with tf.Graph().as_default():
        # dataset and iterator
        ds = otbtf.DatasetFromPatchesImages(filenames_dict={constants.hr_key: params.hr_patches,
                                                            constants.lr_key: params.lr_patches},
                                            use_streaming=params.streaming)
        tf_ds = ds.get_tf_dataset(batch_size=params.batchsize)
        iterator = tf.compat.v1.data.Iterator.from_structure(ds.output_types)
        iterator_init = iterator.make_initializer(tf_ds)
        dataset_inputs = iterator.get_next()

        # model inputs
        def _get_input(key, name):
            default_input = dataset_inputs[key]
            shape = (None, None, None, ds.output_shapes[key][-1])
            return tf.compat.v1.placeholder_with_default(default_input, shape=shape, name=name)

        lr_image = _get_input(constants.lr_key, constants.lr_input_name)
        hr_image = _get_input(constants.hr_key, constants.hr_input_name)

        # model
        hr_nch = ds.output_shapes[constants.hr_key][-1]
        generator = partial(network.generator, scope=constants.gen_scope, nchannels=hr_nch,
                            nresblocks=params.nresblocks, dim=params.depth)
        discriminator = partial(network.discriminator, scope=constants.dis_scope, dim=params.depth)

        hr_images_real = {factor: params.hr_scale * tf.cast(downscale2d(hr_image, factor=factor), tf.float32) 
                          for factor in constants.factors}
        hr_images_fake = generator(params.lr_scale * tf.cast(lr_image, tf.float32))

        # model outputs
        gen = {factor: (1.0 / params.hr_scale) * hr_images_fake[factor] for factor in constants.factors}
        for pad in constants.pads:
            tf.identity(gen[1][:, pad:-pad, pad:-pad, :], name="{}{}".format(constants.outputs_prefix, pad))
        if lr_image_for_prev is not None:
            for factor in constants.factors:
                prev = network.nice_preview(gen[factor])
                tf.compat.v1.summary.image("preview_factor{}".format(factor), prev, collections=[constants.epoch_key])

        # discriminator
        dis_real = discriminator(hr_images=hr_images_real)
        dis_fake = discriminator(hr_images=hr_images_fake)

        # l1 loss
        gen_loss_l1 = tf.add_n([tf.reduce_mean(tf.abs(hr_images_fake[factor] -
                                                      hr_images_real[factor])) for factor in constants.factors])

        # l2 loss
        gen_loss_l2 = tf.add_n([tf.reduce_mean(tf.square(hr_images_fake[factor] -
                                                         hr_images_real[factor])) for factor in constants.factors])

        # VGG loss
        gen_loss_vgg = 0.0
        if params.vggfile is not None:
            gen_loss_vgg = tf.add_n([compute_vgg_loss(hr_images_real[factor],
                                                      hr_images_fake[factor],
                                                      params.vggfeatures,
                                                      params.vggfile) for factor in constants.factors])

        # GAN Losses
        if params.losstype == "LSGAN":
            dis_loss = tf.reduce_mean(tf.square(dis_real - 1) + tf.square(dis_fake))
            gen_loss_gan = tf.reduce_mean(tf.square(dis_fake - 1))
        elif params.losstype == "WGAN-GP":
            dis_loss = dis_fake - dis_real
            alpha = tf.random_uniform(shape=[params.batchsize, 1, 1, 1], minval=0., maxval=1.)
            differences = {factor: hr_images_fake[factor] - hr_images_real[factor] for factor in constants.factors}
            interpolates_scales = {factor: hr_images_real[factor] +
                                           alpha * differences[factor] for factor in constants.factors}
            mixed_loss = tf.reduce_sum(discriminator(interpolates_scales))
            mixed_grads = tf.gradients(mixed_loss, list(interpolates_scales.values()))
            mixed_norms = [tf.sqrt(tf.reduce_sum(tf.square(gradient), reduction_indices=[1, 2, 3])) for gradient in
                           mixed_grads]
            gradient_penalties = [tf.reduce_mean(tf.square(slope - 1.0)) for slope in mixed_norms]
            gradient_penalty = tf.reduce_mean(gradient_penalties)
            dis_loss += 10 * gradient_penalty
            epsilon_penalty = tf.reduce_mean(tf.square(dis_real))
            dis_loss += 0.001 * epsilon_penalty
            gen_loss_gan = -1.0 * tf.reduce_mean(dis_fake)
            dis_loss = tf.reduce_mean(dis_loss)
        else:
            raise Exception("Please select an available cost function")

        # Total losses
        def _new_loss(value, name, collections=None):
            tf.compat.v1.summary.scalar(name, value, collections)
            return value

        train_collections = [constants.train_key]
        all_collections = [constants.pretrain_key, constants.train_key]
        gen_loss = _new_loss(params.ganweight * gen_loss_gan, "gen_loss_gan", train_collections)
        gen_loss += _new_loss(params.vggweight * gen_loss_vgg, "gen_loss_vgg", train_collections)
        pretrain_loss = _new_loss(params.l1weight * gen_loss_l1, "gen_loss_l1", all_collections)
        pretrain_loss += _new_loss(params.l2weight * gen_loss_l2, "gen_loss_l2", all_collections)
        gen_loss += pretrain_loss
        dis_loss = _new_loss(dis_loss, "dis_loss", train_collections)

        # discriminator optimizer
        dis_optim = tf.compat.v1.train.AdamOptimizer(learning_rate=params.adam_lr, beta1=params.adam_b1)
        dis_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, constants.dis_scope)
        dis_grads_and_vars = dis_optim.compute_gradients(dis_loss, var_list=dis_tvars)
        with tf.compat.v1.variable_scope("apply_dis_gradients", reuse=tf.compat.v1.AUTO_REUSE):
            dis_train = dis_optim.apply_gradients(dis_grads_and_vars)

        # generator optimizer
        with tf.control_dependencies([dis_train]):
            gen_optim = tf.compat.v1.train.AdamOptimizer(learning_rate=params.adam_lr, beta1=params.adam_b1)
            gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, constants.gen_scope)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            with tf.compat.v1.variable_scope("apply_gen_gradients", reuse=tf.compat.v1.AUTO_REUSE):
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        pretrain_op = tf.compat.v1.train.AdamOptimizer(learning_rate=params.adam_lr).minimize(pretrain_loss)
        train_nodes = [gen_train]
        if params.losstype == "LSGAN":
            ema = tf.train.ExponentialMovingAverage(decay=0.995)
            update_losses = ema.apply([dis_loss, gen_loss])
            train_nodes.append(update_losses)
        train_op = tf.group(train_nodes, name="optimizer")

        merged_losses_summaries = tf.compat.v1.summary.merge_all(key=constants.train_key)
        merged_pretrain_summaries = tf.compat.v1.summary.merge_all(key=constants.pretrain_key)
        merged_preview_summaries = tf.compat.v1.summary.merge_all(key=constants.epoch_key)

        init = tf.global_variables_initializer()
        saver = tf.compat.v1.train.Saver(max_to_keep=5)

        sess = tf.Session()

        # Writer
        def _append_desc(key, value):
            if value == 0:
                return ""
            return "_{}{}".format(key, value)

        # Summary file name (include all settings in the name)
        now = datetime.datetime.now()
        summaries_fn = "SR4RS"
        summaries_fn += _append_desc("E", params.epochs)
        summaries_fn += _append_desc("B", params.batchsize)
        summaries_fn += _append_desc("LR", params.adam_lr)
        summaries_fn += _append_desc("Gan", params.ganweight)
        summaries_fn += _append_desc("L1-", params.l1weight)
        summaries_fn += _append_desc("L2-", params.l2weight)
        summaries_fn += _append_desc("VGG", params.vggweight)
        summaries_fn += _append_desc("VGGFeat", params.vggfeatures)
        summaries_fn += _append_desc("Loss", params.losstype)
        summaries_fn += _append_desc("D", params.depth)
        summaries_fn += _append_desc("RB", params.nresblocks)
        summaries_fn += _append_desc("LRSC", params.lr_scale)
        summaries_fn += _append_desc("HRSC", params.hr_scale)
        if params.pretrain:
            summaries_fn += "pretrained"
        summaries_fn += "_{}{}_{}h{}min".format(now.day, now.strftime("%b"), now.hour, now.minute)

        train_writer = None
        if params.logdir is not None:
            train_writer = tf.summary.FileWriter(params.logdir + summaries_fn, sess.graph)

        # Helper to write the summary
        def _add_summary(summarized, _step):
            if train_writer is not None:
                train_writer.add_summary(summarized, _step)

        # Weights initialization
        sess.run(init)
        if params.load_ckpt is not None:
            saver.restore(sess, params.load_ckpt)

        # preview
        def _preview(_step):
            if lr_image_for_prev is not None and step % params.previews_step == 0:
                summary_pe = sess.run(merged_preview_summaries, {lr_image: lr_image_for_prev})
                _add_summary(summary_pe, _step)


        # Helper to train the model
        def _do(_train_op, _summary_op, name):
            global step
            for curr_epoch in range(params.epochs):
                logging.info("{} Epoch #{}".format(name, curr_epoch))
                sess.run(iterator_init)
                try:
                    while True:
                        _, _summary = sess.run([_train_op, _summary_op])
                        _add_summary(_summary, step)
                        _preview(curr_epoch)
                        step += 1
                except tf.errors.OutOfRangeError:
                    fs_stall_duration = ds.get_total_wait_in_seconds()
                    logging.info("{}: one epoch done. Total FS stall: {:.2f}s".format(name, fs_stall_duration))
                    pass
                saver.save(sess, params.save_ckpt + summaries_fn, global_step=curr_epoch)

        # pre training
        if params.pretrain:
            _do(pretrain_op, merged_pretrain_summaries, "pre-training")

        # training
        _do(train_op, merged_losses_summaries, "training")

        # cleaning
        if train_writer is not None:
            train_writer.close()

        # Export SavedModel
        if params.savedmodel is not None:
            logging.info("Export SavedModel in {}".format(params.savedmodel))
            outputs = ["{}{}:0".format(constants.outputs_prefix, pad) for pad in constants.pads]
            inputs = ["{}:0".format(constants.lr_input_name)]
            graph = tf.get_default_graph()
            tf.saved_model.simple_save(sess,
                                       params.savedmodel,
                                       inputs={i: graph.get_tensor_by_name(i) for i in inputs},
                                       outputs={o: graph.get_tensor_by_name(o) for o in outputs})

    quit()


if __name__ == "__main__":
    tf.compat.v1.app.run(main)
