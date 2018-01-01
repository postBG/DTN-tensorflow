from __future__ import division
from glob import glob
from six.moves import xrange
import numpy as np
import tensorflow as tf
import os
import time
import math


class DTN(object):
    def __init__(self, sess, input_height=32, input_width=32, batch_size=64, sample_num=64, output_height=32,
                 output_width=32,
                 c_dim=3, dataset_name='default', checkpoint_dir=None, sample_dir=None)

    self.sess = sess

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    # TODO
    # LG = L_Gang + alpha L_const + beta L_tid + gamma L_tv

    def build_model(self):

    # TODO

    def train(self, config):

    # TODO

    def domain_acceptor(self, content, input):  # function f

    # TODO
    # SVHN
    # 64 128 256 128
    # max pooling 4 x 4
    # ReLU



    def discriminator(self, image, reuse=False):

    # TODO
    def generator(self, f):

    # TODO

    def sampler(self, image):

    # TODO?

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DTN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
