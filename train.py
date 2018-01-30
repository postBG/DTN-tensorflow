import tensorflow as tf
import tensorflow.contrib.slim as slim

import os
import preproc.utils as preutils
from preproc.preprocess import SVHN_PATH, MNIST_PATH


class Trainer:
    """
    https://wookayin.github.io/TensorFlowKR-2017-talk-bestpractice/ko/#37
    """

    def __init__(self, model, batch_size=128, pretrain_iter=500, train_iter=2000,
                 sample_iter=100, svhn_dir=SVHN_PATH, mnist_dir=MNIST_PATH, log_dir='./logs', sample_save_path='sample',
                 model_save_path='model', pretrained_model='svhn_model-20000', test_model='dtn-1800'):
        self.model = model
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.batch_size = batch_size
        # iteration
        self.pretrain_iter = pretrain_iter
        self.train_iter = train_iter
        self.sample_iter = sample_iter

        # directory
        self.svhn_dir = svhn_dir
        self.mnist_dir = mnist_dir
        self.log_dir = log_dir

        # path
        self.sample_save_path = sample_save_path
        self.model_save_path = model_save_path

        # model
        self.pretrained_model = os.path.join(self.model_save_path, pretrained_model)
        self.test_model = test_model

    # all process use Adam
    def pretrain(self):
        images, labels = preutils.load_svhn(self.svhn_dir, use='extra')
        self.model.build_pretrain_model()
        with tf.Session(config=self.config) as sess:
            writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'pretrain'), sess.graph)
            saver = tf.train.Saver()
            tf.global_variables_initializer().run()
            limit = images.shape[0] // self.batch_size
            for step in range(self.pretrain_iter):
                i = step % limit
                batch_images = images[i * self.batch_size:(i + 1) * self.batch_size]
                batch_labels = labels[i * self.batch_size:(i + 1) * self.batch_size]

                feed_dict = {self.model.s_images: batch_images, self.model.s_labels: batch_labels}
                _, merged = sess.run([self.model.pretrain_op, self.model.merged], feed_dict)
                writer.add_summary(merged, step)

                if (step % 100) == 0:
                    print("Step {:6} : Loss {:.8}\tAccuracy : {:.5}".format(step, sess.run(self.model.loss, feed_dict),
                                                                            sess.run(self.model.accuracy, feed_dict)))
            # Save pretrained model
            saver.save(sess, self.pretrained_model)

    def train(self):
        src_images, _ = preutils.load_svhn(self.svhn_dir, use='train')
        trg_images, _ = preutils.load_mnist(self.mnist_dir, use='train')

        self.model.build_train_model()

        variables_to_restore = slim.get_model_variables(scope='feature_extractor')
        restorer = tf.train.Saver(variables_to_restore)

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            restorer.restore(sess, self.pretrained_model)

            summary_writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'train'), sess.graph)
            saver = tf.train.Saver()

            num_of_src_batches = src_images.shape[0] // self.batch_size
            num_of_trg_batches = trg_images.shape[0] // self.batch_size
            for step in range(self.train_iter):
                i = step % num_of_src_batches
                j = step % num_of_trg_batches

                src_batch = src_images[i * self.batch_size:(i + 1) * self.batch_size]
                trg_batch = trg_images[j * self.batch_size:(j + 1) * self.batch_size]
                feed_dict = {
                    self.model.s_images: src_batch,
                    self.model.t_images: trg_batch
                }

                _, _, _, _, merged = sess.run([self.model.d_train_op_src,
                                               self.model.g_train_op_src,
                                               self.model.d_train_op_trg,
                                               self.model.g_train_op_trg,
                                               self.model.merged], feed_dict=feed_dict)
                summary_writer.add_summary(merged, step)

                if (step + 1) % 1000 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'dtn'), global_step=step + 1)
                    print('model dtn-%d has saved' % step + 1)
