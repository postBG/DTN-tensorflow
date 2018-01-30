import tensorflow as tf

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
        self.pretrained_model = pretrained_model
        self.test_model = test_model

    # all process use Adam
    def pretrain(self):
        images, labels = preutils.load_svhn(self.svhn_dir, use='extra')
        self.model.build_pretrain_model()
        with tf.Session(config=self.config) as sess:
            writer = tf.summary.FileWriter(self.log_dir + '/pretrain', sess.graph)
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
            saver.save(sess, os.path.join(self.model_save_path, self.pretrained_model))

    def train(self):
        self.model.build_train_model()
        with tf.Session(config=self.config) as sess:
            saver = tf.train.import_meta_graph(os.path.join(self.model_save_path, self.pretrained_model + '.meta'))
            tf.global_variables_initializer().run()
            saver.restore(sess, tf.train.latest_checkpoint(self.model_save_path))

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
