import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import preproc.utils as preutils


class Trainer:
    """
    https://wookayin.github.io/TensorFlowKR-2017-talk-bestpractice/ko/#37
    """

    def __init__(self, model, batch_size=128, pretrain_iter=20000, train_iter=2000, sample_iter=100,
                 svhn_dir='svhn', mnist_dir='mnist', log_dir='logs', sample_save_path='sample',
                 model_save_path='model', pretrained_model='model/svhn_model-20000', test_model='model/dtn-1800'):
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
        images, labels = preutils.load_svhn()
        learning_rate = self.model.learning_rate

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            limit = images.shape[0] // self.batch_size
            self.optimizer = tf.train.AdamOptimizer(learning_rate)

            for step in range(self.pretrain_iter):
                i = step % limit
                batch_images = images[i * self.batch_size:(i + 1) * self.batch_size]
                batch_labels = labels[i * self.batch_size:(i + 1) * self.batch_size]

                feed_dict = {self.model.s_images: batch_images, self.model.s_labels: batch_labels}
                self.train_op = slim.learning.create_train_op(self.model.loss, self.optimizer)
                sess.run(self.model.train_op, feed_dict)

                if ((step + 1) % 100) == 0:
                    print("Step {} : Loss {}".format(step + 1, self.model.loss))

    def train(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
