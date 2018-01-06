import tensorflow as tf
import tensorflow.contrib.slim as slim


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
    
    # all process use Adam
    def pretrain(self):
        learning_rate = self.model.learning_rate
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            
            # some friction occur between below and gpu_options
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = slim.learning.create_train_op(self.model.loss, self.optimizer)
            slim.learning.train(self.train_op,'logs/')
            # TODO

    def train(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
