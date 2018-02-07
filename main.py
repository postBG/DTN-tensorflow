import os
import tensorflow as tf

from model import dtn_model_factory
from preproc.preprocess import SVHN_PATH, MNIST_PATH
from train import Trainer

flags = tf.app.flags
flags.DEFINE_string('mode', 'pretrain', "'pretrain', 'train' or 'sample'")

# hyper-parameters used in model
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("alpha", 15., "The value of alpha [15]")
flags.DEFINE_float("beta", 15., "The value of beta [15]")
flags.DEFINE_float("gamma", 0., "The value of gamma [0]")

# hyper-parameters and settings used in train
flags.DEFINE_integer("epoch", 100, "Epoch to train [100]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_integer("pretrain_iter", 500, "The number of iteration of pretrain")
flags.DEFINE_integer("sample_iter", 100, "The number of iteration of sampling")
flags.DEFINE_integer("train_g_weights", 5, "How many times to train g ops when train d ops once")
flags.DEFINE_string("svhn_dir", SVHN_PATH, "data path of svhn dataset")
flags.DEFINE_string("mnist_dir", MNIST_PATH, "data path of mnist dataset")
flags.DEFINE_string("log_dir", 'logs', "path for logs")
flags.DEFINE_string("model_save_path", 'model', "path for saving models")
flags.DEFINE_string("model_read_path", 'model', "path for reading models")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")

# other settings
flags.DEFINE_string("model", 'svhn2mnist', "One of ['svhn2mnist']")

FLAGS = flags.FLAGS


def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.model_save_path):
        os.makedirs(FLAGS.model_save_path)
    if not os.path.exists(FLAGS.model_read_path):
        os.makedirs(FLAGS.model_read_path)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    s_images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='s_images')
    t_images = tf.placeholder(tf.float32, shape=[None, 32, 32, 1], name='t_images')
    s_labels = tf.placeholder(tf.float32, shape=[None, 10], name='s_labels')

    model = dtn_model_factory(s_images, t_images, s_labels, FLAGS)
    trainer = Trainer(model, batch_size=FLAGS.batch_size, pretrain_iter=FLAGS.pretrain_iter, log_dir=FLAGS.log_dir,
                      sample_iter=FLAGS.sample_iter, svhn_dir=FLAGS.svhn_dir, mnist_dir=FLAGS.mnist_dir,
                      sample_save_path=FLAGS.sample_dir, model_save_path=FLAGS.model_save_path,
                      model_read_path=FLAGS.model_read_path, train_g_weights=FLAGS.train_g_weights)

    if FLAGS.mode == 'pretrain':
        trainer.pretrain()
    if FLAGS.mode == 'train':
        trainer.train()
    if FLAGS.mode == 'sample':
        trainer.sample()


if __name__ == '__main__':
    tf.app.run()
