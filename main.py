import os
import tensorflow as tf

from model import dtn_model_factory
from train import Trainer

flags = tf.app.flags
flags.DEFINE_string('mode', 'pretrain', "'pretrain', 'train' or 'sample'")

# hyper-parameters
flags.DEFINE_integer("epoch", 100, "Epoch to train [100]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_string("model", 'svhn2mnist', "One of ['svhn2mnist']")
flags.DEFINE_float("alpha", 15., "The value of alpha [15]")
flags.DEFINE_float("beta", 15., "The value of beta [15]")
flags.DEFINE_float("gamma", 0., "The value of gamma [0]")

flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")

FLAGS = flags.FLAGS


def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    s_images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='s_images')
    t_images = tf.placeholder(tf.float32, shape=[None, 32, 32, 1], name='t_images')
    s_labels = tf.placeholder(tf.float32, shape=[None, 10], name='s_labels')

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    model = dtn_model_factory(s_images, t_images, s_labels, learning_rate, FLAGS)
    trainer = Trainer(model, learning_rate=FLAGS.learning_rate)

    if FLAGS.mode == 'pretrain':
        trainer.pretrain()


if __name__ == '__main__':
    tf.app.run()
