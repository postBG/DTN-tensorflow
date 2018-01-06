import os
import tensorflow as tf

from model import DTN
from train import Trainer

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', "'pretrain', 'train' or 'sample'")

# hyper-parameters
flags.DEFINE_integer("epoch", 100, "Epoch to train [100]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_integer("input_size", 32, "THe size of input images [32]") # assume square. We need it because we will use other dataset
flags.DEFINE_integer("output_size", 32, "THe size of output images [32]") # assume square. We need it because we will use other dataset
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
    s_labels = tf.placeholder(tf.float32, shape=[None,10], name='s_labels')

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    model = DTN(s_images, t_images, s_labels, learning_rate, FLAGS)
    trainer = Trainer(model)

    trainer.pretrain()

if __name__ == '__main__':
    tf.app.run()
