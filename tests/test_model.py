import unittest
import tensorflow as tf

from model import DTN


class MockConfig:
    def __init__(self, mode='pretrain', alpha=15, beta=15, gamma=0):
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


class TestDTN(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.model = DTN(tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='s_images'),
                         tf.placeholder(tf.float32, shape=[None, 32, 32, 1], name='t_images'),
                         tf.placeholder(tf.float32, name='learning_rate'),
                         MockConfig())

    def test_pretrain모드일_경우_feature_extractor는_logits를_리턴(self):
        src_images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        logits = self.model.feature_extractor(src_images)

        self.assertListEqual([None, 10], logits.get_shape().as_list(),
                             'Incorrect Image Shape.  Found {} shape'.format(logits.get_shape().as_list()))

    def test_pretrain모드가_아닐경우_feature_extractor는_1x1x128형태의_activations반환(self):
        self.model.mode = 'train'
        src_images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        activations = self.model.feature_extractor(src_images)

        self.assertListEqual([None, 1, 1, 128], activations.get_shape().as_list(),
                             'Incorrect Image Shape.  Found {} shape'.format(activations.get_shape().as_list()))

    def test_generator는_1x1x128형태의_입력을_받아_32x32x1형태의_이미지를_생성(self):
        fx = tf.placeholder(tf.float32, shape=[None, 1, 1, 128])
        generated = self.model.generator(fx)

        self.assertListEqual([None, 32, 32, 1], generated.get_shape().as_list(),
                             'Incorrect Image Shape.  Found {} shape'.format(generated.get_shape().as_list()))

    def test_discriminator는_32x32x1형태의_입력을_받아_3개의_클래스_logits으로_요약(self):
        t_shape_images = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
        discriminated = self.model.discriminator(t_shape_images)

        self.assertListEqual([None, 3], discriminated.get_shape().as_list(),
                             'Incorrect Image Shape.  Found {} shape'.format(discriminated.get_shape().as_list()))
