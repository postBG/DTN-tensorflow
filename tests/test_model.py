import unittest
import tensorflow as tf

from model import generator, discriminator, feature_extractor, loss_const, loss_tid


class TestDTNUtils(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_pretrain모드일_경우_feature_extractor는_logits를_리턴(self):
        src_images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        logits = feature_extractor(src_images, pretrain=True)

        self.assertListEqual([None, 10], logits.get_shape().as_list(),
                             'Incorrect Image Shape.  Found {} shape'.format(logits.get_shape().as_list()))

    def test_pretrain모드가_아닐경우_feature_extractor는_1x1x128형태의_activations반환(self):
        src_images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        activations = feature_extractor(src_images, pretrain=False)

        self.assertListEqual([None, 1, 1, 128], activations.get_shape().as_list(),
                             'Incorrect Image Shape.  Found {} shape'.format(activations.get_shape().as_list()))

    def test_generator는_1x1x128형태의_입력을_받아_32x32x1형태의_이미지를_생성(self):
        fx = tf.placeholder(tf.float32, shape=[None, 1, 1, 128])
        generated = generator(fx)

        self.assertListEqual([None, 32, 32, 1], generated.get_shape().as_list(),
                             'Incorrect Image Shape.  Found {} shape'.format(generated.get_shape().as_list()))

    def test_discriminator는_32x32x1형태의_입력을_받아_3개의_클래스_logits으로_요약(self):
        t_shape_images = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
        discriminated = discriminator(t_shape_images)

        self.assertListEqual([None, 3], discriminated.get_shape().as_list(),
                             'Incorrect Image Shape.  Found {} shape'.format(discriminated.get_shape().as_list()))

    def test_loss_const(self):
        batch_size = 10
        fx = tf.fill(dims=[batch_size, 1, 1, 128], value=2.)
        fgfx = tf.zeros(shape=[batch_size, 1, 1, 128], dtype=tf.float32)

        with tf.Session() as sess:
            l_const = sess.run(loss_const(fx, fgfx))
            self.assertAlmostEqual(4. * batch_size, l_const)

    def test_loss_tid(self):
        batch_size = 10
        images = tf.fill(dims=[batch_size, 32, 32, 3], value=2.)
        images2 = tf.zeros(shape=[batch_size, 32, 32, 3])

        with tf.Session() as sess:
            l_tid = sess.run(loss_tid(images, images2))
            self.assertAlmostEqual(4. * batch_size, l_tid)
