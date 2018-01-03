import tensorflow as tf
import tensorflow.contrib.slim as slim


class DTN:
    def __init__(self, s_images, t_images, learning_rate, configs):
        self.s_images = s_images
        self.t_images = t_images
        self.learning_rate = learning_rate
        self.mode = configs.mode

    def feature_extractor(self, images, reuse=False):
        if self._is_mnist(images):
            images = tf.image.grayscale_to_rgb(images)

        with tf.variable_scope("feature_extractor", reuse=reuse):
            with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding="SAME", kernel_size=3, stride=1):
                with slim.arg_scope([slim.max_pool2d], kernel_size=2, stride=2):
                    conv1 = slim.conv2d(images, 64, scope="conv1")
                    pool1 = slim.max_pool2d(conv1, scope="pool1")
                    # now images shape [batch_size, 16, 16, 64]

                    conv2 = slim.conv2d(pool1, 128, scope="conv2")
                    pool2 = slim.max_pool2d(conv2, scope="pool2")
                    # now images shape [batch_size, 8, 8, 128]

                    conv3 = slim.conv2d(pool2, 256, scope="conv3")
                    pool3 = slim.max_pool2d(conv3, scope="pool3")
                    # now images shape [batch_size, 4, 4, 256]

                    conv4 = slim.conv2d(pool3, 128, scope="conv4")
                    pool4 = slim.max_pool2d(conv4, kernel_size=4, stride=4, scope="pool4")
                    # now activation shape [batch_size, 1, 1, 128]

                    if self.mode == 'pretrain':
                        flatten = slim.flatten(pool4)
                        logits = slim.fully_connected(flatten, 10, scope="f_out", activation_fn=None)
                        return logits

                    return pool4

    def generator(self, images, reuse=False, training=True):
        raise NotImplementedError

    def discriminator(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def _is_mnist(self, images):
        return images.get_shape()[3] == 1
