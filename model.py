import tensorflow as tf
import tensorflow.contrib.slim as slim


def _is_mnist(images):
    return images.get_shape()[3] == 1


def feature_extractor(images, reuse=False, pretrain=False):
    if _is_mnist(images):
        images = tf.image.grayscale_to_rgb(images)

    with tf.variable_scope("feature_extractor", reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding="SAME", kernel_size=3, stride=1):
            with slim.arg_scope([slim.max_pool2d], kernel_size=2, stride=2):
                # for now images shape [batch_size, 32, 32, 3]
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

                if pretrain:
                    flatten = slim.flatten(pool4)
                    logits = slim.fully_connected(flatten, 10, activation_fn=None, scope="f_out")
                    return logits

                return pool4


def generator(extracted_features, reuse=False, is_training=True):
    with tf.variable_scope("generator", reuse=reuse):
        with slim.arg_scope([slim.conv2d_transpose], activation_fn=None, kernel_size=3, stride=2, padding='SAME'):
            with slim.arg_scope([slim.batch_norm], is_training=is_training, activation_fn=tf.nn.relu):
                # For now extracted feature shapes [batch_size, 1, 1, 128]
                deconv1 = slim.conv2d_transpose(extracted_features, 512, kernel_size=4, stride=4, scope="deconv1")
                bn1 = slim.batch_norm(deconv1, scope="bn1")
                # shape [batch_size, 4, 4, 512]

                deconv2 = slim.conv2d_transpose(bn1, 256, scope="deconv2")
                bn2 = slim.batch_norm(deconv2, scope="bn2")
                # shape [batch_size, 8, 8, 256]

                deconv3 = slim.conv2d_transpose(bn2, 128, scope="deconv3")
                bn3 = slim.batch_norm(deconv3, scope="bn3")
                # shape [batch_size, 16, 16, 128]

                deconv4 = slim.conv2d_transpose(bn3, 1, scope="deconv4")
                bn4 = slim.batch_norm(deconv4, activation_fn=tf.nn.tanh, scope="bn4")
                # shape [batch_size, 32, 32, 1]

                return bn4


def discriminator(t_shape_images, reuse=False, is_training=True):
    with tf.variable_scope("discriminator", reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=None, kernel_size=3, stride=2, padding='SAME'):
            with slim.arg_scope([slim.batch_norm], is_training=is_training, activation_fn=tf.nn.relu):
                # now shape [batch_size, 32, 32, 1]
                conv1 = slim.conv2d(t_shape_images, 128, scope="conv1")
                bn1 = slim.batch_norm(conv1, scope="bn1")
                # shape [batch_size, 16, 16, 128]

                conv2 = slim.conv2d(bn1, 256, scope="conv2")
                bn2 = slim.batch_norm(conv2, scope="bn2")
                # shape [batch_size, 8, 8, 256]

                conv3 = slim.conv2d(bn2, 512, scope="conv3")
                bn3 = slim.batch_norm(conv3, scope="bn3")
                # shape [batch_size, 4, 4, 512]

                conv4 = slim.conv2d(bn3, 1024, scope="conv4")
                bn4 = slim.batch_norm(conv4, scope="bn4")
                # shape [batch_size, 2, 2, 1024]

                one_by_one_conv = slim.conv2d(bn4, 3, scope="one_by_one_conv")
                # shape [batch_size, 1, 1, 3]
                logits = slim.flatten(one_by_one_conv, scope="logits")
                # shape [batch_size, 3]

                return logits


class DTN:
    def __init__(self, s_images, t_images, learning_rate, configs):
        self.s_images = s_images
        self.t_images = t_images
        self.learning_rate = learning_rate

        self.mode = configs.mode
        self.alpha = configs.alpha
        self.beta = configs.beta
        self.gamma = configs.gamma

    def build_model(self):
        if self.mode == 'pretrain':
            raise NotImplementedError
        elif self.mode == 'train':
            raise NotImplementedError
        else:
            raise NotImplementedError
