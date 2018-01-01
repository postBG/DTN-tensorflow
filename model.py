import tensorflow
import tensorflow.contrib.slim as slim


class DTN:
    def __init__(self, s_images, t_images, learning_rate, configs):
        self.s_images = s_images
        self.t_images = t_images
        self.learning_rate = learning_rate
        pass

    def function_f(self):
        raise NotImplementedError

    def generator(self):
        raise NotImplementedError

    def discriminator(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
