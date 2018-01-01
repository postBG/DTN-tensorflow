class Trainer:
    """
    https://wookayin.github.io/TensorFlowKR-2017-talk-bestpractice/ko/#37
    """

    def __init__(self, model, batch_size=128, pretrain_iter=20000, train_iter=2000, sample_iter=100,
                 svhn_dir='svhn', mnist_dir='mnist', log_dir='logs', sample_save_path='sample',
                 model_save_path='model', pretrained_model='model/svhn_model-20000', test_model='model/dtn-1800'):
        self.model = model

    def train(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
