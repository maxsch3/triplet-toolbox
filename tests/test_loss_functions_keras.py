import tensorflow as tf
import importlib
import pytest
from triplet_tools import triplet_batch_semihard_loss, triplet_batch_priming_loss, triplet_batch_hard_loss

try:
    import keras
except ImportError:
    pass


@pytest.mark.skipif(importlib.util.find_spec("keras") is None,
                    reason='Keras is not installed in this environment (not needed when testing tensorflow 2 )')
class TestLossFunctionsKeras:

    def setup_method(self):
        self.train_data, self.test_data = self.load_mnist()

    def load_mnist(self):
        mnist = tf.keras.datasets.mnist
        return mnist.load_data()

    def make_model(self):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='linear')
        ])
        return model

    def test_batch_priming_loss(self):
        model = self.make_model()
        loss_func = triplet_batch_priming_loss()
        model.compile('adam', loss_func)
        hist = model.fit(self.train_data[0], self.train_data[1], epochs=2)
        loss_hist = hist.history['loss']
        assert loss_hist[-1] < 0.1
        assert loss_hist[0] > loss_hist[1]

    def test_batch_hard_loss(self):
        model = self.make_model()
        loss_func = triplet_batch_priming_loss()
        model.compile('adam', loss_func)
        hist = model.fit(self.train_data[0], self.train_data[1], epochs=1)
        bh_loss_func = triplet_batch_hard_loss()
        model.compile('adam', bh_loss_func)
        hist = model.fit(self.train_data[0], self.train_data[1], epochs=10, batch_size=100, shuffle=True)
        loss_hist = hist.history['loss']
        assert loss_hist[-1] < 0.5
        assert loss_hist[0] > loss_hist[1]
