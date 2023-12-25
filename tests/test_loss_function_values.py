import numpy as np
import tensorflow as tf
from triplet_tools import triplet_batch_priming_loss, triplet_batch_semihard_loss, triplet_batch_hard_loss


class TestSharedFunctions:

    emb = None
    labels = None

    def setup_method(self):
        self.emb = np.array([
            [.22, -.5, .11],
            [.3, -.66, .05],
            [.6, -0.27, .1],
            [.8, 0.2, .1],
            [.72, .24, .12]
        ])
        self.labels = np.array([1, 1, 1, 2, 2])

    def evaluate_tf_graph(self, g):
        if tuple([int(s) for s in tf.__version__.split('.')]) < (2, 0, 0):
            with tf.Session() as sess:
                value = g.eval()
        else:
            value = g
        return value

    def test_priming_loss_basic(self):
        triplet_loss = triplet_batch_priming_loss()
        g = triplet_loss(self.labels, self.emb)
        loss = self.evaluate_tf_graph(g)
        assert abs(loss - 0.30346566) < 0.00001

    def test_batch_hard_loss_basic(self):
        triplet_loss = triplet_batch_hard_loss(margin=0.3)
        g = triplet_loss(self.labels, self.emb)
        loss = self.evaluate_tf_graph(g)
        assert abs(loss - 0.07150213112) < 0.00001

    def test_batch_semihard_loss_basic(self):
        triplet_loss = triplet_batch_semihard_loss()
        g = triplet_loss(self.labels, self.emb)
        loss = self.evaluate_tf_graph(g)
        assert abs(loss - 0.5303652637399063) < 0.00001

    def test_batch_semihard_loss_with_semi_margin(self):
        triplet_loss = triplet_batch_semihard_loss(semi_margin=0.1)
        g = triplet_loss(self.labels, self.emb)
        loss = self.evaluate_tf_graph(g)
        assert abs(loss - 0.332636844872053) < 0.00001
