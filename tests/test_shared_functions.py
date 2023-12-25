import numpy as np
import tensorflow as tf
import pytest
import triplet_tools.losses.shared_functions as sf


class TestSharedFunctions:

    emb = None
    labels = None

    def setup_method(self):
        self.emb = np.array([
            [.22, -.5, .11],
            [.3, -.66, .05],
            [.4, -.55, .2],
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

    def test_normalization(self):
        g = sf.normalize_embeddings(self.emb)
        norm = self.evaluate_tf_graph(g)
        assert norm.shape == self.emb.shape
        assert np.equal(np.sum(norm ** 2, axis=1), np.array([1., 1., 1., 1., 1.])).all()

    def test_memebership(self):
        g = sf.make_adj_matrix(self.labels, self.emb.dtype)
        assert g.dtype == tf.float64
        mem = self.evaluate_tf_graph(g)
        assert mem.shape == (self.emb.shape[0], self.emb.shape[0])
        control = np.array([
            [1., 1., 1., 0., 0.],
            [1., 1., 1., 0., 0.],
            [1., 1., 1., 0., 0.],
            [0., 0., 0., 1., 1.],
            [0., 0., 0., 1., 1.],
        ])
        assert np.equal(mem, control).all()

    def test_cosine(self):
        g = sf.cosine_dist(self.emb)
        cos = self.evaluate_tf_graph(g)
        self.check_cosine_matrix(cos)

    def test_euclidean(self):
        g = sf.euclidean_dist(self.emb)
        ed = self.evaluate_tf_graph(g)
        self.check_euclidean_matrix(ed)

    def test_euclidean_norm(self):
        g = sf.euclidean_dist_normalized(self.emb)
        ed = self.evaluate_tf_graph(g)
        self.check_euclidean_norm_matrix(ed)

    def test_hard_pos(self):
        adj = sf.make_adj_matrix(self.labels, self.emb.dtype)
        dist = sf.euclidean_dist_normalized(self.emb)
        g = sf.hard_positive(dist, adj)
        hp = self.evaluate_tf_graph(g)
        assert hp.shape == (self.emb.shape[0],)
        control = np.array([2.25034931e-01, 2.93190947e-01, 2.93190947e-01, 8.41556606e-02, 8.41556606e-02])
        assert (np.abs(hp - control) < 1e-6).all()

    def test_mask_neg(self):
        adj = sf.make_adj_matrix(self.labels, self.emb.dtype)
        dist = sf.euclidean_dist_normalized(self.emb)
        g = sf.mask_negatives(dist, adj)
        mn = self.evaluate_tf_graph(g)
        assert mn.shape == (self.emb.shape[0], self.emb.shape[0])

    def test_hard_neg(self):
        adj = sf.make_adj_matrix(self.labels, self.emb.dtype)
        dist = sf.euclidean_dist_normalized(self.emb)
        g = sf.hard_negative(dist, adj)
        hn = self.evaluate_tf_graph(g)
        assert hn.shape == (self.emb.shape[0],)
        control = np.array([1.27439449e+00, 1.27499359e+00, 1.10399029e+00, 1.10399029e+00, 1.15717809e+00])
        assert (np.abs(hn - control) < 1e-6).all()

    def test_mean_neg(self):
        adj = sf.make_adj_matrix(self.labels, self.emb.dtype)
        dist = sf.euclidean_dist_normalized(self.emb)
        g = sf.mean_negative(dist, adj, 2)
        mn = self.evaluate_tf_graph(g)
        assert mn.shape == (self.emb.shape[0],)
        control = np.array([1.3003276, 1.3031166, 1.13058419, 1.18919239, 1.24171939])
        assert (np.abs(mn - control) < 1e-6).all()

    def test_semihard(self):
        emb = np.array([
            [.22, -.5, .11],
            [.3, -.66, .05],
            [.4, -.55, .2],
            [.2, -.5, .1],
            [.72, .24, .12]
        ])
        adj = sf.make_adj_matrix(self.labels, emb.dtype)
        dist = sf.euclidean_dist_normalized(emb)
        hp = sf.hard_positive(dist, adj)
        g = sf.semihard_negative(dist, adj, hp)
        shn = self.evaluate_tf_graph(g)
        assert shn.shape == (self.emb.shape[0],)
        control = np.array([1.32626070, 1.33123960, 1.15717809, sf.high_value, sf.high_value])
        assert (np.abs(shn - control) < 1e-6).all()

    def test_pairwise_distance(self):
        g = sf.pairwise_distance(self.emb, 'cosine')
        d = self.evaluate_tf_graph(g)
        self.check_cosine_matrix(d)
        g = sf.pairwise_distance(self.emb, 'euclidean')
        d = self.evaluate_tf_graph(g)
        self.check_euclidean_matrix(d)
        g = sf.pairwise_distance(self.emb, 'euclidean_norm')
        d = self.evaluate_tf_graph(g)
        self.check_euclidean_norm_matrix(d)
        with pytest.raises(ValueError):
            g = sf.pairwise_distance(self.emb, 'Non-supported')
            d = self.evaluate_tf_graph(g)

    def check_cosine_matrix(self, cos):
        assert cos.shape == (self.emb.shape[0], self.emb.shape[0])
        control = np.array([
             [0.00000000e+00, 8.49099331e-03, 2.53203600e-02, 8.12040663e-01, 8.79483718e-01],
             [8.49099331e-03, 1.11022302e-16, 4.29804658e-02, 8.12804323e-01, 8.86099440e-01],
             [2.53203600e-02, 4.29804658e-02, 1.11022302e-16, 6.09397285e-01, 6.69530565e-01],
             [8.12040663e-01, 8.12804323e-01, 6.09397285e-01, 0.00000000e+00, 3.54108761e-03],
             [8.79483718e-01, 8.86099440e-01, 6.69530565e-01, 3.54108761e-03, 0.00000000e+00]
        ])
        assert (np.abs(cos - control) < 1e-6).all()

    def check_euclidean_matrix(self, ed):
        assert ed.shape == (self.emb.shape[0], self.emb.shape[0])
        control = np.array([
            [0., 0.18867962, 0.20736441, 0.90912045, 0.89314053],
            [0.18867962, 0., 0.21118712, 0.99604217, 0.9956405],
            [0.20736441, 0.21118712, 0., 0.85586214, 0.85609579],
            [0.90912045, 0.99604217, 0.85586214, 0., 0.09165151],
            [0.89314053, 0.9956405, 0.85609579, 0.09165151, 0.]
        ])
        assert (np.abs(ed - control) < 1e-6).all()

    def check_euclidean_norm_matrix(self, ed):
        assert ed.shape == (self.emb.shape[0], self.emb.shape[0])
        control = np.array([
            [0.00000000e+00, 1.30314952e-01, 2.25034931e-01, 1.27439449e+00, 1.32626070e+00],
            [1.30314952e-01, 1.49011612e-08, 2.93190947e-01, 1.27499359e+00, 1.33123960e+00],
            [2.25034931e-01, 2.93190947e-01, 1.49011612e-08, 1.10399029e+00, 1.15717809e+00],
            [1.27439449e+00, 1.27499359e+00, 1.10399029e+00, 0.00000000e+00, 8.41556606e-02],
            [1.32626070e+00, 1.33123960e+00, 1.15717809e+00, 8.41556606e-02, 0.00000000e+00]
        ])
        assert (np.abs(ed - control) < 1e-6).all()
