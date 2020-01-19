import tensorflow as tf
import triplet_tools.losses.shared_functions as sf


def triplet_batch_hard_loss(margin=1.0, metric='euclidean_norm'):

    def loss_function(labels, embeddings):
        d = sf.pairwise_distance(embeddings, metric)
        adj = sf.make_adj_matrix(labels, embeddings.dtype)
        hp = sf.hard_positive(d, adj)
        hn = sf.hard_negative(d, adj)
        loss = tf.reduce_mean(tf.maximum(hp - hn + margin, 0.))
        return loss

    return loss_function
