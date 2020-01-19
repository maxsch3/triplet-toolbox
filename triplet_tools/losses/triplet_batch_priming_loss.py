import tensorflow as tf
import triplet_tools.losses.shared_functions as sf


def triplet_batch_priming_loss(metric='euclidean_norm'):

    def loss_function(labels, embeddings):
        d = sf.pairwise_distance(embeddings, metric)
        adj = sf.make_adj_matrix(labels, embeddings.dtype)
        hp = sf.hard_positive(d, adj)
        hn = sf.hard_negative(d, adj)
        # pull hard positives to 0
        hp = hp**2
        # push hard negatives away from 0
        hn = tf.exp(10*tf.negative(hn))
        loss = tf.reduce_mean(hp + hn)
        return loss

    return loss_function
