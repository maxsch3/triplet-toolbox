import tensorflow as tf


high_value = 1e16


def normalize_embeddings(x):
    return tf.linalg.l2_normalize(x, axis=1)


def make_adj_matrix(labels, dtype):
    lm = tf.cond(tf.equal(tf.rank(labels), 1), lambda: tf.expand_dims(labels, -1), lambda: labels)
    mem = tf.equal(lm, tf.transpose(lm))
    mem = tf.cast(mem, dtype)
    return mem


def make_adj_matrix1(labels, dtype):
    l, i = tf.unique(tf.squeeze(labels))
    onehot = tf.one_hot(i, depth=tf.size(l), dtype=dtype)
    mem = tf.matmul(onehot, tf.transpose(onehot))
    return mem


def cosine_dist(x):
    vn = normalize_embeddings(x)
    return 1. - tf.matmul(vn, tf.transpose(vn))


def euclidean_dist_normalized(x):
    cos = tf.maximum(cosine_dist(x), 1e-16)
    return tf.sqrt(2. * cos)


def euclidean_dist(x):
    x_l2 = tf.reduce_sum(x**2, axis=1, keepdims=True)
    d_sq = tf.math.add(x_l2, tf.transpose(x_l2)) - 2. * tf.matmul(x, tf.transpose(x))
    d_sq = tf.maximum(d_sq, 1e-16)
    return tf.sqrt(d_sq)


def hard_positive(x, adj):
    return tf.reduce_max(x * adj, axis=1)


def mask_negatives(x, adj):
    """
        returns a tensor of the same shape as x with all positive pair distances replaced with the
        biggest number in the tensor. This way, positive pairs will not be selected in reduce_min operation
    """
    # TODO: should not use this trick as it might introduce wrong gradients
    # choices = tf.reduce_sum(adj, axis=1)
    # # If below assertion fails, this means there was no negatives to select
    # assert_op = tf.Assert(tf.reduce_all(choices > 0.5), [adj])
    # return x * (1. - adj) + tf.reduce_max(x) * adj
    return x * (1. - adj) + high_value * adj

def hard_negative(x, adj):
    # tf.reduce_max(x) * x makes main diagonal very high, which is otherwize 0
    return tf.reduce_min(mask_negatives(x, adj), axis=1)


def mean_negative(x, adj, k=1):
    neg = mask_negatives(x, adj)
    neg, _ = tf.math.top_k(tf.negative(neg), k=k)
    neg = tf.negative(neg)
    mean_neg = tf.reduce_mean(neg, axis=-1)
    return mean_neg


def semihard_negative(x, adj, hard_pos, semi_margin=.0):
    hard_pos_2d = tf.expand_dims(hard_pos, axis=1)
    # mask_ij = True if a distance between i and j is longer than hard positive for i
    # other words all negatives that have mask = True are semi hard negatives and should be considered
    mask = x > hard_pos_2d + semi_margin
    # adj_ij = 1 for positive pairs and 0 for negative ones. We now need to pick those negatives that
    # have mask = 1
    mask = tf.cast((adj < 0.5) & mask, dtype=x.dtype)
    # TODO: in an unlucky event of really bad hard positive, there will be no hard negatives to select
    # TODO: How to deal with it? As of now, a high constant is selected (meaning that negative is very far
    # TODO: anf thus not making any gradients)
    return tf.reduce_min(mask_negatives(x, 1.-mask), axis=1)


def pairwise_distance(x, metric):
    if metric == 'euclidean':
        return euclidean_dist(x)
    elif metric == 'euclidean_norm':
        return euclidean_dist_normalized(x)
    elif metric == 'cosine':
        return cosine_dist(x)
    else:
        raise ValueError('Error: metric {} is not supported.'.format(metric))
