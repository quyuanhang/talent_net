import tensorflow as tf


class FM:
    def __init__(self, n_feature, emb_dim=64, l2=0):
        features = tf.placeholder(dtype=tf.int32, shape=[None, n_feature], name='input')
        with tf.variable_scope('linear_part'):
            linear_part = tf.layers.dense(
                input=features,
                units=1,
            )
        with tf.variable_scope('pair_part'):
            embs = tf.Variable(
                tf.random_normal(
                    shape=(n_feature, emb_dim),
                    mean=0,
                    stddev=0.1,
                )
            )
            pair_part = 0.5 * tf.reduce_sum(
                tf.subtract(
                    tf.pow(
                        tf.matmul(
                            features,
                            embs,
                        ),
                        2,
                    ),
                    tf.matmul(
                        tf.pow(features, 2),
                        tf.pow(embs, 2),
                    )
                )
            )
        logit = tf.add(linear_part, pair_part, name='logit')
        self.logit = logit
        with tf.variable_scope('loss'):
            label = tf.placeholder(dtype=tf.int32, shape=(None, 1), name='label')
            logit = tf.nn.sigmoid(logit)
            loss = tf.losses.log_loss(label, logit)
            if l2:
                l2_loss = sum([tf.nn.l2_loss(x) for x in tf.trainable_variables()])
                loss += l2 * l2_loss
        self.loss = loss


