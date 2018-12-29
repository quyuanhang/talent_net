import tensorflow as tf


class FM:
    def __init__(self, n_feature, emb_dim=64, l2=0):
        features = tf.placeholder(dtype=tf.float32, shape=[None, n_feature], name='cate_features')
        with tf.variable_scope('linear_part'):
            linear_part = tf.layers.dense(
                inputs=features,
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
                ),
                axis=-1,
                keepdims=True,
            )
        logit = tf.add(linear_part, pair_part, name='logit')
        logit = tf.squeeze(logit, axis=-1)
        self.predict = logit
        with tf.variable_scope('loss'):
            label = tf.placeholder(dtype=tf.int32, shape=(None), name='labels')
            logit = tf.nn.sigmoid(logit)
            loss = tf.losses.log_loss(label, logit)
            if l2:
                l2_loss = sum([tf.nn.l2_loss(x) for x in tf.trainable_variables()])
                loss += l2 * l2_loss
        self.loss = loss
        self.palaceholder_names = [
            'cate_features:0',
            "loss/labels:0",
        ]


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    path = "board/test"
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)
    with tf.Session(graph=tf.Graph()) as sess:
        writer = tf.summary.FileWriter(path)
        fm = FM(100, 64, 0)
        writer.add_graph(sess.graph)
    writer.close()


