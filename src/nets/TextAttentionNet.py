import sys
import tensorflow as tf
from tensorflow import keras


class TextAttentionNet:
    def __init__(
            self,
            doc_len,
            feature_len,
            emb_dim,
            n_feature,
            n_word,
            conv_size,
            emb_pretrain=[],
            l2=0,
            mode='cross',
            top_k=3,
            dropout=0.0,
    ):

        self.doc_len = doc_len
        self.conv_size = conv_size
        self.emb_dim = emb_dim
        self.feature_len = feature_len
        self.n_features = n_feature
        self.n_word = n_word
        self.l2 = l2
        self.top_k = top_k
        self.training = tf.placeholder(dtype=tf.bool, name='training')
        self.related_features = None

        if len(emb_pretrain) > 0:
            def myinit(*args, **kwargs):
                return tf.convert_to_tensor(emb_pretrain, dtype=tf.float32)
            self.emb_init = myinit
        else:
            self.emb_init = 'RandomNormal'

        cate_features = tf.placeholder(dtype=tf.int32, shape=(None, feature_len), name='cate_feature_ids')
        cate_features = self.feature_emb(cate_features, flatten=False)
        jd_id = cate_features[:, 1, :]
        cv_id = cate_features[:, 0, :]

        with tf.variable_scope('jds'):
            jd = tf.placeholder(dtype=tf.int32, shape=(None, doc_len), name='doc')
        with tf.variable_scope('cvs'):
            cv = tf.placeholder(dtype=tf.int32, shape=(None, doc_len), name='doc')

        with tf.variable_scope('jd_cnn'):
            jd = self.feature_emb(jd, mode='word', init=self.emb_init, flatten=False)
            jd = self.cnn(jd)

        with tf.variable_scope('cv_cnn'):
            cv = self.feature_emb(cv, mode='word', init=self.emb_init, flatten=False)
            cv = self.cnn(cv)

        if mode == "attention":
            with tf.variable_scope("attention"):
                self.jd_weights, wjd = self.attention(cv_id, jd)
                self.cv_weights, wcv = self.attention(jd_id, cv)
            # features = tf.concat([jd_id, cv_id, wjd, wcv], axis=1)
            jd = tf.reduce_max(jd, axis=1)
            cv = tf.reduce_max(cv, axis=1)
            features = tf.concat([jd_id, cv_id, jd, cv, wjd, wcv], axis=1)
        elif mode == "doc_id":
            jd = tf.reduce_max(jd, axis=1)
            cv = tf.reduce_max(cv, axis=1)
            features = tf.concat([jd_id, cv_id, jd, cv], axis=1)
        elif mode == "id":
            features = tf.concat([jd_id, cv_id], axis=1)
        elif mode == "doc":
            jd = tf.reduce_max(jd, axis=1)
            cv = tf.reduce_max(cv, axis=1)
            features = tf.concat([jd, cv], axis=1)
        else:
            print("unknow mode")
            sys.exit()

        with tf.variable_scope('classifier'):
            self.predict = tf.nn.sigmoid(
                self.mlp(features, dropout))

        with tf.variable_scope('loss'):
            self.label = tf.placeholder(dtype=tf.int32, shape=None, name='labels')
            self.loss = self.loss_function()

        self.palaceholder_names = [
            "jds/doc:0",
            "cvs/doc:0",
            "cate_feature_ids:0",
            "loss/labels:0",
            "training:0",
        ]

    @staticmethod
    def attention(query: tf.Tensor, features: tf.Tensor):
        """
        :param query: 2d array, batch_size *  emb_dim
        :param features: 3d array, batch_size * n_feature * emb_dim
        :return: 2d array
        """
        d = query.shape.as_list()[-1]
        query = tf.expand_dims(query, axis=1)
        features_weights = tf.divide(
            tf.matmul(query, features, transpose_b=True),
            tf.sqrt(float(d)))
        features_weights = tf.nn.softmax(features_weights, axis=-1)
        weighted_features = tf.matmul(features_weights, features)
        weighted_features = tf.squeeze(weighted_features, axis=1)

        return features_weights, weighted_features

    def feature_emb(self, cate, mode='cate', init='RandomNormal', flatten=True, name=None):
        if mode == 'cate':
            input_dim = self.n_features
        else:
            input_dim = self.n_word
        cate = keras.layers.Embedding(
            input_dim=input_dim,
            output_dim=self.emb_dim,
            embeddings_initializer=init,
            name=name,
        )(cate)
        if flatten:
            cate = tf.layers.flatten(cate)
        return cate

    def cnn(self, x: tf.Tensor):
        x = tf.expand_dims(x, axis=-1)
        x = tf.layers.conv2d(
            inputs=x,
            filters=1,
            kernel_size=[self.conv_size, 1],
            strides=[1, 1],
            padding='valid',
            activation=tf.nn.relu,
            name='cnn',
        )
        x = tf.squeeze(x, axis=3)
        return x

    def sentence_cnn(self, x: tf.Tensor):
        x = tf.map_fn(self.cnn, x)
        x = tf.reduce_max(x, axis=-2)
        return x

    def mlp(self, features, dropout=0):
        features = tf.layers.batch_normalization(features)
        if dropout:
            features = tf.layers.dropout(
                inputs=features,
                rate=dropout,
                training=self.training,
            )
        features = tf.layers.dense(
            features,
            units=self.emb_dim * 2,
            activation=tf.nn.relu,
        )
        features = tf.layers.batch_normalization(features)
        if dropout:
            features = tf.layers.dropout(
                inputs=features,
                rate=dropout,
                training=self.training,
            )
        features = tf.layers.dense(
            features,
            units=self.emb_dim,
            activation=tf.nn.relu,
        )
        features = tf.layers.batch_normalization(features)
        predict = tf.layers.dense(
            features,
            units=1,
            # activation=tf.nn.sigmoid,
            name='deep',
        )
        return predict

    def lr(self, features):
        predict = tf.layers.dense(
            features,
            units=1,
            # activation=tf.nn.sigmoid,
            name='wide',
        )
        return predict

    def classifier(self, features, dropout=0):
        wide = self.lr(features)
        deep = self.mlp(features, dropout=dropout)
        features = tf.concat([wide, deep], axis=1)
        predict = tf.layers.dense(
            features,
            units=1,
            activation=tf.nn.sigmoid,
            name='output'
        )
        return predict

    def classifier2(self, wfeatures, dfeatures, dropout=0):
        wide = self.lr(wfeatures)
        deep = self.mlp(dfeatures, dropout=dropout)
        features = tf.concat([wide, deep], axis=1)
        predict = tf.layers.dense(
            features,
            units=1,
            activation=tf.nn.sigmoid,
            name='output'
        )
        return predict

    def classifier3(self, wfeatures, dfeatures, dropout=0):
        wide = self.lr(wfeatures)
        deep = self.mlp(dfeatures, dropout=dropout)
        predict = tf.nn.sigmoid(tf.add(wide, deep))
        return predict

    def loss_function(self):
        predict = tf.squeeze(self.predict)
        loss = tf.losses.log_loss(self.label, predict)
        if self.l2:
            l2_loss = sum([tf.nn.l2_loss(x) for x in tf.trainable_variables()])
            loss = loss + l2_loss * self.l2
        return loss


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    path = "board/test"
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)
    with tf.Session(graph=tf.Graph()) as sess:
        writer = tf.summary.FileWriter(path)
        text_cross_net = TextAttentionNet(500, 10, 64, 2000, 5000, 3, mode="attention", dropout=0.3)
        writer.add_graph(sess.graph)
    writer.close()
