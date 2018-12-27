import tensorflow as tf
from tensorflow import keras


class TextCrossNet:
    def __init__(
            self,
            doc_len,
            n_skill,
            skill_len,
            n_kws,
            feature_len,
            emb_dim,
            n_feature,
            n_word,
            conv_size,
            emb_pretrain=[],
            l2=0,
            mode='cross',
            top_k=3,
            cate_emb='diag',
            dropout=0,
            global_emb=0,
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

        with tf.variable_scope('jds'):
            jd = tf.placeholder(dtype=tf.int32, shape=(None, doc_len), name='doc')
            jd_skill = tf.placeholder(dtype=tf.int32, shape=(None, n_skill, skill_len), name='skill')
            jd_kws = tf.placeholder(dtype=tf.int32, shape=(None, n_kws), name='keywords')
        with tf.variable_scope('cvs'):
            cv = tf.placeholder(dtype=tf.int32, shape=(None, doc_len), name='doc')
            cv_skill = tf.placeholder(dtype=tf.int32, shape=(None, n_skill, skill_len), name='skill')
            cv_kws = tf.placeholder(dtype=tf.int32, shape=(None, n_kws), name='keywords')
        cate_features = tf.placeholder(dtype=tf.int32, shape=(None, feature_len), name='cate_features')

        if mode == 'cate_only':
            cate_features = self.feature_emb(cate_features)
        else:
            with tf.variable_scope('jd_cnn'):
                jd = self.text_cnn(jd, global_emb=global_emb)
            with tf.variable_scope('cv_cnn'):
                cv = self.text_cnn(cv, global_emb=global_emb)
            with tf.variable_scope('jd_cnn'):
                jd1 = self.feature_emb(jd, mode='word', init=self.emb_init, flatten=False, name="jd_emb")
                jd1 = self.sentence_cnn(jd1)
            with tf.variable_scope('cv_cnn'):
                cv1 = self.feature_emb(cv, mode='word', init=self.emb_init, flatten=False, name="cv_emb")
                cv1 = self.sentence_cnn(cv1)

            if mode == 'text_concat':
                features = self.concat_reduce(jd1, cv1)
            if mode == 'text_cross':
                features = self.feature_emb(features)
                features_ = self.matmul_flatten(jd1, cv1)
                features = tf.concat([features, features_], axis=1)
            if mode == 'cross':
                features = self.cross(jd1, cv1, features, cate_emb)
            if mode in ('concat', 'deep'):
                features = self.concat(jd1, cv1, features)
            if mode == 'wd':
                with tf.variable_scope('deep'):
                    deep_features = self.concat(jd1, cv1, features)
                with tf.variable_scope('wide'):
                    wide_features = self.cross(jd1, cv1, features)
            # if mode == 'wd2':
            #     with tf.variable_scope('wide'):
            #         wide_features = self.cross(jd1, cv1, features)
            #     with tf.variable_scope('deep'):
            #         deep_features = tf.concat((jd2, cv2), axis=1)
            #         features = self.feature_emb(features)
            #         deep_features = tf.concat((deep_features, features), axis=1)

        with tf.variable_scope('classifier'):
            if mode in ("wd", "wd2"):
                self.predict = self.classifier2(wide_features, deep_features, dropout)
            # elif mode == 'cross':
            #     self.predict = tf.nn.sigmoid(
            #         self.lr(features))
            elif mode == 'deep':
                self.predict = tf.nn.sigmoid(
                    self.mlp(features, dropout))
            else:
                self.predict = self.classifier2(features, features, dropout)
        with tf.variable_scope('loss'):
            self.label = tf.placeholder(dtype=tf.int32, shape=None, name='labels')
            self.loss = self.loss_function()

    @staticmethod
    def concat_reduce(jd, cv):
        features = tf.concat([jd, cv], axis=2)
        features = tf.concat(
            (
                tf.reduce_mean(features, axis=1),
                tf.reduce_max(features, axis=1),
            ),
            axis=-1,
        )
        # features = tf.reduce_mean(features, axis=1)
        return features

    @staticmethod
    def matmul_flatten(jd, cv):
        cross = tf.matmul(jd, cv, transpose_b=True)
        cross = tf.layers.flatten(cross)
        return cross

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

    def concat(self, jd, cv, cate):
        features = self.concat_reduce(jd, cv)
        cate = self.feature_emb(cate)
        features = tf.concat([features, cate], axis=1)
        return features

    def text_cnn(self, x: tf.Tensor, global_emb=False):
        if global_emb:
            x = self.word_emb(x)
        else:
            x = keras.layers.Embedding(
                    input_dim=self.n_word,
                    output_dim=self.emb_dim,
                    embeddings_initializer=self.emb_init,
            )(x)
        x = tf.expand_dims(x, axis=-1)
        x = tf.layers.conv2d(
            inputs=x,
            filters=1,
            kernel_size=[self.block_len, 1],
            strides=[3, 1],
            padding='valid',
            activation=tf.nn.relu,
            name='cnn',
        )
        x = tf.squeeze(x, axis=3)
        return x

    def sentence_cnn(self, x: tf.Tensor):
        x = tf.reshape(x, shape=(-1, x.shape.as_list()[-2], x.shape.as_list()[-1], 1))
        x = tf.layers.conv2d(
            inputs=x,
            filters=(self.emb_dim // 2),
            kernel_size=[self.block_len, self.emb_dim],
            strides=[1, 1],
            padding='valid',
            activation=tf.nn.relu,
            name='cnn',
        )
        x = tf.reshape(x, shape=(-1, self.doc_len, x.shape.as_list()[-3], x.shape.as_list()[-1]))
        # x = tf.concat(
        #     (tf.reduce_max(x, axis=-2), tf.reduce_mean(x, axis=-2)),
        #     axis=-1)
        x = tf.reduce_max(x, axis=-2)
        return x

    def sentence_cnn2(self, x: tf.Tensor):
        x = tf.transpose(x, perm=(0, 2, 3, 1))
        x = tf.layers.conv2d(
            inputs=x,
            filters=self.emb_dim,
            kernel_size=[self.block_len, self.emb_dim],
            strides=[1, 1],
            padding='valid',
            activation=tf.nn.relu,
            name='cnn2',
        )
        x = tf.concat(
            (tf.reduce_max(x, axis=(1,2)), tf.reduce_mean(x, axis=(1,2))),
            axis=-1)
        return x

    def cross(self, jd: tf.Tensor, cv: tf.Tensor, cate: tf.Tensor, cate_emb='diag'):
        emb_dim = jd.shape.as_list()[-1]
        if cate_emb == "full":
            cate_emb_dim = emb_dim ** 2
        else:
            cate_emb_dim = emb_dim
        cate = keras.layers.Embedding(
            input_dim=self.n_features,
            output_dim=cate_emb_dim,
            embeddings_initializer='RandomNormal',
            name='cate_embedding'
        )(cate)
        if cate_emb == 'diag':
            cate = tf.matrix_diag(cate)
        else:
            cate = tf.reshape(cate, shape=[-1, self.feature_len, emb_dim, emb_dim])
        cate = tf.reduce_mean(cate, axis=1)
        cross = tf.matmul(jd, cate)
        cross = tf.matmul(cross, cv, transpose_b=True)
        cross = tf.layers.flatten(cross)
        cross = tf.nn.softmax(cross)
        self.related_features = tf.nn.top_k(cross, k=self.top_k)
        return cross

    # def cross(self, jd: tf.Tensor, cv: tf.Tensor, cate: tf.Tensor, cate_emb='diag'):
    #     emb_dim = jd.shape.as_list()[-1]
    #     if cate_emb == "full":
    #         cate_emb_dim = emb_dim ** 2
    #     else:
    #         cate_emb_dim = emb_dim
    #     cate = keras.layers.Embedding(
    #         input_dim=self.n_features,
    #         output_dim=cate_emb_dim,
    #         embeddings_initializer='RandomNormal',
    #         name='cate_embedding'
    #     )(cate)
    #     if cate_emb == 'diag':
    #         cate = tf.matrix_diag(cate)
    #     else:
    #         cate = tf.reshape(cate, shape=[-1, self.feature_len, emb_dim, emb_dim])
    #
    #     jd = tf.tile(
    #         tf.expand_dims(jd, axis=1),
    #         multiples=(1, self.feature_len, 1, 1),
    #     )
    #     cv = tf.tile(
    #         tf.expand_dims(cv, axis=1),
    #         multiples=(1, self.feature_len, 1, 1)
    #     )
    #     cross = tf.matmul(jd, cate)
    #     cross = tf.matmul(cross, cv, transpose_b=True)
    #     cross = tf.reduce_mean(cross, axis=1)
    #     cross = tf.layers.flatten(cross)
    #     cross = tf.nn.softmax(cross)
    #     self.related_features = tf.nn.top_k(cross, k=self.top_k)
    #     return cross

    @staticmethod
    def cross_with_feature(feature_mat, jd, cv):
        cross = tf.matmul(jd, feature_mat)  # b d e
        cross = tf.transpose(cross, perm=(0, 2, 1))  # b e d
        cross = tf.expand_dims(cross, axis=-1)  # b e d 1
        cv = tf.transpose(cv, perm=(0, 2, 1))  # b e d
        cv = tf.expand_dims(cv, axis=-1)  # b e d 1
        cross = tf.matmul(cross, cv, transpose_b=True)  # b e d d
        cross_mean = tf.reduce_mean(cross, axis=(2, 3))  # b e
        cross_max = tf.reduce_max(cross, axis=(2, 3))  # b e
        cross = tf.concat((cross_max, cross_mean), axis=1)
        return cross

    def cross_to_emb(self, jd: tf.Tensor, cv: tf.Tensor, cate: tf.Tensor, cate_emb='diag'):
        if cate_emb == 'diag':
            emb_dim = self.emb_dim
        else:
            emb_dim = self.emb_dim ** 2
        cate_flatten = self.feature_emb(cate)
        cate = keras.layers.Embedding(
            input_dim=self.n_features,
            output_dim=emb_dim,
            embeddings_initializer='RandomNormal',
            name='cate_embedding'
        )(cate)
        if cate_emb == 'diag':
            cate = tf.matrix_diag(cate)
        else:
            cate = tf.reshape(cate, shape=[-1, self.feature_len, self.emb_dim, self.emb_dim])
        cate = tf.transpose(cate, perm=(1, 0, 2, 3))
        cross = tf.map_fn(
            lambda x: self.cross_with_feature(x, jd, cv),
            cate,
        )
        cross = tf.transpose(cross, perm=(1, 0, 2))
        cross = tf.layers.flatten(cross)
        cross = tf.concat([cross, cate_flatten], axis=1)
        cross = tf.nn.softmax(cross)
        return cross

    def get_related_features(self):
        return

    def mlp(self, features, dropout=0):
        # features = tf.layers.batch_normalization(features)
        # if dropout:
        #     features = tf.layers.dropout(
        #         inputs=features,
        #         rate=dropout,
        #         training=self.training,
        #     )
        # features = tf.layers.dense(
        #     features,
        #     units=self.emb_dim * 2,
        #     activation=tf.nn.relu,
        # )
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
        text_cross_net = TextCrossNet(50, 50, 5, 15, 128, 5000, 20000, mode='wd', dropout=0.3)
        writer.add_graph(sess.graph)
    writer.close()
