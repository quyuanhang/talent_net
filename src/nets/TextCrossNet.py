import tensorflow as tf
from tensorflow import keras


class TextCrossNet:
    def __init__(
            self,
            doc_len,
            n_skill,
            skill_len,
            n_keywords,
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

        with tf.variable_scope('jds'):
            jd = tf.placeholder(dtype=tf.int32, shape=(None, doc_len), name='doc')
            jd_skill = tf.placeholder(dtype=tf.int32, shape=(None, n_skill, skill_len), name='skill')
            jd_keywords = tf.placeholder(dtype=tf.int32, shape=(None, n_keywords), name='keywords')
        with tf.variable_scope('cvs'):
            cv = tf.placeholder(dtype=tf.int32, shape=(None, doc_len), name='doc')
            cv_skill = tf.placeholder(dtype=tf.int32, shape=(None, n_skill, skill_len), name='skill')
            cv_keywords = tf.placeholder(dtype=tf.int32, shape=(None, n_keywords), name='keywords')

        cate_features = tf.placeholder(dtype=tf.int32, shape=(None, feature_len), name='cate_feature_ids')
        cate_features = self.feature_emb(cate_features)

        with tf.variable_scope('jd_cnn'):
            jd = self.feature_emb(jd, mode='word', init=self.emb_init, flatten=False)
            jd = self.cnn(jd)
        with tf.variable_scope('jd_skill'):
            jd_skill = self.feature_emb(jd_skill, mode='word', init=self.emb_init, flatten=False)
            jd_skill = self.sentence_cnn(jd_skill)
        with tf.variable_scope("jd_keywords"):
            jd_keywords = self.feature_emb(jd_keywords, mode="word", init=self.emb_init, flatten=False)
        with tf.variable_scope('cv_cnn'):
            cv = self.feature_emb(cv, mode='word', init=self.emb_init, flatten=False)
            cv = self.cnn(cv)
        with tf.variable_scope('cv_skill'):
            cv_skill = self.feature_emb(cv_skill, mode='word', init=self.emb_init, flatten=False)
            cv_skill = self.sentence_cnn(cv_skill)
        with tf.variable_scope("cv_keywords"):
            cv_keywords = self.feature_emb(cv_keywords, mode="word", init=self.emb_init, flatten=False)

        if mode == "category":
            features = cate_features
        if "concat" in mode:
            jd = tf.reduce_max(jd, axis=-2)
            jd_skill = tf.reduce_mean(jd_skill, axis=-2)
            jd_keywords = tf.reduce_mean(jd_keywords, axis=-2)
            cv = tf.reduce_max(cv, axis=-2)
            cv_skill = tf.reduce_mean(cv_skill, axis=-2)
            cv_keywords = tf.reduce_mean(cv_keywords, axis=-2)
            if mode == 'text_concat':
                features = tf.concat([jd, cv, cate_features], axis=1)
            if mode == "talent_concat":
                features = tf.concat([jd_skill, cv_skill], axis=1)
            if mode == "keywords_concat":
                features = tf.concat([jd_keywords, cv_keywords], axis=1)
            if mode == "all_concat":
                features = tf.concat([jd, jd_skill, jd_keywords, cv, cv_skill, cv_keywords, cate_features], axis=1)
        if mode == "attention":
            jd = tf.reduce_max(jd, axis=-2)
            jd_keywords = tf.reduce_mean(jd_keywords, axis=-2)
            cv = tf.reduce_max(cv, axis=-2)
            cv_keywords = tf.reduce_mean(cv_keywords, axis=-2)
            skill_att = self.attention(jd_skill, cv_skill)
            jd_skill = tf.reduce_mean(jd_skill, axis=-2)
            cv_skill = tf.reduce_mean(cv_skill, axis=-2)
            features = tf.concat(
                [jd, jd_skill, jd_keywords, cv, cv_skill, cv_keywords, cate_features, skill_att],
                axis=1)
        # if mode == "explain":
        #     jd = tf.reduce_max(jd, axis=-2)
        #     jd_keywords = tf.reduce_mean(jd_keywords, axis=-2)
        #     cv = tf.reduce_max(cv, axis=-2)
        #     cv_keywords = tf.reduce_mean(cv_keywords, axis=-2)

        with tf.variable_scope('classifier'):
            self.predict = tf.nn.sigmoid(
                self.mlp(features, dropout))

        with tf.variable_scope('loss'):
            self.label = tf.placeholder(dtype=tf.int32, shape=None, name='labels')
            self.loss = self.loss_function()

        self.palaceholder_names = [
            "jds/doc:0",
            "jds/skill:0",
            "jds/keywords:0",
            "cvs/doc:0",
            "cvs/skill:0",
            "cvs/keywords:0",
            "cate_feature_ids:0",
            "loss/labels:0",
            "training:0",
        ]

    def attention(self, jd_skills, cv_skills):
        """
        :param cv_skills: 3d array, batch_size * doc_len * emb_dim
        :param jd_skills: 3d array, batch_size * doc_len * emb_dim
        :return:
        """

        jd_query = tf.reduce_mean(jd_skills, axis=1, keepdims=True)
        cv_weights = tf.matmul(jd_query, cv_skills, transpose_b=True)
        cv_weights = tf.nn.softmax(cv_weights, axis=-1)
        weighted_cv = tf.matmul(cv_weights, cv_skills)
        weighted_cv = tf.squeeze(weighted_cv, axis=1)

        cv_query = tf.reduce_mean(cv_skills, axis=1, keepdims=True)
        jd_weights = tf.nn.softmax(
            tf.matmul(cv_query, jd_skills, transpose_b=True),
            axis=-1)
        weighted_jd = tf.squeeze(
            tf.matmul(jd_weights, jd_skills),
            axis=1
        )

        features = tf.concat(
            [weighted_jd, weighted_cv],
            axis=1
        )

        return features

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

    @staticmethod
    def reduce_max_mean(x, axis=None):
        x = tf.concat(
            (tf.reduce_max(x, axis=axis), tf.reduce_mean(x, axis=axis)),
            axis=-1)
        return x

    def sentence_cnn(self, x: tf.Tensor):
        x = tf.map_fn(self.cnn, x)
        x = tf.reduce_max(x, axis=-2)
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

    def get_related_features(self):
        return

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
        text_cross_net = TextCrossNet(500, 10, 10, 10, 12, 64, 2000, 5000, 3, mode="attention", dropout=0.3)
        writer.add_graph(sess.graph)
    writer.close()
