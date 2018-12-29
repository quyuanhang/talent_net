import numpy as np
import tensorflow as tf
from nets.TextCrossNet import TextCrossNet
from utils.MixData import MixData
from tqdm import tqdm
from sklearn import metrics


def feed_dict(data, model, training=True):
    labels, jds, jd_kws, jd_skill, cvs, cv_kws, cv_skill, cate_features, cate_feature_ids = data
    fd = {
        "jds/doc:0": jds,
        "jds/skill:0": jd_skill,
        "jds/keywords:0": jd_kws,
        "cvs/doc:0": cvs,
        "cvs/skill:0": cv_skill,
        "cvs/keywords:0": cv_kws,
        "cate_features:0": cate_features,
        "cate_feature_ids:0": cate_feature_ids,
        "loss/labels:0": labels,
        "training:0": training,
    }
    fd = {k: fd[k] for k in model.palaceholder_names}
    if "cate_features:0" in fd:
        fd["cate_features:0"] = np.array([x.todense() for x in fd["cate_features:0"]]).squeeze(axis=1)
    return fd


def train(
        sess: tf.Session,
        model,
        writer,
        train_data_fn,
        test_data_fn,
        lr=0.0005,
        n_epoch=100,
    ):

    predict = model.predict

    loss = model.loss

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)

    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epoch):
        epoch_loss = []
        epoch_metric_outside = []
        train_data = train_data_fn()
        for batch in tqdm(train_data, ncols=50):
            fd = feed_dict(batch, model, training=True)
            predict_data, loss_data, _ = sess.run(
                [predict, loss, train_op],
                feed_dict=fd)
            epoch_loss.append(loss_data)
            outside_auc_data = metrics.roc_auc_score(fd['loss/labels:0'], predict_data)
            epoch_metric_outside.append(outside_auc_data)
        val_loss = []
        val_metric_outside = []
        test_data = test_data_fn()
        for batch in tqdm(test_data):
            fd = feed_dict(batch, model, training=False)
            predict_data, loss_data = sess.run(
                [predict, loss],
                feed_dict=fd)
            val_loss.append(loss_data)
            outside_auc_data = metrics.roc_auc_score(fd['loss/labels:0'], predict_data)
            val_metric_outside.append(outside_auc_data)
        print(
            'epoch: {}\n'.format(epoch),
            'train loss: {:.3f} train metric: {:.3f}\n'.format(
                np.array(epoch_loss).mean(),
                np.array(epoch_metric_outside).mean()),
            'valid loss: {:.3f} valid metric: {:.3f}\n'.format(
                np.array(val_loss).mean(),
                np.array(val_metric_outside).mean()),
        )

