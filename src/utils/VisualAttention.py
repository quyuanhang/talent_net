import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils.Trainer import feed_dict
import jieba


def feature_fold(idx, block_len):
    idx_fold = []
    for i in idx:
        idx_fold.append([
            i - block_len // 2,
            i + block_len // 2 + 1,
        ])
    return idx_fold


def highlight(word, attn):
    r, g, b = attn
    html_color = '#%02X%02X%02X' % (int(255*(1-r)), int(255*(1-g)), int(255*(1-b)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)


def mk_html(doc, attns, sep='', end="<br><br>\n"):
    html = ""
    doc = jieba.cut(doc)
    for word, attn in zip(doc, attns):
        html += highlight(
            word,
            attn,
        )
        html += sep
    return html + end


def feature_lookup(predictions, idxs, weights, datas):
    visual_str = ""
    for predict, idx, weight, data in zip(predictions, idxs, weights, datas):
        if round(predict[0]) == 0:
            continue
        jid, eid, jd, cv, label = data
        cv_attn = np.zeros(shape=(len(jd), 3))
        if label == 0:
            continue
        visual_str += "<p> jobid: {} expectid: {} </p>".format(jid, eid)
        for i, row in enumerate(idx):
            cv_idx1, cv_idx2 = row
            cv_attn[cv_idx1: cv_idx2] += 0.05
        cv = mk_html(cv, cv_attn)
        visual_str += "<br><br>\n" + cv + '\n'
        visual_str += "<p>==============================================</p>"
    return visual_str


def visual(
        sess: tf.Session,
        model,
        test_data_fn,
        raw_data_fn,
        data_len=1000000,
    ):

    predict = model.predict
    related_features = model.cv_weights
    test_data = test_data_fn()
    raw_data = raw_data_fn()
    visual_str = ""
    for i, batch in tqdm(enumerate(test_data)):
        if i > data_len:
            break
        fd = feed_dict(batch, model)
        predict_data, related_features_data = sess.run([predict, related_features], feed_dict=fd)
        feature_indexs = [range(model.doc_len - model.conv_size + 1)] * len(related_features_data)
        feature_indexs = [
            feature_fold(idx, model.conv_size)
            for idx in feature_indexs]
        batch_raw = next(raw_data)
        batch_visual_str = feature_lookup(predict_data, feature_indexs, related_features_data, batch_raw)
        visual_str += batch_visual_str
        if len(visual_str) >= data_len:
            break
    return visual_str
