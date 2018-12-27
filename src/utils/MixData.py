import re
import requests
import collections
import numpy as np
from tqdm import tqdm
from sklearn import feature_extraction
from scipy import sparse


class MixData:
    def __init__(self, fpin, fpout, wfreq, doc_len, n_skill=10, skill_len=10, n_keywords=10, emb_dim=64, load_emb=0):
        self.fpin = fpin
        self.fpout = fpout
        self.doc_len = doc_len
        self.n_skill = n_skill
        self.skill_len = skill_len
        self.n_keywords = n_keywords

        exp_features_names = [
            'expect_id',
            'geek_id',
            'position',  # l3_name
            'city',
            'gender',
            'degree',
            'fresh_graduate',
            'apply_status',
            'completion',
            'doc',
            'doc_token',
            'skills',
            'keywords',
        ]
        job_features_names = [
            'job_id',
            'boss_id',
            'position',
            'city',
            'degree',
            'experience',
            'area_business_name',
            'boss_title',
            'is_hr',
            'stage',
            'doc',
            'doc_token',
            'skills',
            'keywords',
        ]

        require_exp_feature = [
            'position',
            'city',
            'gender',
            'degree',
            'fresh_graduate',
        ]
        require_job_feature = [
            'position',
            'city',
            'degree',
            'experience',
            'boss_title',
            'is_hr',
            'stage',
        ]

        if load_emb:
            self.word_dict, self.embs = self.load_embedding(
                "{}.word_emb".format(fpout), emb_dim)
        else:
            fps = [
                '{}.profile.job'.format(fpin),
                '{}.profile.expect'.format(fpin)]
            self.word_dict = self.build_dict(fps, wfreq)
            self.embs = np.random.normal(size=[len(self.word_dict), emb_dim])
            self.embs[:2] = 0

        self.feature_name = require_exp_feature + require_job_feature
        self.exp_to_row, self.exp_features, exp_features_names_sparse, self.exp_docs, \
            self.exp_kws, self.exp_skills, self.exp_doc_raw = self.build_features(
                fp='{}.profile.expect'.format(fpin),
                feature_name=exp_features_names,
                requir_feature_name=require_exp_feature,
            )
        self.job_to_row, self.job_features, job_features_names_sparse, self.job_docs, \
            self.job_kws, self.job_skills, self.job_doc_raw = self.build_features(
                fp='{}.profile.job'.format(fpin),
                feature_name=job_features_names,
                requir_feature_name=require_job_feature,
            )
        self.feature_name_sparse = job_features_names_sparse + exp_features_names_sparse
        print("num of raw feature: {}\nnum of sparse feature: {}".format(
            len(self.feature_name),
            len(self.feature_name_sparse)))

    @staticmethod
    def load_embedding(fp, emb_dim):
        words = ['__pad__', '__unk__']
        embs = [[0] * emb_dim] * 2
        with open(fp) as f:
            print('loading embs ...')
            for line in tqdm(f):
                data = line.strip().split()
                if len(data) != emb_dim + 1:
                    continue
                word = data[0]
                emb = [float(x) for x in data[1:]]
                words.append(word)
                embs.append(emb)
        word_dict = {k: v for v, k in enumerate(words)}
        embs = np.array(embs, dtype=np.float32)
        return word_dict, embs

    @staticmethod
    def build_dict(fps, w_freq):
        words = []
        for fp in fps:
            with open(fp) as f:
                for line in tqdm(f):
                    line = line.strip().split('\001')[-3:]
                    line = '\t'.join(line)
                    line = re.split("[ \t]", line)
                    words.extend(line)
        words_freq = collections.Counter(words)
        word_list = [k for k, v in words_freq.items() if v >= w_freq]
        word_list = ['__pad__', '__unk__'] + word_list
        word_dict = {k: v for v, k in enumerate(word_list)}
        print('n_words: {}'.format(len(word_dict)), len(word_list))
        return word_dict

    def build_features(self, fp, feature_name, requir_feature_name, one_hot=True):
        n_feature = len(feature_name)
        with open(fp) as f:
            data = f.read().strip().split('\n')
        print('split raw data ...')
        data_list = []
        for line in tqdm(data):
            features = line.split('\001')
            if len(features) != n_feature:
                continue
            # id
            cid = features[0]
            # category feature
            features_dict = dict(zip(feature_name, features))
            cate_features_dict = {k: features_dict[k] for k in requir_feature_name}
            # text feature
            words = features_dict["doc_token"].strip()
            word_ids = self.doc1d(words, self.doc_len)
            # skills
            skills = features_dict['skills']
            skills = self.skill(skills)
            # keywords
            keywords = features_dict['keywords']
            keywords = self.doc1d(keywords, self.n_keywords)
            # raw text
            doc = features_dict['doc']
            # reduce
            data_list.append([cid, cate_features_dict, word_ids, keywords, skills, doc])
        ids_list, category_features_list, word_ids, keywords, skills, doc = list(zip(*data_list))
        # id
        id_to_row = {k: v for v, k in enumerate(ids_list)}
        # category features
        vec = feature_extraction.DictVectorizer()
        features_matrix = vec.fit_transform(category_features_list)
        features_name = vec.get_feature_names()
        # doc
        word_ids, keywords, skills = [
            np.array(lis) for lis in [word_ids, keywords, skills]]
        return id_to_row, features_matrix, features_name, word_ids, keywords, skills, doc

    def doc2d(self, doc, shape):
        doc_len, sent_len = shape
        if type(doc) == str:
            doc = doc.strip().split('\t')
        id_doc = []
        for sent in doc[:doc_len]:
            sent = self.doc1d(sent, sent_len)
            id_doc.append(sent)
        if len(id_doc) < doc_len:
            id_doc += [[0] * sent_len] * (doc_len - len(id_doc))
        return id_doc

    def doc1d(self, sent, sent_len):
        if type(sent) == str:
            sent = sent.strip().split(' ')
        sent = [self.word_dict.get(word, 0) for word in sent][:sent_len]
        if len(sent) < sent_len:
            sent += [0] * (sent_len - len(sent))
        return sent

    def feature_lookup(self, idstr, idtype):
        if idtype == 'expect':
            row = self.exp_to_row[idstr]
            features = self.exp_features[row]
            docs = self.exp_docs[row]
            kws = self.exp_kws[row]
            skills = self.exp_skills[row]
            raw_docs = self.exp_doc_raw[row]
        else:
            row = self.job_to_row[idstr]
            features = self.job_features[row]
            docs = self.job_docs[row]
            kws = self.job_kws[row]
            skills = self.job_skills[row]
            raw_docs = self.job_doc_raw[row]
        return [features, docs, kws, skills, raw_docs]

    def data_generator(self, fp, batch_size):
        with open(fp) as f:
            batch_data = []
            for line in f:
                eid, jid, label = line.strip().split('\001')
                label = int(label)
                if jid not in self.job_to_row or eid not in self.exp_to_row:
                    print('loss id')
                    continue
                job_data, jd, jd_kws, jd_skill, jd_raw = self.feature_lookup(jid, 'job')
                exp_data, cv, cv_kws, cv_skill, cv_raw = self.feature_lookup(eid, 'expect')
                cate_features = sparse.hstack([job_data, exp_data])
                batch_data.append([jd, jd_kws, jd_skill, cv, cv_kws, cv_skill, cate_features, label])
                if len(batch_data) >= batch_size:
                    batch_data = list(zip(*batch_data))
                    yield [np.array(x) for x in batch_data]
                    batch_data = []


if __name__ == '__main__':
    mix_data = MixData(
        fpin='../Data/multi_data4/multi_data4',
        fpout='./data/multi_data4',
        wfreq=5,
        doc_len=50,
        emb_dim=64,
        load_emb=1,
    )
    fp = './data/multi_data4.train'
    g = mix_data.data_generator(fp, 2)
    print("done")