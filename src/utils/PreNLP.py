import jieba
import requests
import re
from multiprocessing import Pool, Lock, Manager
import sys
import time

ST = 0
TOP_N = 5
NUM = 0
lock = Lock()
manager = Manager()
ITEM = manager.Value('tmp', 0)


def print_num():
    global NUM
    global ST
    with lock:
        ITEM.value += 1
        if ITEM.value % 100 == 0:
            sys.stdout.write('\r{}/{}({:.1f}s/{:.1f}s {:.2f}h)'.format(
                ITEM.value,
                NUM,
                time.time()-ST,
                (time.time() - ST) * (NUM / ITEM.value),
                (time.time() - ST) * (NUM / ITEM.value) / 3600,
            ))


def get_structs(doc):
    url = 'http://192.168.1.45:8078/api/jdstructure'
    body = doc.encode('utf8')
    result = requests.post(url, data=body)
    try:
        result = result.json()
    except:
        return []
    keys = ['Softskill', 'Hardskill']
    result = [result[k] for k in keys if result[k] != '']
    result = '\n'.join(result)
    result = result.replace("'", '')
    result = result.split('\n')
    return result


def get_words(doc, position, top_n):
    url = "http://192.168.1.45:8078/api/tagwordMatch"
    body = {
        'sentence': doc,
        'positionCode': position,
        'topN': top_n
    }
    result = requests.post(url, data=body)
    result = result.json()
    result = result['tagwords']
    if result is not None:
        result = [dic['word'] for dic in result]
    else:
        result = []
    return result


# def nlp_features(fpin, fpout, n_col, top_n):
#     with open(fpin) as fin:
#         datain = fin.readlines()
#     dataout = []
#     for line in tqdm(datain):
#         data = line.strip().split('\001')
#         if len(data) != n_col:
#             continue
#         position = data[2]
#         doc = data[-1]
#         doc = doc.replace(' ', '')
#         doc = re.sub("[。]+", "。", doc)
#         words = jieba.cut(doc)
#         words = ' '.join(words)
#         skills = get_structs(doc)
#         skills = '\t'.join(skills)
#         keywords = get_words(doc, position, top_n)
#         keywords = ' '.join(keywords)
#         new_line = "{}\001{}\001{}\001{}\n".format(line, words, skills, keywords)
#         dataout.append(new_line)
#     with open(fpout, 'w') as fout:
#         fout.write('\n'.join(new_line))
#     return


def nlp_feature(line):
    line = line.strip()
    data = line.split('\001')
    if len(data) != N_COL:
        new_line = "{}\001{}\001{}\001{}".format(line, '', '', '')
        return new_line
    position = data[2]
    doc = data[-1]
    doc = doc.replace(' ', '')
    doc = re.sub("[。]+", "。", doc)
    words = jieba.cut(doc)
    words = ' '.join(words)
    skills = get_structs(doc)
    skills = '\t'.join(skills)
    keywords = get_words(doc, position, TOP_N)
    keywords = ' '.join(keywords)
    new_line = "{}\001{}\001{}\001{}".format(line, words, skills, keywords)
    print_num()
    return new_line


def nlp_features(fpin, fpout):
    with open(fpin) as fin:
        datain = fin.readlines()
    global NUM
    NUM = len(datain)
    global ST
    ST = time.time()
    global ITEM
    ITEM.value = 0
    with Pool() as pool:
        dataout = pool.map(nlp_feature, datain)
    with open(fpout, 'w') as fout:
        fout.write('\n'.join(dataout))
    return


if __name__ == "__main__":
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
        'cv'
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
        'jd'
    ]
    N_COL = len(exp_features_names)
    nlp_features(
        fpin="../Data/multi_data5/multi_data5.profile.expect",
        fpout="./data/multi_data5/multi_data5.profile.expect",
    )
    N_COL = len(job_features_names)
    nlp_features(
        fpin="../Data/multi_data5/multi_data5.profile.job",
        fpout="./data/multi_data5/multi_data5.profile.job",
    )

