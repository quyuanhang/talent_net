import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm


def binary(s):
    if int(s) > 2:
        return 1
    return 0




def calibration(frame, rate):
    if len(frame[frame['label'] == 1]) < rate * len(frame[frame['label'] == 0]):
        frame = pd.concat([
            frame[frame['label'] == 1],
            frame[frame['label'] == 0].sample(frac=rate)
        ])
        col = frame.columns
        frame = pd.DataFrame(np.random.permutation(frame.values))
        frame.columns = col
    return frame


def his_sample(frame: pd.DataFrame, posi_label=1):
    his_posi = frame[frame["label"] == posi_label]
    his_posi = his_posi.sort_values(by=["jid", "ds"])
    his_posi.index = his_posi["jid"]
    return his_posi


def his_query(frame: pd.DataFrame, itv_his: pd.DataFrame, n_his=3):
    sample_with_his = []
    for i, row in tqdm(frame.iterrows()):
        eid, jid, ds, label = row.values
        if jid not in itv_his.index:
            continue
        job_his = itv_his.loc[[jid]]
        job_his = job_his[job_his["ds"] <= ds]
        if len(job_his) == 0:
            continue
        job_his = list(job_his["eid"][-n_his:])
        if len(job_his) < n_his:
            job_his = [0] * (n_his-len(job_his)) + job_his
        sample = job_his + [eid, jid, label]
        sample_with_his.append(sample)
    names = ["his_{}".format(i+1) for i in range(n_his)]
    names = names + ["eid", "jid", "label"]
    sample_with_his = pd.DataFrame(sample_with_his, columns=names)
    return sample_with_his


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datain')
    parser.add_argument('--dataout')
    parser.add_argument('--rand_sample', type=int, default=1)
    args = parser.parse_args()

    train_test_frame = pd.read_csv(
        "{}.all".format(args.datain),
        sep='\001',
        header=None,
        dtype=str,
    )
    train_test_frame.columns = ['eid', 'jid', 'ds', 'label']
    train_test_frame["label"] = train_test_frame["label"].map(binary)
    his_frame = his_sample(train_test_frame)

    train_frame = pd.read_csv(
        '{}.train'.format(args.datain),
        sep='\001',
        header=None,
        dtype=str,
    )
    train_frame.columns = ['eid', 'jid', 'ds', 'label']
    train_frame['label'] = train_frame['label'].map(binary)
    train_frame = his_query(train_frame, his_frame)
    train_frame = calibration(train_frame, 0.2)

    if args.rand_sample:
        train_frame, test_frame = train_frame.iloc[:-20000], train_frame.iloc[-20000:]
    else:
        test_frame = pd.read_csv(
            '{}.test'.format(args.datain),
            sep='\001',
            header=None,
            dtype=str,
        )
        test_frame = test_frame.iloc[:20000]
        test_frame.columns = ['eid', 'jid', 'label']
        test_frame['label'] = test_frame['label'].map(binary)
        test_frame = his_query(test_frame, his_frame)
        test_frame = calibration(test_frame, 0.2)

    train_frame.to_csv('{}.train'.format(args.dataout), sep='\001', header=None, index=False)
    test_frame.to_csv('{}.test'.format(args.dataout), sep='\001', header=None, index=False)



