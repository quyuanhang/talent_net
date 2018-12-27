import tqdm
import argparse
import pandas as pd
import numpy as np


def build_negative_sample(frame, n_neg=1):
    frame['label'] = '1'
    values = frame.values
    n_sample = len(values)
    negative_sample = list()
    while len(negative_sample) < n_sample * n_neg:
        idx_tmp = np.random.randint(n_sample, size=[int(n_sample * 1.1), 2])
        sample_tmp = [
            [values[j, 0], values[e, 1], '0']
            for j, e in idx_tmp
            if j != e
        ]
        negative_sample.extend(sample_tmp)
    negative_sample = np.array(negative_sample[:int(n_sample * n_neg)])
    sample = np.r_[values, negative_sample]
    sample = np.random.permutation(sample)
    sample = pd.DataFrame(sample)
    return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datain')
    parser.add_argument('--dataout')
    parser.add_argument('--rand_sample', type=int, default=1)
    args = parser.parse_args()

    train_frame = pd.read_csv(
        '{}.train'.format(args.datain),
        sep='\001',
        header=None,
        dtype=str,
    )
    train_frame = build_negative_sample(train_frame)
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
        test_frame = build_negative_sample(test_frame)

    train_frame.to_csv('{}.train'.format(args.dataout), sep='\001', header=None, index=False)
    test_frame.to_csv('{}.test'.format(args.dataout), sep='\001', header=None, index=False)



