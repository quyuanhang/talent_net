import tqdm
import argparse
import pandas as pd
import numpy as np


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
    train_frame.columns = ['eid', 'jid', 'label']
    train_frame['label'] = train_frame['label'].map(binary)
    train_frame = calibration(train_frame, 0.2)

    # train_frame = train_frame[train_frame['label'] != '0']
    # train_frame['label'] = train_frame['label'].map(lambda x: int(x=='2'))

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
        test_frame = calibration(test_frame, 0.2)

    train_frame.to_csv('{}.train'.format(args.dataout), sep='\001', header=None, index=False)
    test_frame.to_csv('{}.test'.format(args.dataout), sep='\001', header=None, index=False)



