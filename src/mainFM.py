import tensorflow as tf
from utils import MixData, Trainer, Visual
from nets import SMP
from nets import FM
import argparse
import os
import shutil
import numpy as np
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default='0,1')
    parser.add_argument('--datain', type=str, default='../Data/multi_data6/multi_data6')
    parser.add_argument('--dataout', default='./data/multi_data6/multi_data6')
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    # word2vec arguments
    parser.add_argument('--min_count', type=int, default=5)
    # parser.add_argument('--pad_zero', type=int, default=1)
    parser.add_argument('--load_emb', type=int, default=0)
    # model arguments
    parser.add_argument('--doc_len', type=int, default=200)
    parser.add_argument('--n_skill', type=int, default=10)
    parser.add_argument('--n_keywords', type=int, default=10)
    parser.add_argument('--skill_len', type=int, default=10)
    parser.add_argument('--conv_size', type=int, default=3)
    parser.add_argument('--mode', type=str, default='all_concat')
    parser.add_argument('--cate_emb', default='diag')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--reg', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--n_epoch', type=int, default=10)
    # tf arguments
    parser.add_argument('--board_dir', default='board')
    return parser.parse_args()


if __name__ == '__main__':
    # 参数接收器
    args = parse_args()

    # 显卡占用
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    if os.path.exists('{}.pkl'.format(args.dataout)):
        with open('{}.pkl'.format(args.dataout), 'rb') as f:
            mix_data = pickle.load(f)
    else:
        mix_data = MixData.MixData(
            fpin=args.dataout,
            fpout=args.dataout,
            wfreq=args.min_count,
            doc_len=args.doc_len,
            n_skill=args.n_skill,
            skill_len=args.skill_len,
            n_keywords=args.n_keywords,
            emb_dim=args.emb_dim,
            load_emb=args.load_emb,
        )
        with open('{}.pkl'.format(args.dataout), 'wb') as f:
            pickle.dump(mix_data, f)

    train_data = lambda: mix_data.data_generator(
        fp='{}.train'.format(args.dataout),
        batch_size=args.batch_size
    )

    test_data = lambda: mix_data.data_generator(
        fp='{}.test'.format(args.dataout),
        batch_size=args.batch_size
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        board_dir = './board/{}'.format(args.board_dir)
        if os.path.exists(board_dir):
            shutil.rmtree(board_dir)
        writer = tf.summary.FileWriter(board_dir)
        model = FM.Model(
            n_feature=len(mix_data.feature_name_sparse),
            emb_dim=args.emb_dim,
            l2=args.reg,
        )
        writer.add_graph(sess.graph)

        Trainer.train(
            sess=sess,
            model=model,
            writer=writer,
            train_data_fn=train_data,
            test_data_fn=test_data,
            lr=args.lr,
            n_epoch=args.n_epoch,
        )

    writer.close()

