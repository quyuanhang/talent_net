import tensorflow as tf
from utils import MixData, Trainer, Visual
from nets import TextCrossNet
import argparse
import os
import shutil
import numpy as np
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default='0,1')
    parser.add_argument('--datain', type=str, default='../Data/multi_data/multi_data')
    parser.add_argument('--dataout', default='./data/multi_data')
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=512)
    # word2vec arguments
    parser.add_argument('--min_count', type=int, default=5)
    # parser.add_argument('--pad_zero', type=int, default=1)
    parser.add_argument('--load_emb', type=int, default=0)
    # model arguments
    parser.add_argument('--doc_len', type=int, default=100)
    parser.add_argument('--sent_len', type=int, default=0)
    parser.add_argument('--block_len', type=int, default=5)
    parser.add_argument('--mode', type=str, default='cross')
    parser.add_argument('--cate_emb', default='diag')
    parser.add_argument('--global_emb', type=int, default=0)
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
            fpin=args.datain,
            fpout=args.dataout,
            wfreq=args.min_count,
            doc_len=args.doc_len,
            sent_len=args.sent_len,
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

    test_data_raw = lambda: mix_data.data_generator(
        fp='{}.test'.format(args.dataout),
        batch_size=args.batch_size,
        raw=1
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        board_dir = './board/{}'.format(args.board_dir)
        if os.path.exists(board_dir):
            shutil.rmtree(board_dir)
        writer = tf.summary.FileWriter(board_dir)

        model = TextCrossNet.TextCrossNet(
            doc_len=args.doc_len,
            sent_len=args.sent_len,
            block_len=args.block_len,
            feature_len=len(mix_data.feature_name),
            emb_dim=args.emb_dim,
            n_feature=len(mix_data.feature_name_sparse),
            n_word=len(mix_data.word_dict),
            emb_pretrain=mix_data.embs,
            l2=args.reg,
            mode=args.mode,
            cate_emb=args.cate_emb,
            dropout=args.dropout,
            global_emb=args.global_emb,
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

        visual_str = Visual.visual(
            sess=sess,
            model=model,
            test_data_fn=test_data,
            raw_data_fn=test_data_raw,
        )
        with open('./data/visual.html', 'w') as f:
            f.write(visual_str)

        # constant_graph = tf.graph_util.convert_variables_to_constants(
        #     sess, sess.graph_def, ['classifier/output/Sigmoid'])

        # with tf.gfile.FastGFile('./data/' + 'model.pb', mode='wb') as f:
        #     f.write(constant_graph.SerializeToString())

    writer.close()

