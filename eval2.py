# -*- coding: utf-8 -*-
# /usr/bin/python2



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from data_load import get_batch
from models import Model
import argparse
import hparams as hp


def eval(logdir, hparams):
    # Load graph
    model = Model(mode="test2", hparams=hparams)

    # Loss
    loss_op = model.loss_net2()

    # Summary
    summ_op = summaries(loss_op)

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        device_count={'CPU': 1, 'GPU': 0},
    )
    with tf.Session(config=session_conf) as sess:
        # Load trained model
        sess.run(tf.global_variables_initializer())
        model.load(sess, 'test2', logdir=logdir)

        writer = tf.summary.FileWriter(logdir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        mfcc, spec, mel = get_batch(model.mode, model.batch_size)
        summ, loss = sess.run([summ_op, loss_op], feed_dict={model.x_mfcc: mfcc, model.y_spec: spec, model.y_mel: mel})

        writer.add_summary(summ)
        writer.close()

        coord.request_stop()
        coord.join(threads)

        print(("loss:", loss))


def summaries(loss):
    tf.summary.scalar('net2/eval/loss', loss)
    return tf.summary.merge_all()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-case', type=str, default='default' ,help='experiment case name')
    parser.add_argument('-logdir', type=str, default='./logdir' ,help='tensorflow logdir, default: ./logdir')

    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    logdir = '{}/{}/train2'.format(args.logdir, args.case)
    
    eval(logdir=logdir, hparams=hp)
    
    print("Done")
