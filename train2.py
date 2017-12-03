# -*- coding: utf-8 -*-
#/usr/bin/python3


import os, sys
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

from models import Model
from data_load import get_batch
import convert, eval2

import hparams as hp


def train(logdir1, logdir2, hparams):
    model = Model(mode="train2", hparams=hparams)

    # Loss
    loss_op = model.loss_net2()

    # Training Scheme
    global_step = tf.Variable(0, name='global_step', trainable=False)

    optimizer = tf.train.AdamOptimizer(learning_rate=hp.Train2.lr)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net2')
        train_op = optimizer.minimize(loss_op, global_step=global_step, var_list=var_list)

    # Summary
    summ_op = summaries(loss_op)

    session_conf = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=0.6,
        ),
    )
    # Training
    with tf.Session(config=session_conf) as sess:
        # Load trained model
        sess.run(tf.global_variables_initializer())
        model.load(sess, mode='train2', logdir=logdir1, logdir2=logdir2)

        writer = tf.summary.FileWriter(logdir2, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(1, hp.Train2.num_epochs + 1):
            for step in list(range(model.num_batch)):
                mfcc, spec, mel = get_batch(model.mode, model.batch_size)
                sess.run(train_op, feed_dict={model.x_mfcc: mfcc, model.y_spec: spec, model.y_mel: mel})

            # Write checkpoint files at every epoch
            summ, gs = sess.run([summ_op, global_step], feed_dict={model.x_mfcc: mfcc, model.y_spec: spec, model.y_mel: mel})

            if epoch % hp.Train2.save_per_epoch == 0:
                tf.train.Saver().save(sess, '{}/epoch_{}_step_{}'.format(logdir2, epoch, gs))

                # Eval at every n epochs
                with tf.Graph().as_default():
                    eval2.eval(logdir2, hparams)

                # Convert at every n epochs
                with tf.Graph().as_default():
                    convert.convert(logdir2, hparams)

            writer.add_summary(summ, global_step=gs)

        writer.close()
        coord.request_stop()
        coord.join(threads)


def summaries(loss):
    # for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net2'):
    #     tf.summary.histogram(v.name, v)
    tf.summary.scalar('net2/train/loss', loss)
    return tf.summary.merge_all()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-case1', type=str, default='default' ,help='experiment case name of train1')
    parser.add_argument('-case2', type=str, default='default' ,help='experiment case name of train2')
    parser.add_argument('-logdir', type=str, default='./logdir' ,help='tensorflow logdir, default: ./logdir')

    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':
    args = get_arguments()
    logdir1 = '{}/{}/train1'.format(args.logdir, args.case1)
    logdir2 = '{}/{}/train2'.format(args.logdir, args.case2)
    
    train(logdir1=logdir1, logdir2=logdir2, hparams=hp)
    
    print("Done")