# -*- coding: utf-8 -*-
# /usr/bin/python3

import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

from data_load import get_batch
from models import Model
import eval1

import hparams as hp



def train(logdir, hparams):

    model = Model(mode="train1", hparams=hparams)

    # Loss
    loss_op = model.loss_net1()

    # Accuracy
    acc_op = model.acc_net1()

    # Training Scheme
    global_step = tf.Variable(0, name='global_step', trainable=False)

    optimizer = tf.train.AdamOptimizer(learning_rate=hparams.Train1.lr)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net1')
        train_op = optimizer.minimize(loss_op, global_step=global_step, var_list=var_list)

    # Summary
    # for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net1'):
    #     tf.summary.histogram(v.name, v)
    tf.summary.scalar('net1/train/loss', loss_op)
    tf.summary.scalar('net1/train/acc', acc_op)
    summ_op = tf.summary.merge_all()

    #session_conf = tf.ConfigProto(
    #    gpu_options=tf.GPUOptions(
    #        allow_growth=True,
    #    ),
    #)

    session_conf=tf.ConfigProto()
    session_conf.gpu_options.per_process_gpu_memory_fraction=0.9

    # Training
    with tf.Session(config=session_conf) as sess:
        # Load trained model
        sess.run(tf.global_variables_initializer())
        model.load(sess, 'train1', logdir=logdir)

        writer = tf.summary.FileWriter(logdir, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(1, hparams.Train1.num_epochs + 1):
            for step in range(model.num_batch):
                mfcc, ppg = get_batch(model.mode, model.batch_size)
                sess.run(train_op, feed_dict={model.x_mfcc: mfcc, model.y_ppgs: ppg})

            # Write checkpoint files at every epoch
            summ, gs = sess.run([summ_op, global_step], feed_dict={model.x_mfcc: mfcc, model.y_ppgs: ppg})


            if epoch % hparams.Train1.save_per_epoch == 0:
                tf.train.Saver().save(sess, '{}/epoch_{}_step_{}'.format(logdir, epoch, gs))

            # Write eval accuracy at every epoch
            with tf.Graph().as_default():
                eval1.eval(logdir=logdir, hparams=hparams)

            writer.add_summary(summ, global_step=gs)

        writer.close()
        coord.request_stop()
        coord.join(threads)



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-case1', type=str, default='default' ,help='experiment case name')
    parser.add_argument('-logdir', type=str, default='./logdir' ,help='tensorflow logdir, default: ./logdir')
    
    parser.add_argument('-batch_size', type=int, default=hp.Train1.batch_size,
        help='batch size, default {}'.format(hp.Train1.batch_size))
    parser.add_argument('-lr', type=float, default=hp.Train1.lr,
        help='learning rate, default: {}'.format(hp.Train1.lr) )
    parser.add_argument('-num_epochs', type=int, default=hp.Train1.num_epochs,
        help='number of epochs, default: {}'.format(hp.Train1.num_epochs) )
    parser.add_argument('-save_per_epoch', type=int, default=hp.Train1.save_per_epoch,
        help='save model every n epoch, default: {}'.format(hp.Train1.save_per_epoch) )
    parser.add_argument('-data_path', type=str, default=hp.Train1.data_path,
        help='trainign data path, default: {}'.format(hp.Train1.data_path) )

    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':
    args = get_arguments()
    logdir = '{}/{}/train1'.format(args.logdir, args.case1)

    #update hparamas
    hp.Train1.batch_size = args.batch_size
    hp.Train1.lr = args.lr
    hp.Train1.num_epochs = args.num_epochs
    hp.Train1.save_per_epoch = args.save_per_epoch
    hp.Train1.data_path = args.data_path

    train(logdir=logdir, hparams = hp)
    
    print("Done")