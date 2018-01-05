# -*- coding: utf-8 -*-
# /usr/bin/python2



import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import librosa
import argparse
import numpy as np
import datetime

from data_load import get_wav
from models import Model
from utils import spectrogram2wav, inv_preemphasis

import hparams as hp


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)


def convert(logdir, hparams, input_file, output_file):

    # Load graph
    model = Model(mode="convert", hparams=hparams)

    #session_conf = tf.ConfigProto(
    #    allow_soft_placement=True,
    #    device_count={'CPU': 1, 'GPU': 0},
    #    gpu_options=tf.GPUOptions(
    #        allow_growth=True,
    #        per_process_gpu_memory_fraction=0.6
    #    ),
    #)

    session_conf=tf.ConfigProto()
    session_conf.gpu_options.per_process_gpu_memory_fraction=0.9


    with tf.Session(config=session_conf) as sess:
        # Load trained model
        sess.run(tf.global_variables_initializer())
        model.load(sess, 'convert', logdir=logdir)

        writer = tf.summary.FileWriter(logdir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        gs = Model.get_global_step(logdir)

        mfcc, spec, mel = get_wav(input_file, model.batch_size)

        pred_log_specs, y_log_spec, ppgs = sess.run([model(), model.y_spec, model.ppgs], feed_dict={model.x_mfcc: mfcc, model.y_spec: spec, model.y_mel: mel})

        # Denormalizatoin
        # pred_log_specs = hparams.mean_log_spec + hparams.std_log_spec * pred_log_specs
        # y_log_spec = hparams.mean_log_spec + hparams.std_log_spec * y_log_spec
        # pred_log_specs = hparams.min_log_spec + (hparams.max_log_spec - hparams.min_log_spec) * pred_log_specs
        # y_log_spec = hparams.min_log_spec + (hparams.max_log_spec - hparams.min_log_spec) * y_log_spec

        # Convert log of magnitude to magnitude
        pred_specs, y_specs = np.e ** pred_log_specs, np.e ** y_log_spec

        # Emphasize the magnitude
        pred_specs = np.power(pred_specs, hparams.Convert.emphasis_magnitude)
        y_specs = np.power(y_specs, hparams.Convert.emphasis_magnitude)

        # Spectrogram to waveform
        audio = np.array([spectrogram2wav(spec.T, hparams.Default.n_fft, hparams.Default.win_length, hparams.Default.hop_length, hparams.Default.n_iter) for spec in pred_specs])
        y_audio = np.array([spectrogram2wav(spec.T, hparams.Default.n_fft, hparams.Default.win_length, hparams.Default.hop_length, hparams.Default.n_iter) for spec in y_specs])

        # Apply inverse pre-emphasis
        audio = inv_preemphasis(audio, coeff=hparams.Default.preemphasis)
        y_audio = inv_preemphasis(y_audio, coeff=hparams.Default.preemphasis)

        # Concatenate to a wav
        y_audio = np.reshape(y_audio, (y_audio.size, 1), order='C')
        audio = np.reshape(audio, ( audio.size, 1), order='C')

        write_wav(audio, hparams.Default.sr, output_file)

        # Write the result
        tf.summary.audio('A', y_audio, hparams.Default.sr, max_outputs=hparams.Convert.batch_size)
        tf.summary.audio('B', audio, hparams.Default.sr, max_outputs=hparams.Convert.batch_size)

        # Visualize PPGs
        heatmap = np.expand_dims(ppgs, 3)  # channel=1
        tf.summary.image('PPG', heatmap, max_outputs=ppgs.shape[0])

        writer.add_summary(sess.run(tf.summary.merge_all()), global_step=gs)
        writer.close()

        coord.request_stop()
        coord.join(threads)



def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-case', type=str, default='default' ,help='experiment case name of train2')
    parser.add_argument('-logdir', type=str, default='./logdir' ,help='tensorflow logdir, default: ./logdir')
    
    parser.add_argument('-input', type=str, default='datasets/arctic/bdl/arctic_a0090.wav', help='wav file to regenerate')
    parser.add_argument('-output', type=str, default='regenerate.wav', help='output file')
    parser.add_argument('-batch_size', type=int, default=hp.Convert.batch_size, help='batch size')
    parser.add_argument('-emphasis_magnitude', type=int, default=hp.Convert.emphasis_magnitude, help='emphasis magnitude')

    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    
    logdir = '{}/{}/train2'.format(args.logdir, args.case)
    print('case: {}, logdir: {}'.format(args.case, logdir))
    
    s = datetime.datetime.now()

    hp.Convert.batch_size = args.batch_size
    hp.Convert.emphasis_magnitude = args.emphasis_magnitude

    convert(logdir, hp, args.input, args.output)

    e = datetime.datetime.now()
    diff = e - s
    print("Done. elapsed time:{}s".format(diff.seconds))
