#! /usr/bin/env python -u
# coding=utf-8
import json

from tensorflow.contrib.slim import nets

__author__ = 'Sayed Hadi Hashemi'

import argparse
import tensorflow as tf

FLAGS = None
BATCH_SIZE = 32


def base_model():
    inputs = tf.random_uniform([BATCH_SIZE, 299, 299, 3], name="Inputs")
    labels = tf.random_uniform([BATCH_SIZE, 1, 1, 1000], name="Labels")
    logit, _ = nets.resnet_v1.resnet_v1_152(inputs, 1000, scope=None)
    loss = tf.losses.mean_squared_error(labels, logit)
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    return optimizer.minimize(loss)


def main():
    with open(FLAGS.cluster_spec_file) as fp:
        cluster_spec = json.load(fp)
    workers = cluster_spec["worker"]
    master = workers[0]
    number_of_ps = len(cluster_spec["ps"])
    ps_job = "/job:ps/"
    train_ops = []

    for dev_id, _ in enumerate(workers):
        device = "/job:worker/task:{}/".format(dev_id)
        with tf.variable_scope("resnet", reuse=dev_id != 0):
            with tf.device(tf.train.replica_device_setter(ps_tasks=number_of_ps, ps_device=ps_job,
                                                          worker_device=device)):
                train = base_model()
                train_ops.append(train)

    with tf.train.MonitoredTrainingSession(master="grpc://{}".format(master)) as sess:
        for _ in range(100):
            sess.run(train_ops)


def arg_parser():
    global FLAGS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster_spec_file",
        type=str,
        help="Cluster Spec File Address"
    )
    FLAGS, _ = parser.parse_known_args()


if __name__ == "__main__":
    arg_parser()
    main()
