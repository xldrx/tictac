#! /usr/bin/env python -u
# coding=utf-8

__author__ = 'Sayed Hadi Hashemi'

from tictac import TIC
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets

BATCH_SIZE = 32
NUM_PS = 4

def main():
    with tf.variable_scope("resnet"):
        with tf.device(tf.train.replica_device_setter(ps_tasks=NUM_PS, ps_device="/job:ps/",
                                                      worker_device="/job:worker/task:0/")):
            inputs = tf.random_uniform([BATCH_SIZE, 299, 299, 3], name="Inputs")
            logit, _ = nets.resnet_v1.resnet_v1_152(inputs, 1000, scope=None)

    tic = TIC(endpoint=logit)
    tic.save("tic_rpc_orders.txt", partition_devices=True)


if __name__ == "__main__":
    main()