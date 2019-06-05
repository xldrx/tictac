#! /usr/bin/env python -u
# coding=utf-8
import json

__author__ = 'Sayed Hadi Hashemi'

import argparse
import tensorflow as tf

FLAGS = None


def main():
    global FLAGS
    with open(FLAGS.cluster_spec_file, "r") as fp:
        cluster_spec_str = json.load(fp)

    config = tf.ConfigProto()
    if FLAGS.job_name == "ps":
        config.inter_op_parallelism_threads = 768
        config.intra_op_parallelism_threads = 0
        config.device_count['GPU'] = 0

    cluster = tf.train.ClusterSpec(cluster_spec_str)
    print("::: Starting {}->{}".format(FLAGS.job_name, FLAGS.task_id), flush=True)
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id, config=config)
    server.join()


def arg_parser():
    global FLAGS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster_spec_file",
        type=str,
        help="Cluster Spec File Address"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        help="Job name. e.g. 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_id",
        type=int,
        help="Index of task within the job"
    )
    FLAGS, _ = parser.parse_known_args()


if __name__ == "__main__":
    arg_parser()
    main()
