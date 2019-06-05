#! /usr/bin/env python -u
# coding=utf-8
import re
from functools import lru_cache, cmp_to_key

from tftracer import Timeline

from tictac.common import BaseOrdering, OrderingPostProcesser

__author__ = 'Sayed Hadi Hashemi'


class TimeOracle:
    def __init__(self, timeline_file):
        self.__timeline = Timeline.from_pickle(timeline_file)
        self.__costs = self.__timeline_analyser(self.__timeline)

    @staticmethod
    def __timeline_analyser(timeline):
        costs = {}
        for device in timeline._run_metadata.step_stats.dev_stats:
            for n in device.node_stats:
                try:
                    if "RecvTensor" in n.node_name:
                        node_name = re.findall("edge_\d+_(.+) from", n.timeline_label)[0]
                    elif ":" in n.node_name:
                        node_name, op_type = n.node_name.split(":")
                    else:
                        node_name = n.node_name

                    time = n.all_end_rel_micros / 1e6
                    costs[node_name] = time + costs.get(node_name, 0)
                except:
                    print(n)
        return costs

    def get(self, name):
        return self.__costs.get(name, None)


class TAC(BaseOrdering, OrderingPostProcesser):

    def __init__(self, endpoint, timeline_file) -> None:
        super().__init__(endpoint)
        self.__oracle = TimeOracle(timeline_file)

    @staticmethod
    def _comparator(op1, op2):
        a = min(op2.P, op1.M)
        b = min(op1.P, op2.M)

        if a != b:
            return -1 if a < b else 0 if a == b else 1
        else:
            return -1 if op1.Mp < op2.Mp else 0 if op1.Mp == op2.Mp else 1

    @lru_cache()
    def _get_time(self, op):
        if self._is_recv(op):
            op_name = self._is_recv(op)
            time = self.__network_costs(op_name)
            # print(op_name, op_size, time)
        else:
            op_name = op.op.name
            time = self.__graph_costs(op_name)

        if time:
            return time

        else:
            print("// >>> Error (Server-Client version mismatch?): {}".format(op_name))
            return 1e-3

    def _get_results(self):
        outstanding_comm_ops = list(self._comm_ops)
        priorities = []
        counter = 0
        while outstanding_comm_ops:
            self._update_properties(outstanding_comm_ops)
            outstanding_comm_ops.sort(key=cmp_to_key(self._comparator))
            # for op in outstanding_comm_ops:
            #     print(op.op.name, op.P, op.M, op.Mp)
            priorities.append((counter, outstanding_comm_ops[0].op.name))
            counter += 1
            del outstanding_comm_ops[0]
        return priorities

    def __network_costs(self, op_name):
        return self.__oracle.get(op_name+"/read")

    def __graph_costs(self, op_name):
        return self.__oracle.get(op_name)
