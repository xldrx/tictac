#! /usr/bin/env python -u
# coding=utf-8
from functools import lru_cache

from tictac.common import BaseOrdering, OrderingPostProcesser

__author__ = 'Sayed Hadi Hashemi'


class TIC(BaseOrdering, OrderingPostProcesser):
    @lru_cache()
    def _get_time(self, op):
        if self._is_recv(op):
            return 1
        else:
            return 0

    def _get_results(self):
        outstanding_comm_ops = list(self._comm_ops)
        self._update_properties(outstanding_comm_ops)
        priorities = []
        counter = 0
        last_counter = -1
        last_Mp = -1
        for op in sorted(outstanding_comm_ops, key=lambda recv_op: recv_op.Mp):
            if last_Mp < op.Mp:
                last_counter = counter
                last_Mp = op.Mp
            priorities.append((last_counter, op.op.name))
            counter += 1
        return priorities
