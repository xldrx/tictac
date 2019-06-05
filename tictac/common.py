#! /usr/bin/env python -u
# coding=utf-8
import math
from abc import ABC

import tensorflow as tf

__author__ = 'Sayed Hadi Hashemi'


class BaseOrdering:
    def __init__(self, endpoint) -> None:
        self._variables_sizes = {}
        self._target = endpoint
        self._model = tf.get_default_graph()
        self._find_comm_dependencies()
        self._seperate_comp_comm()
        self._get_variable_sizes()

    @staticmethod
    def _get_size(var: tf.Variable) -> float:
        ret = var.dtype.size
        for d in var.shape:
            ret *= d.value
        return ret

    def _get_variable_sizes(self):
        variables_sizes = {var.op.name: self._get_size(var) for var in tf.global_variables()}
        self._variables_sizes = variables_sizes

    @staticmethod
    def _is_recv(op):
        if op.op.name.endswith("/read"):
            return op.op.name[:-5]
        else:
            return None

    def _find_comm_dependencies(self):
        stack = [self._target]
        processed = {}
        while stack:
            op = stack.pop()
            if processed.get(op, False):
                deps = set()
                for input_op in op.op.inputs:
                    deps.update(input_op.deps)
                op.deps = deps
            else:
                if self._is_recv(op):
                    op.deps = {op}
                else:
                    stack.append(op)
                    for input_op in op.op.inputs:
                        stack.append(input_op)
                processed[op] = True

    def _seperate_comp_comm(self):
        self._comp_ops = []
        self._comm_ops = []
        stack = [self._target]
        processed = {}
        while stack:
            op = stack.pop()
            if op not in processed:
                if self._is_recv(op):
                    self._comm_ops.append(op)
                else:
                    self._comp_ops.append(op)
                    for input_op in op.op.inputs:
                        stack.append(input_op)
                processed[op] = True

    def _update_properties(self, outstanding_comm_ops):
        for op in self._comm_ops:
            op.P = 0
            op.Mp = math.inf
            op.M = self._get_time(op)

        for op in self._comp_ops:
            op_deps = op.deps.intersection(outstanding_comm_ops)
            if len(op_deps) == 1:
                for read in op_deps:
                    read.P += self._get_time(op)
            elif len(op_deps) > 1:
                op_M = sum(self._get_time(r) for r in op_deps)
                for read in op_deps:
                    read.Mp = min(op_M, read.Mp)
        pass

    def _get_time(self, op):
        raise NotImplementedError()

    def _get_results(self):
        raise NotImplementedError()


class OrderingPostProcesser(ABC):
    def save(self, filename, variable_scope=None, partition_devices=False):
        orders = self._get_results()
        orders = self.__fix_names(variable_scope, orders)
        if partition_devices:
            # TODO
            orders = self.__partition_devices(orders)
        self.__save_to_disk(filename, orders)

    @staticmethod
    def __normalized_list(rpc_list):
        counter = 0
        last_counter = -1
        last_pri = -1
        ret = []
        for pri, rpc_name in sorted(rpc_list):
            if last_pri < pri:
                last_counter = counter
                last_pri = pri
            ret.append((last_counter, rpc_name))
            counter += 1
        return ret

    @staticmethod
    def __list_to_text(rpc_list):
        return "\n".join([
            "{name}\t{priority}".format(name=name, priority=priority)
            for priority, name in sorted(rpc_list)
        ])

    def _get_results(self):
        raise NotImplemented()

    @staticmethod
    def __fix_names(prefix, orders):
        ret = []
        for priority, name in orders:
            base_name = name[:-5]  # removing /read postfix
            if prefix is not None:
                base_name = "{prefix}/{name}".format(prefix=prefix, name=base_name)
            ret.append((priority, base_name))
        return ret

    def __save_to_disk(self, filename, orders):
        content = self.__list_to_text(orders)
        with open(filename, "w") as fp:
            fp.write(content)

    def __partition_devices(self, orders):
        orders_dict = {name: priority for priority, name in orders}
        devices = {}
        for var in tf.trainable_variables():
            device = var.device
            if device not in devices:
                devices[device] = []
            name = var.op.name
            if name in orders_dict:
                devices[device].append((orders_dict[name], name))
            else:
                print("{} not found".format(name))

        ret = []
        for r_list in devices.values():
            ret += self.__normalized_list(r_list)
        return ret
