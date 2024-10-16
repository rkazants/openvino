# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.front.common.partial_infer.utils import int64_array, shape_array
from mo.graph.graph import Node, Graph
from mo.middle.passes.convert_data_type import data_type_str_to_np
from mo.ops.op import Op
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


class Memory(Op):
    op = 'Memory'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': 'Memory',
            'op': 'Memory',
            'id': None,
            'size': None,
            'index': None,
            'infer': Memory.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'type_infer': __class__.type_infer,
        }, attrs)

    def supported_attrs(self):
        return ['id', 'size', 'index']

    @staticmethod
    def infer(node: Node):
        if len(node.in_nodes()) > 0:
            # In case this is a memory node with input,
            # It should not have output
            # However in order not to break MO pipeline,
            # we just set the same shape to the output
            # node that will be removed later in pipeline
            copy_shape_infer(node)
            return
        elif node.has_valid('shape'):
            # For Memories, that has not input infer shapes is very difficult
            # But often we can know shape in extracting attributes
            # And we can set the attribute 'shape' in extracting
            batch = 1
            for out_node in node.out_nodes().values():
                out_node.shape = shape_array([batch, *node.shape[:]])
            return
        else:
            raise Error('Model Optimizer is unable to calculate output shape of Memory node {}. ' +
                        refer_to_faq_msg(88),
                        node.id)

    @staticmethod
    def type_infer(node: Node):
        if node.has_valid('dst_type'):
            node.out_port(0).set_data_type(node.dst_type)
        else:
            node.out_port(0).set_data_type(data_type_str_to_np(node.graph.graph['cmd_params'].data_type))
