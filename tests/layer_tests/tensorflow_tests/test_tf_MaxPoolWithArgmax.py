# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestMaxPoolWithArgmax(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input' in inputs_info
        input_shape = inputs_info['input']
        inputs_data = {}
        inputs_data['input'] = np.random.randint(-5, 5, input_shape).astype(self.input_type)
        return inputs_data

    def create_max_pool_with_argmax_net(self, input_shape, ksize, strides, input_type, padding, targmax,
                                        include_batch_in_index, with_second_output):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(input_type, input_shape, 'input')
            max_pool_with_argmax = tf.raw_ops.MaxPoolWithArgmax(input=input, ksize=ksize, strides=strides,
                                                                padding=padding, Targmax=targmax,
                                                                include_batch_in_index=include_batch_in_index
                                                                )
            tf.identity(max_pool_with_argmax[0], name='max_pool')
            if with_second_output:
                tf.identity(max_pool_with_argmax[1], name='output_indices')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[1, 25, 24, 3],
             ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1]),
        dict(input_shape=[1, 10, 20, 3],
             ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.parametrize("input_type", [
        np.float32, np.int32
    ])
    @pytest.mark.parametrize("padding", [
        'VALID', 'SAME'
    ])
    @pytest.mark.parametrize("targmax", [
        tf.int32, tf.int64
    ])
    @pytest.mark.parametrize("include_batch_in_index", [
        True, False
    ])
    @pytest.mark.parametrize("with_second_output", [
        pytest.param(
            True,
            marks=pytest.mark.xfail(reason="Ticket: TransposeSinking crash")
        ),
        False
    ])
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_max_pool_with_argmax_basic(self, params, input_type, padding, targmax,
                                        include_batch_in_index, with_second_output,
                                        ie_device, precision, ir_version, temp_dir,
                                        use_new_frontend, use_old_api):
        self._test(
            *self.create_max_pool_with_argmax_net(**params, input_type=input_type, padding=padding, targmax=targmax,
                                                  include_batch_in_index=include_batch_in_index,
                                                  with_second_output=with_second_output),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_new_frontend=use_new_frontend, use_old_api=use_old_api)
