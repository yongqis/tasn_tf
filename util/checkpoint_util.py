#!/usr/bin/env python
# -*- coding:utf-8 -*-
import logging
import re
import tensorflow as tf


def get_variables_available_in_checkpoint(variables, checkpoint_path, include_global_step=True):
    """Returns the subset of variables available in the checkpoint.

    Inspects given checkpoint and returns the subset of variables that are
    available in it.

    TODO(rathodv): force input and output to be a dictionary.

    Args:
      variables: a list or dictionary of variables to find in checkpoint.
      checkpoint_path: path to the checkpoint to restore variables from.
      include_global_step: whether to include `global_step` variable, if it
        exists. Default True.

    Returns:
      A list or dictionary of variables.
    Raises:
      ValueError: if `variables` is not a list or dict.
    """
    # 类型判断，统一转换为字典类型
    if isinstance(variables, list):
        variable_names_map = {}
        for variable in variables:
            # if isinstance(variable, tf_variables.PartitionedVariable):
            #     name = variable.name
            # else:
            name = variable.op.name
            variable_names_map[name] = variable
    elif isinstance(variables, dict):
        variable_names_map = variables
    else:
        raise ValueError('`variables` is expected to be a list or dict.')

    ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
    ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
    if not include_global_step:
        ckpt_vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)

    vars_in_ckpt = {}
    for variable_name, variable in sorted(variable_names_map.items()):
        # 是否存在对应变量
        if variable_name in ckpt_vars_to_shape_map:
            # 对应变量的shape是否一致
            if ckpt_vars_to_shape_map[variable_name] == variable.shape.as_list():
                vars_in_ckpt[variable_name] = variable
            else:
                logging.warning('Variable [%s] is available in checkpoint, but has an '
                                'incompatible shape with model variable. Checkpoint '
                                'shape: [%s], model variable shape: [%s]. This '
                                'variable will not be initialized from the checkpoint.',
                                variable_name, ckpt_vars_to_shape_map[variable_name],
                                variable.shape.as_list())
        else:
            logging.warning('Variable [%s] is not available in checkpoint', variable_name)
    # 与输入类型保持一致 输出list or dict
    if isinstance(variables, list):
        return vars_in_ckpt.values()
    return vars_in_ckpt


def restore_map(include_scope_list=None):
    """Returns a map of variables to load from a foreign checkpoint.

    See parent class for details.

    Args:


    Returns:
      A dict mapping variable names (to load from a checkpoint) to variables in
      the model graph.
    Raises:
      ValueError: if fine_tune_checkpoint_type is neither `classification`
        nor `detection`.
    """

    variables_to_restore = tf.global_variables()
    variables_to_restore.append(tf.train.get_or_create_global_step())
    include_patterns = include_scope_list
    feature_extractor_variables = filter_variables(variables_to_restore, include_patterns=include_patterns)
    return {var.op.name: var for var in feature_extractor_variables}


def filter_variables(var_list, include_patterns=None, exclude_patterns=None, reg_search=True):
    """Filter a list of variables using regular expressions.

    First includes variables according to the list of include_patterns.
    Afterwards, eliminates variables according to the list of exclude_patterns.

    For example, one can obtain a list of variables with the weights of all
    convolutional layers (depending on the network definition) by:

    ```python
    variables = tf.contrib.framework.get_model_variables()
    conv_weight_variables = tf.contrib.framework.filter_variables(
        variables,
        include_patterns=['Conv'],
        exclude_patterns=['biases', 'Logits'])
    ```

    Args:
      var_list: list of variables.
      include_patterns: list of regular expressions to include. Defaults to None,
          which means all variables are selected according to the include rules.
          A variable is included if it matches any of the include_patterns.
      exclude_patterns: list of regular expressions to exclude. Defaults to None,
          which means all variables are selected according to the exclude rules.
          A variable is excluded if it matches any of the exclude_patterns.
      reg_search: boolean. If True (default), performs re.search to find matches
          (i.e. pattern can match any substring of the variable name). If False,
          performs re.match (i.e. regexp should match from the beginning of the
          variable name).

    Returns:
      filtered list of variables.
    """
    if reg_search:
        reg_exp_func = re.search
    else:
        reg_exp_func = re.match

    # First include variables.
    if include_patterns is None:
        included_variables = list(var_list)
    else:
        included_variables = []
        for var in var_list:
            if any(reg_exp_func(ptrn, var.name) for ptrn in include_patterns):
                included_variables.append(var)

    # Afterwards, exclude variables.
    if exclude_patterns is None:
        filtered_variables = included_variables
    else:
        filtered_variables = []
        for var in included_variables:
            if not any(reg_exp_func(ptrn, var.name) for ptrn in exclude_patterns):
                filtered_variables.append(var)

    return filtered_variables
