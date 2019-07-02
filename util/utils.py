"""General utility functions"""
import os
import cv2
import json
import shutil
import logging
import tensorflow as tf
from util.label_map_util import get_label_map_dict


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def get_ab_path(root_path):
    # 得到当前文件夹内的所有子文件
    image_paths = []
    # c_folder当前文件夹完全路径，subfolder当前文件夹的子文件夹，files当前文件夹的子文件
    for c_folder, subfolder, files in os.walk(root_path):
        for file in files:
            if file.endswith('.jpg'):
                image = os.path.join(c_folder, file)
                # print(image)
                image_paths.append(image)
    return image_paths


def get_data(train_data_dir, label_map_path):
    label_map = get_label_map_dict(label_map_path)  # lable_map[name:id] id begin with 1
    image_path_list = []
    image_label_list = []
    train_data_num = 0
    for cur_folder, sub_folders, sub_files in os.walk(train_data_dir):
        for file in sub_files:
            if file.endswith('jpg'):
                train_data_num += 1
                image_path_list.append(os.path.join(cur_folder, file))
                image_label_list.append(label_map[os.path.split(cur_folder)[-1]])
    print('train_image_num:', train_data_num)
    data_tuple = (image_path_list, image_label_list)
    return data_tuple


def get_dict(root_path):
    # 当前root_path的 子文件夹名 作为key，子文件夹内的 子文件完全路径 作为value
    truth_dict = {}
    for c_folder, subfolder, files in os.walk(root_path):
        image_list = []
        for file in files:
            if file.endswith('.jpg'):
                image = os.path.join(c_folder, file)
                image_list.append(image)
        label = os.path.split(c_folder)[-1]
        truth_dict[label] = image_list
    return truth_dict


def image_size(img_dir):
    import os
    import cv2

    small_num = 0
    mid_num = 0
    large_num = 0
    for cur_folder, sub_folders, sub_files in os.walk(img_dir):
        for file in sub_files:
            if file.endswith('jpg'):
                img = cv2.imread(os.path.join(cur_folder, file))
                pixel_areas = img.shape[0] * img.shape[1]
                if pixel_areas < 3600:
                    small_num += 1
                elif pixel_areas < 8100:
                    mid_num += 1
                else:
                    large_num += 1

    print('small num:', small_num)
    print('mid_num:', mid_num)
    print('large_num:', large_num)


def get_variable_to_restore(vars_dict, checkpoint_path, include_global_step=False):
    """

    :param vars_dict:
    :param checkpoint_path:
    :param include_global_step:
    :return:
    """
    if isinstance(vars_dict, list):
        vars_map = {v.op.name: v for v in vars_dict}
    elif isinstance(vars_dict, dict):
        vars_map = vars_dict
    else:
        raise ValueError("`vars_dict` is excepted to be a list or dict")

    ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
    ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
    if not include_global_step:
        ckpt_vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)
    var_in_ckpt={}
    for var_name, var in sorted(vars_map):
        if var_name in ckpt_vars_to_shape_map:
            if var.shape.as_list() == ckpt_vars_to_shape_map[var_name]:
                var_in_ckpt[var_name] = var
            else:
                logging.WARNING('Variable [%s] is available in checkpoint, but has an '
                                'incompatible shape with model variable. Checkpoint '
                                'shape: [%s], model variable shape: [%s]. This '
                                'variable will not be initialized from the checkpoint.',
                                var_name, ckpt_vars_to_shape_map[var_name], var.shape.as_list())
        else:
            logging.warning('Variable [%s] is not available in checkpoint', var_name)

    if isinstance(vars_dict, list):
        return var_in_ckpt.values()
    return var_in_ckpt
