#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""作者：少卿de少羽
链接：https: // zhuanlan.zhihu.com / p / 41589754
来源：知乎
著作权归作者所有"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras import backend as K


def visualize_model_output(model, image_path, layer_id, weights_path, num_filter=8):
    image = proprecess_image(image_path)
    output_model = get_model_by_layers(model, layer_id, weights_path)
    print("======output model summary ======")
    print(output_model.summary())
    result = output_model.predict(image)

    for i in range(num_filter):
        plt.subplot(2, 4, i+1)
        plt.imshow(result[0, :, :, i])
        plt.title(layer_id)
    plt.show()


inception_model = InceptionV3(include_top=True, weights=None, input_shape=(224, 224, 3), classes=5)

# layers name
for i, layer in enumerate(inception_model.layers):
    print(i, layer.name)

# print(inception_model.input_layers)
image_path = '/home/jiangmingchao/Gan_tensorflow/flower_dataset/train_dataset/daisy/105806915_a9c13e2106_n.jpg'
weights_path = 'flower_inceptionv3.h5'
conv_layers_id = [1, 4, 7, 11, 14, 18, 155, 187, 232, 290]
for layer_id in conv_layers_id:
    visualize_model_output(inception_model, image_path, layer_id, weights_path, num_filter=8)


def visualize_heat_map_on_image(model, image_path):
    """
    热力图可以反映出图片上特征的ROI区域，也可以绘制反映分布范围的变换等。
    这里的热力图实现用的方法叫做Grad-CAM(加权梯度类激活映射)
    取最终的卷积层的特征图，将该特征中的每个通道通过与该通道相关的类的梯度进行加权。
    :param model:
    :param image_path:
    :return:
    """
    image = proprecess_image(image_path)
    preds = model.predict(image)
    class_idx = np.argmax(preds[0])
    print(model.output)
    class_output = model.output[:, class_idx]
    # get the conv feature from last convolution
    last_conv_layer = model.get_layer(name='conv2d_188')

    # calculate grads
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=[0, 1, 2])
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    # pool grad
    pooled_grads_value, conv_layer_output_value = iterate([image])
    # for i in range()
    for i in range(192):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    original_image = cv2.imread(image_path)
    # original_image = cv2.resize(original_image, (224, 224))
    heatmap_image = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_image = np.uint8(255 * heatmap_image)

    heatmap_image = cv2.applyColorMap(heatmap_image, cv2.COLORMAP_HSV)
    print(heatmap_image.shape)
    print(original_image.shape)

    superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap_image, 0.4, 0)
    plt.imshow(superimposed_img)
    plt.show()


def visulaize_kernel_output(model, image_path, layer_id, weights_path):
    image = proprecess_image(image_path)
    img_shape = image.shape
    layers_dict = {}
    for index, layer in enumerate(model.layers):
        layers_dict[index] = layer.name

    def deprocess_image(x):
        x -= x.mean()
        x /= (x.std() + K.epsilon())
        x *= 0.1

        x += 0.5
        x = np.clip(x, 0, 1)

        x *= 255
        if K.image_data_format() == 'channels_first':
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def normalize(x):
        return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

    conv_model = get_model_by_layers(model, layer_id, weights_path)
    print(conv_model.summary())

    input_img = conv_model.input
    kept_filters = []
    for filter_index in range(32):
        print('Processing filter %d' % filter_index)
        start_time = time.time()

        layer_output = conv_model.get_layer(name=layers_dict[layer_id]).output
        if K.image_data_format() == 'channel_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        grads = K.gradients(loss, input_img)[0]

        grads = normalize(grads)

        iterate = K.function([input_img], [loss, grads])

        step = 1

        if K.image_data_format() == 'channels_first':
            input_img_data = np.random.random((img_shape[0], img_shape[3], img_shape[1], img_shape[2]))
        else:
            input_img_data = np.random.random((img_shape[0], img_shape[1], img_shape[2], img_shape[3]))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            print('current loss value:', loss_value)
            if loss_value <= 0.:
                break

        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

    print('filters number: ', len(kept_filters))
    n = 3

    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[: n * n]

    margin = 5

    width = n * image.shape[1] + (n - 1) * margin
    height = n * image.shape[2] + (n - 1) * margin

    stritched_filters = np.zeros((width, height, 3))

    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            stritched_filters[(image.shape[1] + margin) * i: (image.shape[1] + margin) * i + image.shape[1],
            (image.shape[2] + margin) * j: (image.shape[2] + margin) * j + image.shape[2], :] = img
    plt.imshow(stritched_filters)
    plt.show()