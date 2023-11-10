import os
import json
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import ConvNet, FireNet_v1, FireNet_v2, AlexNet_v1

# 配置信息
# 数据信息
img_height = 96 # 输入图片大小
img_width = 96
dataset = 'ForestFire_dataset'
batch_size = 32
classes = 2
# 测试信息
binary_class = False
# 模型信息
selectedmodel = FireNet_v1
# 文件保存地址信息
saveh5 = 'FireNet_v1_on_ForestFire.h5'

def main():
    # load image
    # 构造数据集路径
    data_root = os.getcwd()
    # data_root = os.path.abspath(os.path.join(os.getcwd(), '..'))  
    image_path = os.path.join(data_root, 'Dataset', dataset) 
    test_data_path = os.path.join(image_path, 'test') 

    # 构造用于test的Dataset对象
    train_ds = tf.keras.utils.image_dataset_from_directory(test_data_path,
                                                           label_mode="int",
                                                           image_size=(img_height, img_width),
                                                           batch_size=batch_size)
    if binary_class:
         num_classes = 1
    else:
         num_classes = classes

    # reload model
    model_path = "./save_weights/" + saveh5
    assert os.path.exists(model_path), "file: '{}' dose not exist.".format(saveh5)
    test_model = tf.keras.models.load_model(model_path)

    # test
    loss, acc = test_model.evaluate(train_ds)
    print("Loss on test set: {:.3f}".format(loss))
    print("Accuracy on test set: {:.3f}".format(acc))

if __name__ == '__main__':
    main()
