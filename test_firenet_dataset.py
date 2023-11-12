import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import ConvNet, FireNet_v1, FireNet_v2, AlexNet_v1

# 配置信息
# 数据信息
img_height = 64 # 输入图片大小
img_width = 64
dataset = 'FireNet'
batch_size = 32
classes = 2
# 模型信息
selectedmodel = ConvNet # 模型修改
# 文件保存地址信息
saveh5 = 'ConvNet_on_FireNet_64.h5' # 图片大小与模型修改
output_figure_dir = 'FireNet_dataset_output_figure'
confusion_matrix_dir = 'ConvNet_confusion_matrix_64' # 图片大小与模型修改


# plot the confusion matrix
def show_confusion_matrix(cm, labels):
     plt.figure(figsize=(7, 5))
     sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True, fmt='g')
     plt.xlabel('Prediction')
     plt.ylabel('Label')
     plt.savefig(output_figure_dir + '/' + confusion_matrix_dir)


def calculate_classification_metrics(cm):
    tp = cm[0][0].numpy()
    fp = cm[1][0].numpy()
    fn = cm[0][1].numpy()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


def main():
    # load image
    # construct the dataset path
    data_root = os.getcwd()
    # data_root = os.path.abspath(os.path.join(os.getcwd(), '..'))  
    image_path = os.path.join(data_root, 'Dataset', dataset) 
    test_data_path = os.path.join(image_path, 'test') 

    # construct the Dataset object for test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(test_data_path,
                                                           label_mode="int",
                                                           shuffle=False,
                                                           image_size=(img_height, img_width),
                                                           batch_size=batch_size)
    label_names = np.array(test_ds.class_names)

    # reload model
    model_path = "./save_weights/" + saveh5
    assert os.path.exists(model_path), "file: '{}' dose not exist.".format(saveh5)
    test_model = tf.keras.models.load_model(model_path)

    # acc and loss on test dataset
    loss, acc = test_model.evaluate(test_ds)
    print("Loss on test set: {:.3f}".format(loss))
    print("Accuracy on test set: {:.3f}".format(acc))

    # confusion matrix
    pred = tf.argmax(test_model.predict(test_ds), axis=1)
    true = tf.concat(list(test_ds.map(lambda x,y: y)), axis=0)
    confusion_mtx = tf.math.confusion_matrix(true, pred)
    # print(confusion_mtx)
    show_confusion_matrix(confusion_mtx, label_names)

    # recall and precision on test dataset
    precision, recall = calculate_classification_metrics(confusion_mtx)
    print("Precision on test set: {:.3f}".format(precision))
    print("Recall on test set: {:.3f}".format(recall))


if __name__ == '__main__':
    main()
