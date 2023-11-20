import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 数据信息
classes = 2
# 文件保存地址信息
saveh5 = '_on_FLAME.h5'
output_figure_dir = 'FLAME_output_figure'


# plot the confusion matrix
def show_confusion_matrix(cm, labels, confusion_matrix_fig):
     plt.figure(figsize=(7, 5))
     sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True, fmt='g')
     plt.xlabel('Prediction')
     plt.ylabel('Label')
     plt.savefig("figure/" + output_figure_dir + '/' + confusion_matrix_fig)


def calculate_classification_metrics(cm):
    tp = cm[0][0].numpy()
    fp = cm[1][0].numpy()
    fn = cm[0][1].numpy()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


def test_on_FLAME(selectedmodel,
                  batch_size=32,  
                  img_height=224, 
                  img_width=224,
                  binary_class=False):
    # load image
    # construct the dataset path
    data_root = os.getcwd()
    test_data_path = os.path.join(data_root, "Dataset/FLAME/test")

    # 构造文件存储地址
    dataset_saveh5 = str(img_height) + "_" + selectedmodel.__name__ + saveh5
    dataset_confusion_matrix_fig = str(img_height) + "_" + selectedmodel.__name__ + '_confusion_matrix'

    # construct the Dataset object for test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(test_data_path,
                                                          label_mode="int",
                                                          shuffle=False,
                                                          image_size=(img_height, img_width),
                                                          batch_size=batch_size)
    label_names = np.array(test_ds.class_names)

    # reload model
    model_path = "./save_weights/" + dataset_saveh5
    assert os.path.exists(model_path), "file: '{}' dose not exist.".format(dataset_saveh5)
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
    show_confusion_matrix(confusion_mtx, label_names, dataset_confusion_matrix_fig)

    # recall and precision on test dataset
    precision, recall = calculate_classification_metrics(confusion_mtx)
    print("Precision on test set: {:.3f}".format(precision))
    print("Recall on test set: {:.3f}".format(recall))


